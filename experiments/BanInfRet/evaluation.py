import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import torch
from typing import List
import tempfile
import shutil

# Define the CustomEmbeddings class
class CustomEmbeddings:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

    def embed_documents(self, texts):
        """Embeds a list of documents (texts) into embeddings."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        """Embeds a single query into an embedding."""
        return self.model.encode([text], convert_to_numpy=True).tolist()[0]

def create_vector_store_from_json(json_path: str, vector_db_path: str, embedding_model) -> None:
    """
    Create a vector store from a JSON file containing passages, storing 'annotation_id' as metadata.

    Args:
        json_path (str): Path to the JSON file.
        vector_db_path (str): Path where the vector store will be saved.
        embedding_model: Embedding model to use for vector store creation.
    """
    # Load JSON entries
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure output directory exists
    os.makedirs(vector_db_path, exist_ok=True)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )

    # Initialize Chroma vector store
    vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embedding_model)

    for entry in tqdm(data, desc="Creating vector store"):
        try:
            text = entry["text"]
            annotation_id = entry["annotation_id"]

            # Wrap with Document and attach metadata
            documents = [Document(page_content=text, metadata={"annotation_id": annotation_id})]
            split_docs = text_splitter.split_documents(documents)

            vectorstore.add_documents(split_docs)
        except Exception as e:
            print(f"Error processing entry {annotation_id}: {e}")
    vectorstore.persist()
    print(f"Vector store created and persisted at: {vector_db_path}")

def evaluate_multiple_embeddings(json_path: str,
                                embedding_models: dict,
                                top_k_list: List[int] = [1, 3, 5],
                                thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
                                plot_file: str = "pr_curve.png",
                                metrics_file: str = "metrics.json") -> None:
    """
    Create temporary vector stores for each embedding model and evaluate them across thresholds,
    save PR + F1 curve for all thresholds to file, compute metrics, and save metrics to a JSON file.

    Args:
        json_path (str): Path to JSON file with QAs.
        embedding_models (dict): Mapping of model_name -> embedding_model_object.
        top_k_list (List[int]): List of K values (e.g. [1,3,5]).
        thresholds (List[float]): List of similarity score thresholds (e.g. [0.5, 0.6, 0.7, 0.8, 0.9]).
        plot_file (str): Path to save PR plot.
        metrics_file (str): Path to save metrics JSON file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = [entry for entry in data if entry.get("question")]
    total_questions = sum(len(entry.get("question", [])) for entry in data)

    # Initialize output metrics dictionary
    output_metrics = {}

    plt.figure(figsize=(12, 8))  # Increased figure size for clarity

    # Define markers for different thresholds to distinguish them
    markers = ['o', 's', '^', 'D', '*']  # One marker per threshold

    for model_name, emb_model in embedding_models.items():
        print(f"\nProcessing: {model_name}")

        # Create a temporary directory for the vector store
        temp_vector_db_path = tempfile.mkdtemp()
        try:
            # Create vector store for this embedding model
            create_vector_store_from_json(json_path, temp_vector_db_path, emb_model)

            print(f"\nEvaluating: {model_name}")
            vectorstore = Chroma(persist_directory=temp_vector_db_path, embedding_function=emb_model)

            # Initialize metrics for each threshold and k
            metrics = {
                t: {
                    'recall_at_k': {k: 0 for k in top_k_list},
                    'precision_at_k': {k: [] for k in top_k_list},
                    'mrr_at_k': {k: 0.0 for k in top_k_list},
                    'f1_at_k': {k: [] for k in top_k_list},
                    'top1_similarity_scores': []
                } for t in thresholds
            }

            for entry in tqdm(data, desc="Evaluating"):
                annotation_id = entry["annotation_id"]
                questions = entry.get("question", [])

                for question in questions:
                    # Retrieve documents with the lowest threshold to get maximum documents
                    retrieved_docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
                        question, k=max(top_k_list), score_threshold=min(thresholds)
                    )

                    # Evaluate metrics for each threshold
                    for t in thresholds:
                        # Filter documents by threshold
                        valid_docs = [(doc, score) for doc, score in retrieved_docs_with_scores if score >= t]
                        retrieved_ids = [doc.metadata.get("annotation_id") for doc, _ in valid_docs]

                        # Record top-1 similarity score if available
                        if valid_docs:
                            metrics[t]['top1_similarity_scores'].append(valid_docs[0][1])

                        # Compute other metrics
                        for k in top_k_list:
                            top_k_ids = retrieved_ids[:k]
                            if annotation_id in top_k_ids:
                                metrics[t]['recall_at_k'][k] += 1
                                rank = top_k_ids.index(annotation_id) + 1
                                prec = 1.0 / rank
                                metrics[t]['precision_at_k'][k].append(prec)
                                metrics[t]['mrr_at_k'][k] += prec
                                f1 = 2 * prec * 1 / (prec + 1)
                                metrics[t]['f1_at_k'][k].append(f1)
                            else:
                                metrics[t]['precision_at_k'][k].append(0.0)
                                metrics[t]['f1_at_k'][k].append(0.0)

            # Compute and store metrics for each threshold
            output_metrics[model_name] = {}
            for idx, t in enumerate(thresholds):
                print(f"\n  Threshold: {t}")
                recalls = [metrics[t]['recall_at_k'][k] / total_questions for k in top_k_list]
                precisions = [np.mean(metrics[t]['precision_at_k'][k]) for k in top_k_list]
                f1_scores = [np.mean(metrics[t]['f1_at_k'][k]) for k in top_k_list]
                mrrs = [metrics[t]['mrr_at_k'][k] / total_questions for k in top_k_list]
                avg_top1_similarity = np.mean(metrics[t]['top1_similarity_scores']) if metrics[t]['top1_similarity_scores'] else 0.0

                # Store metrics in output dictionary
                output_metrics[model_name][str(t)] = {
                    'recall_at_k': {str(k): r for k, r in zip(top_k_list, recalls)},
                    'precision_at_k': {str(k): p for k, p in zip(top_k_list, precisions)},
                    'mrr_at_k': {str(k): m for k, m in zip(top_k_list, mrrs)},
                    'f1_at_k': {str(k): f for k, f in zip(top_k_list, f1_scores)},
                    'avg_top1_similarity': avg_top1_similarity
                }

                # Print metrics
                for k, r, p, f1, m in zip(top_k_list, recalls, precisions, f1_scores, mrrs):
                    print(f"    Recall@{k}: {r:.4f} | Precision@{k}: {p:.4f} | MRR@{k}: {m:.4f} | f1@{k}: {f1:.4f}")
                print(f"    Avg Similarity Score (Top-1): {avg_top1_similarity:.4f}")

                # Plot PR curve for this threshold
                plt.plot(recalls, precisions, marker=markers[idx % len(markers)], 
                         label=f"{model_name} (Threshold={t})", linestyle='-')
                for i, k in enumerate(top_k_list):
                    plt.annotate(f"{model_name}@{k}\nT={t}\nF1={f1_scores[i]:.2f}", 
                                 (recalls[i], precisions[i]),
                                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        finally:
            # Clean up temporary vector store
            shutil.rmtree(temp_vector_db_path, ignore_errors=True)
            print(f"Cleaned up temporary vector store for {model_name}")

    # Save metrics to JSON file
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(output_metrics, f, indent=2)
    print(f"\n✅ Metrics saved to: {metrics_file}")

    plt.title("Precision-Recall Curve with F1 (All Thresholds)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_file, bbox_inches='tight')
    print(f"✅ PR + F1 curve saved to: {plot_file}")

if __name__ == "__main__":
    # Define paths
    json_path = "data/json___unvalidated__annotation_data_d5f877c8-963e-4a96-925d-6e2c28417895.json"

    # Initialize embedding models
    embedding_models = {
        "base": CustomEmbeddings("l3cube-pune/bengali-sentence-similarity-sbert"),
        "d3_version": CustomEmbeddings("experiments/BanInfRet/d3_baninfret"),
    }

    # Evaluate and plot
    evaluate_multiple_embeddings(json_path, embedding_models)
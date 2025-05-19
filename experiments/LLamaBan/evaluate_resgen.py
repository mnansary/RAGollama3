import json
import re
from bnlp.tokenizer.nltk import NLTKTokenizer
from bnlp.tokenizer.basic import BasicTokenizer
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Dict
import nltk
nltk.download('wordnet')
nltk.download('punkt_tab')

class CustomEmbeddings:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True).tolist()[0]

def calculate_hallucination_score(data: List[Dict[str, any]]) -> float:
    """
    Calculate hallucination score for Bengali text.
    """
    try:
        sent_tokenizer = NLTKTokenizer()
        word_tokenizer = BasicTokenizer()

        total_claims = 0
        hallucinated_claims = 0

        for item in data:
            context = item.get('context', '')
            prediction = item.get('predictions', [''])[0]
            answer = item.get('answer', [''])[0]

            if not prediction or not answer or not context:
                continue

            pred_sentences = sent_tokenizer.sentence_tokenize(prediction)
            answer_sentences = sent_tokenizer.sentence_tokenize(answer)
            context_sentences = sent_tokenizer.sentence_tokenize(context)

            context_words = set(word_tokenizer.tokenize(' '.join(context_sentences)))
            answer_words = set(word_tokenizer.tokenize(' '.join(answer_sentences)))

            for pred_sent in pred_sentences:
                total_claims += 1
                pred_words = set(word_tokenizer.tokenize(pred_sent))

                is_supported = False
                for ctx_sent in context_sentences + answer_sentences:
                    ctx_words = set(word_tokenizer.tokenize(ctx_sent))
                    common_words = pred_words.intersection(ctx_words)
                    if len(common_words) >= min(len(pred_words) * 0.5, len(ctx_words) * 0.5):
                        is_supported = True
                        break

                contradicts = False
                for ans_sent in answer_sentences:
                    ans_words = set(word_tokenizer.tokenize(ans_sent))
                    if pred_words.difference(ans_words).difference(context_words):
                        if not is_supported:
                            contradicts = True

                if not is_supported or contradicts:
                    hallucinated_claims += 1

        return hallucinated_claims / total_claims if total_claims > 0 else 0.0
    except Exception as e:
        print(f"Error in hallucination score: {e}")
        return 0.0

    

def calculate_rouge(data: List[Dict[str, any]]) -> Dict[str, float]:
    """
    Calculate ROUGE-1, ROUGE-2, ROUGE-L scores.
    """
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
        total_examples = 0

        for item in data:
            prediction = item.get('predictions', [''])[0]
            answer = item.get('answer', [''])[0]

            if not prediction or not answer:
                continue

            scores = scorer.score(answer, prediction)
            rouge1 += scores['rouge1'].fmeasure
            rouge2 += scores['rouge2'].fmeasure
            rougeL += scores['rougeL'].fmeasure
            total_examples += 1

        return {
            'rouge1': rouge1 / total_examples if total_examples > 0 else 0.0,
            'rouge2': rouge2 / total_examples if total_examples > 0 else 0.0,
            'rougeL': rougeL / total_examples if total_examples > 0 else 0.0
        }
    except Exception as e:
        print(f"Error in ROUGE calculation: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

def calculate_meteor_bengali(data: List[Dict[str, any]]) -> float:
    """
    Calculate METEOR score for Bengali text using bnlp tokenizer.
    """
    try:
        tokenizer = BasicTokenizer()
        total_meteor = 0.0
        total_examples = 0

        for item in data:
            prediction = item.get('predictions', [''])[0]
            answer = item.get('answer', [''])[0]

            if not prediction or not answer:
                continue

            pred_tokens = tokenizer.tokenize(prediction)
            ans_tokens = tokenizer.tokenize(answer)
            score = meteor_score([ans_tokens], pred_tokens)
            total_meteor += score
            total_examples += 1

        return total_meteor / total_examples if total_examples > 0 else 0.0
    except Exception as e:
        print(f"Error in METEOR calculation: {e}")
        return 0.0


def calculate_cosine_similarity(data: List[Dict[str, any]], embedder: CustomEmbeddings) -> Dict[str, float]:
    """
    Calculate average cosine similarity between prediction and ground truth, and prediction and context.
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        gt_sim, ctx_sim = 0.0, 0.0
        total_examples = 0

        for item in data:
            prediction = item.get('predictions', [''])[0]
            answer = item.get('answer', [''])[0]
            context = item.get('context', '')

            if not prediction or not answer or not context:
                continue

            pred_emb = embedder.embed_query(prediction)
            ans_emb = embedder.embed_query(answer)
            ctx_emb = embedder.embed_query(context)

            gt_sim += cosine_similarity([pred_emb], [ans_emb])[0][0]
            ctx_sim += cosine_similarity([pred_emb], [ctx_emb])[0][0]
            total_examples += 1

        return {
            'gt_to_pred': gt_sim / total_examples if total_examples > 0 else 0.0,
            'context_to_pred': ctx_sim / total_examples if total_examples > 0 else 0.0
        }
    except Exception as e:
        print(f"Error in cosine similarity calculation: {e}")
        return {'gt_to_pred': 0.0, 'context_to_pred': 0.0}

def evaluate_models(model_files: Dict[str, str], embedder: CustomEmbeddings):
    """
    Evaluate multiple models and save metrics to JSON files.
    Args:
        model_files: Dict mapping model names to result JSON file paths.
        embedder: CustomEmbeddings instance for BERTScore and cosine similarity.
    """
    for model_name, file_path in model_files.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            metrics = {
                'hallucination_score': calculate_hallucination_score(data),
                'rouge': calculate_rouge(data),
                'meteor': calculate_meteor_bengali(data),
                'cosine_similarity': calculate_cosine_similarity(data, embedder)
            }
            print(f"Metrics for {model_name}: {metrics}")
            output_file = f"{model_name}_metrics.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            print(f"Saved metrics for {model_name} to {output_file}")
        except Exception as e:
            print(f"Error processing {model_name}: {e}")


if __name__ == "__main__":
    # Load Bengali embedding model
    print("Loading Bengali embedding model...")
    embedder = CustomEmbeddings("l3cube-pune/bengali-sentence-similarity-sbert")

    # Example model files dictionary
    model_files = {
        "llama3.3": "data/ministry_1k_llama3.3.json",
        "llama_d3": "data/ministry_1k_llama_d3.json"
    }

    # Evaluate models
    evaluate_models(model_files, embedder)
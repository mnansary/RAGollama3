import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma 
from langchain.schema import Document
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3" 
# IMPORTANT: Use the correct path to your CSV file
CSV_FILE_PATH = "archive/aggregated_source.csv" 
VECTOR_DB_PATH = "prototype"

# --- Jina V3 Specific Embedding Class (No changes here) ---
class JinaV3Embeddings:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"JinaV3Embeddings using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        print(f"Successfully loaded model: {model_name}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        processed_texts = [str(text) if text is not None else "" for text in texts]
        embeddings = self.model.encode(
            processed_texts, 
            task='retrieval.passage', 
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            [str(text) if text is not None else ""], 
            task='retrieval.query', 
            convert_to_numpy=True
        )
        return embedding.tolist()[0]

# --- Core Function to Create the Vector Store (with Batching and Final Count) ---
def create_vector_store_from_csv(
    csv_path: str,
    text_column: str,
    metadata_columns: list,
    vector_db_path: str,
    embedding_func: JinaV3Embeddings,
    batch_size: int = 2  # Adjust batch size based on your GPU memory if needed
) -> None:
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV from '{csv_path}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file was not found at '{csv_path}'")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    documents_to_add = []
    print("Preparing documents from DataFrame...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        passage_text = row.get(text_column)
        if pd.isna(passage_text) or not str(passage_text).strip():
            continue
        metadata = {col: row.get(col, "") for col in metadata_columns if pd.notna(row.get(col))}
        doc = Document(page_content=str(passage_text), metadata=metadata)
        documents_to_add.append(doc)

    if not documents_to_add:
        print("No valid documents found in the CSV to add to the vector store.")
        return

    # 1. Initialize an empty Chroma vector store first
    print(f"\nInitializing Chroma vector store at: {vector_db_path}")
    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embedding_func
    )

    # 2. Add documents in batches
    print(f"Adding {len(documents_to_add)} documents to Chroma in batches of {batch_size}...")
    for i in tqdm(range(0, len(documents_to_add), batch_size), desc="Adding documents to Chroma"):
        batch = documents_to_add[i:i + batch_size]
        vectorstore.add_documents(documents=batch)

    # 3. Persist the store
    print("Persisting the vector store...")
    vectorstore.persist()
    
    # 4. *** NEW: Get and print the final document count from the collection ***
    try:
        total_docs_in_db = vectorstore._collection.count()
        print(f"\n✅ Success! Vector store contains {total_docs_in_db} documents.")
    except Exception as e:
        print(f"Could not retrieve document count. Error: {e}")

    print(f"Vector store is persisted at: {vector_db_path}")


# --- Main Execution Block (No changes here) ---
if __name__ == "__main__":
    embedding_function = JinaV3Embeddings(EMBEDDING_MODEL_NAME)

    # Make sure these column names match your CSV file
    text_col = 'text' 
    metadata_cols = ['url',"site_name"] 
    
    create_vector_store_from_csv(
        csv_path=CSV_FILE_PATH,
        text_column=text_col,
        metadata_columns=metadata_cols,
        vector_db_path=VECTOR_DB_PATH,
        embedding_func=embedding_function
    )
    
    print("\nProcess finished.")
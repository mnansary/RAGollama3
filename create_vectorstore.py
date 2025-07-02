import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.schema import Document
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# --- Jina V3 Specific Embedding Class ---
class JinaV3Embeddings:
    """
    Custom LangChain-compatible embedding class for jina-embeddings-v3.
    It correctly handles the 'task' parameter for passage and query embeddings.
    """
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"JinaV3Embeddings using device: {self.device}")
        
        # Load the Jina V3 model using SentenceTransformer
        # trust_remote_code=True is required for this model
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        print(f"Successfully loaded model: {model_name}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of documents (passages) for storage.
        Uses the 'retrieval.passage' task as recommended by Jina V3 docs.
        """
        # Ensure texts are strings and handle potential None or non-string types
        processed_texts = [str(text) if text is not None else "" for text in texts]
        
        # Use the 'retrieval.passage' task for storing documents
        embeddings = self.model.encode(
            processed_texts, 
            task='retrieval.passage', 
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embeds a single query for searching.
        Uses the 'retrieval.query' task as recommended by Jina V3 docs.
        """
        # Use the 'retrieval.query' task for search queries
        embedding = self.model.encode(
            [str(text) if text is not None else ""], 
            task='retrieval.query', 
            convert_to_numpy=True
        )
        return embedding.tolist()[0]

# --- Core Function to Create the Vector Store ---
def create_vector_store_from_csv(
    csv_path: str,
    text_column: str,
    metadata_columns: list,
    vector_db_path: str,
    embedding_func: JinaV3Embeddings
) -> None:
    """
    Reads a CSV, creates documents, and builds a Chroma vector store.
    Each row in the CSV becomes a separate document in the store.

    Args:
        csv_path (str): Path to the input CSV file.
        text_column (str): The name of the column containing the text to be embedded.
        metadata_columns (list): A list of column names to include as metadata.
        vector_db_path (str): Path to the directory for the Chroma store.
        embedding_func (JinaV3Embeddings): The embedding function instance.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV from '{csv_path}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file was not found at '{csv_path}'")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Prepare documents for Chroma
    documents_to_add = []
    print("Preparing documents from DataFrame...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        passage_text = row.get(text_column)

        # Skip rows where the main text content is missing or empty
        if pd.isna(passage_text) or not str(passage_text).strip():
            continue

        metadata = {col: row.get(col, "") for col in metadata_columns if pd.notna(row.get(col))}
        
        doc = Document(page_content=str(passage_text), metadata=metadata)
        documents_to_add.append(doc)

    if not documents_to_add:
        print("No valid documents found in the CSV to add to the vector store.")
        return

    # Create and persist the Chroma vector store
    print(f"\nInitializing Chroma vector store at: {vector_db_path}")
    print(f"Using the '{embedding_func.model.encode.__defaults__[0]}' task for embedding documents.") # A bit of a hack to see the default task

    vectorstore = Chroma.from_documents(
        documents=documents_to_add,
        embedding=embedding_func,
        persist_directory=vector_db_path
    )

    print(f"\nSuccessfully added {len(documents_to_add)} documents to Chroma.")
    print(f"Vector store persisted at: {vector_db_path}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    # Set the model name for Jina V3
    EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3" 
    # Path to your source CSV file
    CSV_FILE_PATH = "aggregated_source.csv" 
    # Path where the single Chroma vector store will be created
    VECTOR_DB_PATH = "prototype"

    # 1. Initialize the embedding model
    # This single instance will be used for creating the store.
    # When you build your retrieval app, you can use the same instance.
    # The class will automatically use the correct task ('retrieval.passage' vs 'retrieval.query').
    embedding_function = JinaV3Embeddings(EMBEDDING_MODEL_NAME)

    # 2. Define the columns from your CSV
    # The 'text' column will be embedded and stored as the main document content.
    # The 'url' column will be stored as metadata for each document.
    text_col = 'text'
    metadata_cols = ['url']
    
    # 3. Create the vector store
    create_vector_store_from_csv(
        csv_path=CSV_FILE_PATH,
        text_column=text_col,
        metadata_columns=metadata_cols,
        vector_db_path=VECTOR_DB_PATH,
        embedding_func=embedding_function
    )
    
    print("\nProcess finished.")
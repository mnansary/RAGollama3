#---------------------------------
# imports
#---------------------------------
import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from tqdm import tqdm
from embedding import CustomEmbeddings
from config import EMBEDDING_MODEL,VECTOR_DB_PATH,DATA_SOURCE
import warnings
warnings.filterwarnings("ignore")
#--------------------------------------
# global
#--------------------------------------
embedding_model=CustomEmbeddings(EMBEDDING_MODEL)

#--------------------------------------
# helper functions
#--------------------------------------
def create_vector_store(df: pd.DataFrame, 
                        vector_db_path: str) -> None:
    """
    Create a vector store from a dataframe with a single column, where each entry in the dataframe
    is treated as a separate document.

    Args:
        df (pd.DataFrame): DataFrame containing a single column with text data.
        vector_db_path (str): Path to the directory where the vector store will be persisted.
    """
    # Check if GPU is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create data directory structure
    os.makedirs(vector_db_path, exist_ok=True)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )

    
    # Initialize the vector store with GPU-based embeddings
    vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embedding_model)

    # Process each document in the dataframe
    for data in tqdm(df.iloc[:, 0]):
        documents = [Document(page_content=data)]
        all_splits = text_splitter.split_documents(documents)

        # Add the split documents to the vector store
        vectorstore.add_documents(all_splits)

    # Persist the vector store
    vectorstore.persist()
    print("Vector store created and persisted at:", vector_db_path)


if __name__=="__main__":
    df=pd.read_csv(DATA_SOURCE)
    create_vector_store(df,VECTOR_DB_PATH)
# your_main_retriever_file.py

from langchain_community.vectorstores import Chroma
from tqdm import tqdm
import warnings

# Import your JinaV3Embeddings class from its file
from .embedding import JinaV3Embeddings

warnings.filterwarnings("ignore")

class RetrieverService:
    def __init__(self, 
                 vector_db_path: str,
                 embedding_model_name: str = "jinaai/jina-embeddings-v3",
                 num_passages_to_retrieve: int = 3):
        """
        Initializes a simple, direct retriever service.

        Args:
            vector_db_path (str): Path to the Chroma vector store.
            embedding_model_name (str): The name of the embedding model to use.
            num_passages_to_retrieve (int): The number of relevant passages to retrieve.
        """
        print("Initializing RetrieverService...")
        self.embedding_model = JinaV3Embeddings(embedding_model_name)
        
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model
        )
        
        # We will use the number of passages directly in the search method
        self.num_passages_to_retrieve = num_passages_to_retrieve
        
        try:
            db_count = self.vectorstore._collection.count()
            print(f"✅ RetrieverService initialized successfully. Vector store at '{vector_db_path}' contains {db_count} documents.")
        except Exception as e:
            print(f"⚠️ Warning: Could not get count from vector store. It might be empty or improperly loaded: {e}")

    def retrieve(self, query: str) -> dict:
        """
        Retrieves relevant passages directly from the vector store based on the query.

        Args:
            query (str): The user's question.

        Returns:
            dict: A dictionary containing the original query and a list of retrieved passages.
        """
        print(f"\nPerforming direct retrieval for query: \"{query}\"")
        
        # Use similarity_search_with_score to get documents and their relevance scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query, 
            k=self.num_passages_to_retrieve
        )

        # Format the results
        retrieved_passages = []
        if not docs_with_scores:
            print("No relevant documents found.")
        else:
            for doc, score in docs_with_scores:
                retrieved_passages.append({
                    "text": doc.page_content,
                    "url": doc.metadata.get("url", "URL not found"),
                    "score": score,
                    "metadata": doc.metadata
                })
            print(f"Found {len(retrieved_passages)} relevant passages.")
            
        # Sort final passages by score (lower is better for distance metrics like L2)
        retrieved_passages.sort(key=lambda x: x["score"])

        return {
            "query": query,
            "retrieved_passages": retrieved_passages
        }

# --- Example Usage ---
if __name__ == "__main__":
    # Path to the vector store you created with create_vectorstore.py
    VECTOR_STORE_PATH = "prototype" 
    
    # 1. Initialize the service
    retriever_service = RetrieverService(
        vector_db_path=VECTOR_STORE_PATH,
        num_passages_to_retrieve=3 
    )
    
    # 2. Use the service to retrieve documents for a query
    user_query = "পদ্মা নদীর দৈর্ঘ্য কত?" # "What is the length of the Padma Bridge?"
    results = retriever_service.retrieve(user_query)
    
    # 3. Print the results
    import json
    print("\n--- Retrieval Results ---")
    print(json.dumps(results, indent=2, ensure_ascii=False))
# your_main_retriever_file.py
"""
Retriever Service Module.

This module defines the RetrieverService class, which acts as a dedicated interface
for interacting with the Chroma vector database. It encapsulates all the logic for
embedding queries and performing similarity searches, including advanced features like
dynamic 'k' retrieval and metadata filtering. This decouples the main chat application
from the specifics of the vector store implementation.
"""

from langchain_community.vectorstores import Chroma
from tqdm import tqdm
import warnings
from typing import Dict, Any, List

# Import custom components from the local 'core' package.
from .embedding import JinaV3Embeddings
from .config import EMBEDDING_MODEL, VECTOR_DB_PATH

# Suppress common warnings for a cleaner console output.
warnings.filterwarnings("ignore")


class RetrieverService:
    """
    A service class to handle all interactions with the vector database.

    This class is responsible for:
    1.  Initializing a connection to the Chroma vector store.
    2.  Using a specified embedding model (JinaV3) to convert text queries into vectors.
    3.  Providing a clean `retrieve` method to search for relevant documents based on a query,
        a specified number of results (k), and optional metadata filters.
    """

    def __init__(self, vector_db_path: str = VECTOR_DB_PATH, embedding_model: str = EMBEDDING_MODEL):
        """
        Initializes the RetrieverService.

        This constructor sets up the embedding model and loads the persisted Chroma
        vector store from disk. It also performs a quick health check by counting
        the documents in the store.

        Args:
            vector_db_path (str): The local file path to the directory where the
                                  Chroma database is stored.
            embedding_model (str): The model name or path for the JinaV3Embeddings
                                   to use.
        """
        print("Initializing RetrieverService...")
        # Initialize our custom embedding function wrapper.
        self.embedding_model = JinaV3Embeddings(embedding_model)

        # Load the existing vector store from the specified path.
        # The embedding function must match the one used to create the store.
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model
        )

        try:
            # A simple check to confirm the database is loaded and not empty.
            db_count = self.vectorstore._collection.count()
            print(f"✅ RetrieverService initialized successfully. Vector store at '{vector_db_path}' contains {db_count} documents.")
        except Exception as e:
            print(f"⚠️ Warning: Could not get document count from vector store: {e}")

    def retrieve(self, query: str, k: int = 3, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retrieves relevant passages with dynamic k and metadata filtering.

        This is the primary method for searching the vector store. It performs a
        similarity search and formats the results into a standardized dictionary.

        Args:
            query (str): The user's question or search term.
            k (int): The number of documents to retrieve for this specific query.
                     Defaults to 3.
            filters (dict, optional): A dictionary for metadata-based filtering.
                                      This allows for targeted searches within a
                                      specific category.
                                      Example: `{"topic": "জন্ম নিবন্ধন"}`

        Returns:
            Dict[str, Any]: A dictionary containing the original query and a list
                            of retrieved passages. Each passage in the list is a
                            dictionary with "text", "score", and "metadata".
                            Example:
                            {
                                "query": "Your question",
                                "retrieved_passages": [
                                    {
                                        "text": "Document content...",
                                        "score": 0.1234,
                                        "metadata": {"topic": "...", "url": "..."}
                                    }
                                ]
                            }
        """
        print(f"\nPerforming retrieval for query: \"{query}\" with k={k} and filters={filters}")

        # Perform the similarity search using the vector store.
        # The 'filter' parameter is the correct keyword for the LangChain Chroma wrapper.
        # This is a common point of confusion, as one might expect 'filters'.
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filters
        )

        retrieved_passages: List[Dict[str, Any]] = []
        if not docs_with_scores:
            print("No relevant documents found with the given criteria.")
        else:
            # Process the results into a cleaner, more standardized format.
            # This decouples the rest of the app from LangChain's 'Document' object.
            for doc, score in docs_with_scores:
                retrieved_passages.append({
                    "text": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata
                })
            print(f"Found {len(retrieved_passages)} relevant passages.")

        # Sort the results by score (lower is better for similarity).
        # While similarity_search_with_score usually returns sorted results,
        # this adds an extra layer of robustness.
        retrieved_passages.sort(key=lambda x: x["score"])

        return {
            "query": query,
            "retrieved_passages": retrieved_passages
        }


# --- Example Usage & Self-Test Block ---
if __name__ == "__main__":
    """
    This block allows the file to be run directly for testing purposes.
    It demonstrates how to use the RetrieverService and verifies its key features.
    """
    # 1. Initialize the service using configuration from config.py
    retriever_service = RetrieverService(
        vector_db_path=VECTOR_DB_PATH,
        embedding_model=EMBEDDING_MODEL
    )

    # 2. Define a list of diverse test cases to showcase different features
    test_cases = [
        {
            "description": "--- Test Case 1: Simple query with default k (k=3) ---",
            "query": "ড্রাইভিং লাইসেন্স করার পদ্ধতি কি?", # "What is the procedure for getting a driving license?"
            "params": {}
        },
        {
            "description": "--- Test Case 2: Query with a specific k value ---",
            "query": "পাসপোর্ট নবায়ন", # "Passport renewal"
            "params": {"k": 5} # Requesting the top 5 documents
        },
        {
            "description": "--- Test Case 3: Broad query narrowed down by a metadata filter ---",
            "query": "আবেদন প্রক্রিয়া", # "Application process"
            # This is powerful: it finds docs about "application process" but ONLY for the topic "জন্ম নিবন্ধন"
            "params": {"filters": {"topic": "জন্ম নিবন্ধন"}}
        },
        {
            "description": "--- Test Case 4: Query combining a specific k AND a filter ---",
            "query": "যোগাযোগের ঠিকানা", # "Contact address"
            # Gets just the single most relevant contact document for Bangladesh Railway
            "params": {"k": 1, "filters": {"topic": "বাংলাদেশ রেলওয়ে"}}
        },
        {
            "description": "--- Test Case 5: Query that is expected to find no relevant results ---",
            "query": "চাঁদে জমি কেনার নিয়ম", # "Rules for buying land on the moon"
            "params": {}
        }
    ]

    import json

    # 3. Loop through the test cases, execute them, and print the results
    for test in test_cases:
        print("\n" + "="*80)
        print(test["description"])
        print("="*80)

        # Unpack the query and parameters for the retrieve call
        query = test["query"]
        params = test["params"]

        # The **params syntax elegantly handles cases where k or filters are not specified
        results = retriever_service.retrieve(query, **params)

        # Pretty-print the JSON output.
        # ensure_ascii=False is crucial for correctly displaying non-English characters like Bengali.
        print(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n" + "="*80)
    print("All test cases complete.")
    print("="*80)
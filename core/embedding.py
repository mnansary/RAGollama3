
import torch 
from sentence_transformers import SentenceTransformer

class JinaV3Embeddings:
    """
    LangChain-compatible embedding class for jina-embeddings-v3.
    It correctly handles the 'task' parameter for optimal retrieval performance.
    """
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"JinaV3Embeddings using device: {self.device}")
        
        # Load the Jina V3 model with trust_remote_code=True
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        print(f"Successfully loaded model: {model_name}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of documents (passages) for storage.
        Uses the 'retrieval.passage' task as recommended by Jina V3 docs.
        """
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

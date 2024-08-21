from sentence_transformers import SentenceTransformer
import torch

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

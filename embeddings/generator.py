from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding model"""
        self.model = SentenceTransformer(model_name)
    
    def generate(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings for the given texts"""
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        
        # Convert to numpy array and then to list
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu()
        embeddings_np = np.array(embeddings)
        
        return embeddings_np.tolist()
    
    def normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit length"""
        embeddings_np = np.array(embeddings)
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized = embeddings_np / norms
        return normalized.tolist()
    
    async def generate_async(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> List[List[float]]:
        """Async wrapper for generate method"""
        return self.generate(texts, batch_size) 
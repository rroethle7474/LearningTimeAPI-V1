from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

class EmbeddingGenerator:
    def __init__(self):
        # Initialize the model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def generate(self, text: str) -> List[float]:
        """Generate embeddings for the given text"""
        try:
            # Encode the text and get embeddings
            embeddings = self.model.encode(text)
            # Convert numpy array to list and ensure correct shape
            return embeddings.tolist()  # Changed from reshape to tolist()
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise
            
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            # Encode multiple texts at once
            embeddings = self.model.encode(texts)
            # Convert numpy array to list
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            raise
    
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
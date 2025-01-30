from typing import List, Dict, Any, Optional
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator

class SemanticSearch:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator
    ):
        """Initialize search with vector store and embedding generator"""
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    async def search(
        self,
        query: str,
        collection: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform semantic search across specified collection
        """
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate(query)
        
        # Search in vector store
        results = self.vector_store.search_collection(
            collection_name=collection,
            query_embeddings=query_embedding,
            n_results=limit,
            where=filters
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"])):
            formatted_results.append({
                "id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
                "distance": results["distances"][i] if "distances" in results else None
            })
        
        return {
            "query": query,
            "results": formatted_results
        }
    
    async def multi_collection_search(
        self,
        query: str,
        collections: List[str],
        limit_per_collection: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across multiple collections
        """
        results = {}
        
        for collection in collections:
            collection_results = await self.search(
                query=query,
                collection=collection,
                limit=limit_per_collection,
                filters=filters
            )
            results[collection] = collection_results["results"]
        
        return {
            "query": query,
            "collections": results
        } 
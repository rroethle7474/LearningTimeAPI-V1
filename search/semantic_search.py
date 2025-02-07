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
        try:
            # Generate embedding for query
            query_embedding = self.embedding_generator.generate(query)
            
            # Search in vector store
            results = self.vector_store.search_collection(
                collection_name=collection,
                query_embeddings=query_embedding,
                n_results=limit,
                where=filters
            )
            
            # print(f"Raw results for {collection}:", results)  # Debug print
            
            # Check if we got any results and if the first id list is empty
            if (not results or 
                not results.get("ids") or 
                len(results["ids"]) == 0 or 
                (len(results["ids"]) == 1 and len(results["ids"][0]) == 0)):
                print(f"No results found for collection: {collection}")
                return {
                    "query": query,
                    "results": []
                }
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"])):
                # Skip empty results
                if not results["ids"][i]:
                    continue
                
                # Ensure all required fields exist and are lists
                ids = results["ids"][i] if isinstance(results["ids"][i], list) else [results["ids"][i]]
                documents = results["documents"][i] if isinstance(results["documents"][i], list) else [results["documents"][i]]
                metadatas = results["metadatas"][i] if isinstance(results["metadatas"][i], list) else [results["metadatas"][i]]
                distances = results.get("distances", [None])[i] if results.get("distances") else [None]
                
                # Skip if any required field is empty
                if not ids or not documents or not metadatas:
                    continue
                
                formatted_results.append({
                    "id": ids,
                    "content": documents,
                    "metadata": metadatas,
                    "distance": distances if isinstance(distances, list) else [distances]
                })
            
            return {
                "query": query,
                "results": formatted_results
            }
            
        except Exception as e:
            print(f"Error searching collection {collection}: {str(e)}")
            # Return empty results instead of failing
            return {
                "query": query,
                "results": []
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
        # Always include documents collection unless explicitly excluded
        if "documents" not in collections and "all" in collections:
            collections.append("notes")
        
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
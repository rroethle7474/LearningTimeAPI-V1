import chromadb
from chromadb.config import Settings
from typing import Optional

class VectorStore:
    def __init__(self, persist_directory: str = "./chromadb"):
        """Initialize ChromaDB client and collections"""
        # self.client = chromadb.Client(Settings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=persist_directory
        # ))
        self.client = chromadb.Client()
        self._init_collections()
    
    def _init_collections(self):
        """Initialize or get existing collections"""
        self.articles = self.client.get_or_create_collection(
            name="articles_content",
            metadata={"description": "Stores article content and embeddings"}
        )
        
        self.youtube = self.client.get_or_create_collection(
            name="youtube_content",
            metadata={"description": "Stores video transcripts and embeddings"}
        )
        
        self.tutorials = self.client.get_or_create_collection(
            name="tutorials",
            metadata={"description": "Stores generated tutorials"}
        )
    
    def get_collection(self, collection_name: str):
        """Get a collection by name"""
        if collection_name == "article":
            return self.articles
        elif collection_name == "youtube":
            return self.youtube
        elif collection_name == "tutorial":
            return self.tutorials
        else:
            raise ValueError(f"Unknown collection: {collection_name}")
    
    def add_to_collection(
        self,
        collection_name: str,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None
    ):
        """Add documents to a specific collection"""
        collection = self.get_collection(collection_name)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def search_collection(
        self,
        collection_name: str,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: Optional[dict] = None
    ):
        """Search within a specific collection"""
        collection = self.get_collection(collection_name)
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where
        )
    
    def delete_from_collection(self, collection_name: str, ids: list[str]):
        """Delete documents from a collection by IDs"""
        collection = self.get_collection(collection_name)
        collection.delete(ids=ids)
    
    def get_by_id(self, collection_name: str, id: str):
        """Get a specific document by ID"""
        collection = self.get_collection(collection_name)
        return collection.get(ids=[id]) 
import chromadb
from chromadb.config import Settings
from typing import Optional, Dict, Any, List
from datetime import datetime
from app_types.tutorial import TutorialSectionType
import logging
from embeddings.generator import EmbeddingGenerator  # Add this import
import json

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(
        self, 
        embedding_generator: EmbeddingGenerator,  # Make it required and typed
        persist_directory: str = "./chromadb",
    ):
        """Initialize ChromaDB client and collections"""
        # self.client = chromadb.Client(Settings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=persist_directory
        # ))
        self.client = chromadb.Client()
        self.embedding_generator = embedding_generator
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
        
        # Updated tutorial collection with string metadata
        self.tutorials = self.client.get_or_create_collection(
            name="tutorials",
            metadata={
                "description": "Stores generated tutorials",
                "schema_version": "2.0",
                "section_types": ",".join(TutorialSectionType.__args__)  # Use the Literal types
            }
        )

        # New collection for tutorial sections
        self.tutorial_sections = self.client.get_or_create_collection(
            name="tutorial_sections",
            metadata={
                "description": "Stores individual tutorial sections",
                "schema_version": "2.0"
            }
        )
    
    def add_tutorial(
        self,
        tutorial_id: str,
        tutorial_data: dict,
        embeddings: List[float]
    ) -> None:
        """Add a tutorial to the vector store"""
        # Create a deep copy and process all data for storage
        processed_tutorial_data = tutorial_data.copy()
        
        # Convert datetime to ISO format string
        if isinstance(processed_tutorial_data["metadata"]["generated_date"], datetime):
            processed_tutorial_data["metadata"]["generated_date"] = processed_tutorial_data["metadata"]["generated_date"].isoformat()
        
        # Serialize any dictionary metadata in sections
        for section in processed_tutorial_data["sections"]:
            if isinstance(section["metadata"], dict):
                section["metadata"] = json.dumps(section["metadata"])
        
        self.tutorials.add(
            ids=[tutorial_id],
            documents=[json.dumps(processed_tutorial_data)],
            embeddings=[embeddings],
            metadatas=[{"type": "tutorial"}]
        )

    def get_tutorial_with_sections(self, tutorial_id: str) -> Dict[str, Any]:
        """Retrieve a tutorial with all its sections"""
        tutorial = self.tutorials.get(ids=[tutorial_id])
        if not tutorial or not tutorial["documents"]:
            raise ValueError(f"Tutorial not found: {tutorial_id}")
        
        # Parse the stored JSON
        tutorial_data = json.loads(tutorial["documents"][0])
        
        # Convert ISO datetime string back to datetime object
        if "metadata" in tutorial_data and "generated_date" in tutorial_data["metadata"]:
            tutorial_data["metadata"]["generated_date"] = datetime.fromisoformat(
                tutorial_data["metadata"]["generated_date"]
            )
        
        # Deserialize metadata strings back to dictionaries
        for section in tutorial_data["sections"]:
            if isinstance(section["metadata"], str):
                try:
                    section["metadata"] = json.loads(section["metadata"])
                except json.JSONDecodeError:
                    section["metadata"] = {}
        
        return tutorial_data

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

    def search_tutorial_sections(
        self,
        query_embeddings: List[float],
        section_type: Optional[str] = None,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search for specific tutorial sections
        """
        where_clause = {"type": section_type} if section_type else None
        return self.tutorial_sections.query(
            query_embeddings=[query_embeddings],
            n_results=n_results,
            where=where_clause
        )

    def inspect_collection(self, collection_name: str) -> Dict[str, Any]:
        """Debug helper to inspect collection contents"""
        collection = self.get_collection(collection_name)
        try:
            # Get all items in collection
            results = collection.get()
            return {
                "count": len(results["ids"]) if results["ids"] else 0,
                "ids": results["ids"],
                "metadatas": results["metadatas"],
                "sample": {
                    "ids": results["ids"][:5] if results["ids"] else [],
                    "metadatas": results["metadatas"][:5] if results["metadatas"] else [],
                    "documents": results["documents"][:5] if results["documents"] else []
                }
            }
        except Exception as e:
            logger.error(f"Error inspecting collection {collection_name}: {str(e)}")
            return {"error": str(e)}

    def add_content(
        self,
        content_id: str,
        content_type: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, Any]
    ):
        """Add content chunks and metadata to the appropriate collection"""
        logger.debug(f"Adding content {content_id} to {content_type} collection")
        
        collection = self.get_collection(content_type)
        chunk_ids = []
        chunk_metadatas = []
        
        # Sanitize metadata to ensure no None values
        base_metadata = {
            "content_id": content_id,
            "title": metadata.get("title", ""),
            "author": metadata.get("author", "Unknown"),
            "source_url": metadata.get("source_url", ""),
            "content_type": metadata.get("content_type", ""),
            "duration": metadata.get("duration", ""),
            "published_date": metadata.get("published_date", ""),
            "view_count": int(metadata.get("view_count", 0)),  # Ensure integer
            "summary": metadata.get("summary", ""),  # Empty string instead of None
            "processed_date": metadata.get("processed_date", datetime.utcnow().isoformat())
        }
        
        # Ensure all metadata values are valid types for ChromaDB
        base_metadata = {
            k: str(v) if v is not None else ""  # Convert None to empty string
            for k, v in base_metadata.items()
        }
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # First chunk gets all metadata including summary
            if i == 0:
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": i
                }
            else:
                # Subsequent chunks get basic metadata without summary
                chunk_metadata = {
                    "content_id": content_id,
                    "chunk_index": i,
                    "title": base_metadata["title"],
                    "content_type": base_metadata["content_type"]
                }
            
            chunk_metadatas.append(chunk_metadata)
            chunk_ids.append(f"{content_id}_{i}")
        
        # Add to collection
        collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=chunk_metadatas
        )
        logger.debug(f"Successfully added content {content_id} with {len(chunks)} chunks")

    def get_content_by_id(self, content_id: str, content_type: str) -> Optional[Dict[str, Any]]:
        """Get all chunks and metadata for a specific content_id"""
        collection = self.get_collection(content_type)
        
        # Query for all chunks with this content_id
        results = collection.get(
            where={"content_id": content_id}
        )
        
        if not results or not results["ids"]:
            return None
        
        # Sort chunks by index
        sorted_indices = sorted(range(len(results["ids"])), 
                              key=lambda i: results["metadatas"][i]["chunk_index"])
        
        # Get full metadata from first chunk
        full_metadata = {k: v for k, v in results["metadatas"][0].items() 
                        if k not in ["chunk_index"]}
        
        return {
            "content_id": content_id,
            "documents": [results["documents"][i] for i in sorted_indices],
            "metadata": full_metadata
        } 
import chromadb
from chromadb.config import Settings
from typing import Optional, Dict, Any, List
from datetime import datetime
from app_types.tutorial import TutorialSectionType
import logging
from embeddings.generator import EmbeddingGenerator  # Add this import
import json
from app_types.document import DocumentMetadata

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(
        self, 
        embedding_generator: EmbeddingGenerator,
        persist_directory: str = "./chromadb",
        client_settings: Optional[Settings] = None
    ):
        """Initialize ChromaDB client and collections"""
        try:
            # Use the new recommended configuration
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )
            
            print(f"Initialized ChromaDB client: {type(self.client).__name__}")
            self.embedding_generator = embedding_generator
            
            # First ensure collections exist
            self._ensure_collections_exist()
            # Then initialize instance variables
            self._init_collections()
        except Exception as e:
            print(f"ChromaDB initialization error: {str(e)}")
            raise
    
    def _ensure_collections_exist(self):
        """Ensure required collections exist"""
        required_collections = [
            "articles_content",
            "youtube_content",
            "tutorials",
            "tutorial_sections",
            "documents"  # Add documents collection
        ]
        
        # In v0.6.0, list_collections() returns collection names directly
        existing_collections = self.client.list_collections()
        
        for collection_name in required_collections:
            if collection_name not in existing_collections:
                logger.info(f"Creating collection: {collection_name}")
                self.client.create_collection(name=collection_name)

    def _init_collections(self):
        """Initialize instance variables for collections"""
        logger.debug("Initializing collection instance variables")
        try:
            self.articles = self.client.get_collection("articles_content")
            self.youtube = self.client.get_collection("youtube_content")
            self.tutorials = self.client.get_collection("tutorials")
            self.tutorial_sections = self.client.get_collection("tutorial_sections")
            self.documents = self.client.get_collection("documents")  # Add documents collection
            logger.debug("Successfully initialized all collection instance variables")
        except Exception as e:
            logger.error(f"Error initializing collections: {str(e)}")
            raise

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
        print("GET TUTORIAL WITH SECTIONS", tutorial_id)
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

    def get_collection(self, content_type: str) -> Any:
        """Get collection by content type or collection name"""
        # First check if this is a direct collection name
        if content_type in ["articles_content", "youtube_content", "tutorials", "tutorial_sections"]:
            try:
                return self.client.get_collection(name=content_type)
            except Exception as e:
                raise ValueError(f"Error accessing collection {content_type}: {str(e)}")
        
        # If not, try to map from content type to collection name
        collection_map = {
            "article": "articles_content",
            "youtube": "youtube_content",
            "youtubes_content": "youtube_content"
        }
        
        collection_name = collection_map.get(content_type)
        if not collection_name:
            raise ValueError(f"Invalid content type: {content_type}")
        
        try:
            return self.client.get_collection(name=collection_name)
        except Exception as e:
            raise ValueError(f"Error accessing collection {collection_name}: {str(e)}")
    
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
        print("COLLECTION", collection)
        # Query for all chunks with this content_id
        results = collection.get(
            where={"content_id": content_id}
        )
        print("RESULTS", results)
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

    def get_collection_contents(
        self,
        collection_name: str,
        offset: int = 0,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get paginated contents of a collection
        """
        try:
            collection = self.client.get_collection(collection_name)
            
            # Get total count
            total = collection.count()
            
            # Get paginated results
            results = collection.get(
                limit=limit,
                offset=offset
            )
            
            items = []
            for i in range(len(results["ids"])):
                items.append({
                    "id": results["ids"][i],
                    "document": results["documents"][i] if "documents" in results else None,
                    "metadata": results["metadatas"][i] if "metadatas" in results else {}
                })
                
            return {
                "total": total,
                "items": items
            }
            
        except Exception as e:
            raise Exception(f"Error getting collection contents: {str(e)}")

    def find_tutorial_for_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Find a tutorial generated from specific content"""
        try:
            # Search in tutorials collection for matching content_id
            results = self.tutorials.get(
                where={"source_content_id": content_id}
            )
            
            if results and results["ids"]:
                # Return first matching tutorial
                return {
                    "id": results["ids"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {},
                    "document": results["documents"][0] if results["documents"] else None
                }
            return None
            
        except Exception as e:
            logger.error(f"Error finding tutorial for content {content_id}: {str(e)}")
            return None 

    def delete_tutorial(self, tutorial_id: str) -> None:
        """Delete a tutorial and all its sections"""
        try:
            # Get the tutorial first to verify it exists and get section IDs
            tutorial = self.get_tutorial_with_sections(tutorial_id)
            if not tutorial:
                raise ValueError(f"Tutorial not found: {tutorial_id}")
            
            # Delete the tutorial
            self.tutorials.delete(
                ids=[tutorial_id]
            )
            
            # Delete all sections
            section_ids = [section["id"] for section in tutorial["sections"]]
            if section_ids:
                self.tutorial_sections.delete(
                    ids=section_ids
                )
            
        except Exception as e:
            logger.error(f"Error deleting tutorial {tutorial_id}: {str(e)}")
            raise 

    def add_document(
        self,
        document_id: str,
        content: str,
        metadata: DocumentMetadata,
        embeddings: List[float]
    ) -> None:
        """
        Add a document to the vector store with metadata and tags
        
        Args:
            document_id: Unique identifier for the document
            content: Extracted text content from the document
            metadata: Document metadata including title, tags, etc.
            embeddings: Vector embeddings for the document content
        """
        try:
            self.documents.add(
                ids=[document_id],
                documents=[content],
                embeddings=[embeddings],
                metadatas=[{
                    "title": metadata.title,
                    "tags": ",".join(metadata.tags),
                    "file_type": metadata.file_type,
                    "file_size": metadata.file_size,
                    "upload_date": metadata.upload_date.isoformat(),
                    "source_file": metadata.source_file
                }]
            )
            logger.debug(f"Successfully added document {document_id}")
        except Exception as e:
            logger.error(f"Error adding document {document_id}: {str(e)}")
            raise

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            Dictionary containing document content and metadata, or None if not found
        """
        try:
            result = self.documents.get(
                ids=[document_id]
            )
            
            if not result["ids"]:
                return None
                
            metadata = result["metadatas"][0]
            return {
                "document_id": document_id,
                "content": result["documents"][0],
                "metadata": {
                    "title": metadata["title"],
                    "tags": metadata["tags"].split(","),
                    "file_type": metadata["file_type"],
                    "file_size": metadata["file_size"],
                    "upload_date": datetime.fromisoformat(metadata["upload_date"]),
                    "source_file": metadata["source_file"]
                }
            }
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {str(e)}")
            raise

    def delete_document(self, document_id: str) -> None:
        """
        Delete a document by its ID
        
        Args:
            document_id: The ID of the document to delete
        """
        try:
            self.documents.delete(
                ids=[document_id]
            )
            logger.debug(f"Successfully deleted document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise

    def search_documents(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search document chunks and return results with parent document info"""
        try:
            # Add chunk filter
            chunk_filter = {"is_chunk": True}
            if filters:
                filters = {**filters, **chunk_filter}
            else:
                filters = chunk_filter
            
            results = self.documents.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=filters
            )
            
            # Process results to include parent document info
            processed_results = []
            for i, (doc_id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
                metadata = results["metadatas"][0][i]
                parent_doc = self.get_document(metadata["document_id"])
                
                processed_results.append({
                    "chunk_id": doc_id,
                    "chunk_content": results["documents"][0][i],
                    "chunk_index": metadata["chunk_index"],
                    "document_id": metadata["document_id"],
                    "document_title": metadata["title"],
                    "document_metadata": parent_doc["metadata"] if parent_doc else metadata,
                    "similarity": 1 - distance
                })
            
            return processed_results
        
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise

    def store_full_document(self, document_id: str, content: str, metadata: Dict[str, Any]):
        """Store full document text and metadata (without embeddings)"""
        try:
            self.documents.add(
                ids=[document_id],
                documents=[content],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"Error storing full document {document_id}: {str(e)}")
            raise 
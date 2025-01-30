import chromadb
from chromadb.config import Settings
from typing import Optional, Dict, Any, List
from datetime import datetime
from app_types.tutorial import TutorialSectionType

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
    
    def add_tutorial(self, tutorial_id: str, tutorial_data: Dict[str, Any], embeddings: List[float]):
        """
        Add a tutorial with its sections to the database
        """
        # Store the tutorial metadata
        self.tutorials.add(
            ids=[tutorial_id],
            embeddings=[embeddings],
            documents=[tutorial_data["metadata"]["title"]],
            metadatas=[{
                "title": tutorial_data["metadata"]["title"],
                "content_id": tutorial_data["metadata"]["content_id"],
                "source_url": tutorial_data["metadata"]["source_url"],
                "content_type": tutorial_data["metadata"]["content_type"],
                "generated_date": tutorial_data["metadata"]["generated_date"].isoformat(),
                "section_count": len(tutorial_data["sections"])
            }]
        )

        # Store each section with a reference to the tutorial
        for section in tutorial_data["sections"]:
            section_embedding = self.embedding_generator.generate([section["content"]])[0]
            self.tutorial_sections.add(
                ids=[section["id"]],
                embeddings=[section_embedding],
                documents=[section["content"]],
                metadatas=[{
                    "tutorial_id": tutorial_id,
                    "type": section["type"],
                    "title": section["title"],
                    "metadata": section.get("metadata", {}),
                }]
            )

    def get_tutorial_with_sections(self, tutorial_id: str) -> Dict[str, Any]:
        """
        Retrieve a tutorial with all its sections
        """
        # Get tutorial metadata
        tutorial = self.tutorials.get(ids=[tutorial_id])
        if not tutorial or not tutorial["documents"]:
            raise ValueError(f"Tutorial not found: {tutorial_id}")
        
        # Get all sections for this tutorial
        sections = self.tutorial_sections.get(
            where={"tutorial_id": tutorial_id}
        )
        
        return {
            "metadata": tutorial["metadatas"][0],
            "sections": [
                {
                    "id": sections["ids"][i],
                    "type": sections["metadatas"][i]["type"],
                    "title": sections["metadatas"][i]["title"],
                    "content": sections["documents"][i],
                    "metadata": sections["metadatas"][i].get("metadata", {})
                }
                for i in range(len(sections["ids"]))
            ]
        }

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
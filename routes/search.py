from fastapi import APIRouter, Query, HTTPException, Depends, Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator
from search.semantic_search import SemanticSearch
from datetime import datetime
from urllib.parse import unquote

router = APIRouter()

class SearchResult(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

class MultiCollectionSearchResponse(BaseModel):
    query: str
    collections: Dict[str, List[SearchResult]]

class ContentListItem(BaseModel):
    id: str
    title: str
    type: str
    source: str
    metadata: Dict[str, Any]

class ContentListResponse(BaseModel):
    total: int
    items: List[ContentListItem]

class MultiCollectionContentResponse(BaseModel):
    collections: Dict[str, ContentListResponse]

class ContentDetailResponse(BaseModel):
    id: str
    title: str
    content_type: str
    author: str
    source_url: str
    summary: Optional[str]
    published_date: Optional[str]
    processed_date: str
    tutorial_id: Optional[str] = None  # Reference to generated tutorial if exists
    content_chunks: List[str]  # The actual content broken into chunks
    metadata: Dict[str, Any]  # Additional type-specific metadata

def get_embedding_generator():
    """Dependency to get embedding generator instance"""
    from main import embedding_generator
    return embedding_generator

def get_vector_store(
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator)
):
    """Dependency to get vector store instance"""
    return VectorStore(
        embedding_generator=embedding_generator,
        persist_directory="./chromadb"
    )

def get_semantic_search(
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator)
):
    """Dependency to get semantic search instance"""
    return SemanticSearch(vector_store, embedding_generator)

@router.get("/single", response_model=SearchResponse)
async def search_single_collection(
    query: str,
    collection: str,
    limit: int = 5,
    semantic_search: SemanticSearch = Depends(get_semantic_search)
):
    """Search within a single collection"""
    try:
        results = await semantic_search.search(
            query=query,
            collection=collection,
            limit=limit
        )
        return SearchResponse(query=query, results=results["results"])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/multi", response_model=MultiCollectionSearchResponse)
async def search_multiple_collections(
    query: str,
    collections: List[str] = Query(...),
    limit_per_collection: int = 3,
    semantic_search: SemanticSearch = Depends(get_semantic_search)
):
    """Search across multiple collections"""
    try:
        results = await semantic_search.search_multi(
            query=query,
            collections=collections,
            limit_per_collection=limit_per_collection
        )
        return MultiCollectionSearchResponse(
            query=query,
            collections=results["collections"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/similar/{content_id}")
async def find_similar_content(
    content_id: str,
    collection: str,
    limit: int = 5,
    semantic_search: SemanticSearch = Depends(get_semantic_search)
):
    """Find content similar to a given item"""
    try:
        # Get original content
        content_data = semantic_search.vector_store.get_by_id(collection, content_id)
        if not content_data or not content_data["documents"]:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Use content as query
        results = await semantic_search.search(
            query=content_data["documents"][0],
            collection=collection,
            limit=limit + 1  # Add 1 to account for the original document
        )
        
        # Remove the original document from results if present
        filtered_results = [
            r for r in results["results"]
            if r["id"] != content_id
        ][:limit]
        
        return {
            "content_id": content_id,
            "similar_items": filtered_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collection/{collection_name}/contents", response_model=ContentListResponse)
async def get_collection_contents(
    collection_name: str = Path(..., description="Name of the collection to fetch"),
    offset: int = Query(0, description="Number of records to skip"),
    limit: int = Query(50, description="Maximum number of records to return"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get paginated contents of a single collection"""
    try:
        # Get collection contents
        contents = vector_store.get_collection_contents(
            collection_name=collection_name,
            offset=offset,
            limit=limit
        )
        
        # Format response
        items = []
        for item in contents["items"]:
            metadata = item.get("metadata", {})
            items.append(ContentListItem(
                id=item["id"],
                title=metadata.get("title", "Untitled"),
                type=metadata.get("content_type", "unknown"),
                source=metadata.get("source_url", ""),
                metadata=metadata
            ))
            
        return ContentListResponse(
            total=contents["total"],
            items=items
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections/contents", response_model=MultiCollectionContentResponse)
async def get_multiple_collections_contents(
    collections: List[str] = Query(
        ..., 
        description="Comma-separated list of collections to fetch (e.g. articles_content,youtube_content)",
        example="articles_content,youtube_content"
    ),
    offset: int = Query(0, description="Number of records to skip"),
    limit: int = Query(50, description="Maximum number of records per collection"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get paginated contents from multiple collections"""
    try:
        result = {"collections": {}}
        
        for collection_name in collections:
            contents = vector_store.get_collection_contents(
                collection_name=collection_name,
                offset=offset,
                limit=limit
            )
            
            # Format items for this collection
            items = []
            for item in contents["items"]:
                metadata = item.get("metadata", {})
                items.append(ContentListItem(
                    id=item["id"],
                    title=metadata.get("title", "Untitled"),
                    type=metadata.get("content_type", "unknown"),
                    source=metadata.get("source_url", ""),
                    metadata=metadata
                ))
                
            result["collections"][collection_name] = ContentListResponse(
                total=contents["total"],
                items=items
            )
            
        return MultiCollectionContentResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content/{collection_name}/{content_id}", response_model=ContentDetailResponse)
async def get_content_by_id(
    collection_name: str = Path(..., description="Name of the collection"),
    content_id: str = Path(..., description="ID of the content to retrieve"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Get content details by ID from a specific collection.
    Returns a normalized view of the content regardless of type.
    """
    try:
        # Get content from vector store
        content = vector_store.get_content_by_id(content_id, collection_name)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        # Get base metadata
        metadata = content["metadata"]
        
        # Check if there's an associated tutorial
        tutorial_id = None
        if collection_name in ["articles_content", "youtube_content"]:
            # You might need to implement this method in vector_store
            tutorial = vector_store.find_tutorial_for_content(content_id)
            if tutorial:
                tutorial_id = tutorial["id"]

        # Normalize the response
        response = ContentDetailResponse(
            id=content_id,
            title=metadata.get("title", "Untitled"),
            content_type=metadata.get("content_type", "unknown"),
            author=metadata.get("author", "Unknown"),
            source_url=metadata.get("source_url", ""),
            summary=metadata.get("summary"),
            published_date=metadata.get("published_date"),
            processed_date=metadata.get("processed_date", datetime.utcnow().isoformat()),
            tutorial_id=tutorial_id,
            content_chunks=content["documents"],
            metadata=metadata  # Include all metadata for type-specific UI enhancements
        )
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/content/{collection_name}/{content_id}", status_code=204)
async def delete_content(
    collection_name: str = Path(..., description="Name of the collection"),
    content_id: str = Path(..., description="ID of the content to delete"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Delete content from a collection by ID.
    Returns 204 on success with no content.
    """
    try:
        # Get content first to verify it exists
        content = vector_store.get_content_by_id(content_id, collection_name)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
            
        # Delete the content
        vector_store.delete_from_collection(collection_name, [content_id])
        
        return None  # 204 No Content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content/{collection_name}/by-url")
async def get_content_by_url(
    collection_name: str = Path(..., description="Name of the collection"),
    source_url: str = Query(..., description="Source URL to search for"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Check if content exists by source URL in a specific collection.
    Returns 200 if found, 404 if not found.
    """
    try:
        # Decode the URL to handle any URL encoding
        decoded_url = unquote(source_url)
        
        # Get collection
        collection = vector_store.get_collection(collection_name)
        
        # Query for content with matching source_url
        results = collection.get(
            where={"source_url": decoded_url}
        )
        
        if not results or not results["ids"]:
            # Try with the original encoded URL as fallback
            results = collection.get(
                where={"source_url": source_url}
            )
            
        if not results or not results["ids"]:
            raise HTTPException(status_code=404, detail="Content not found for given URL")

        return {"exists": True, "content_id": results["ids"][0]}
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
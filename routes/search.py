from fastapi import APIRouter, Query, HTTPException, Depends, Path
from typing import List, Optional, Dict, Any, Annotated
from pydantic import BaseModel
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator
from search.semantic_search import SemanticSearch
from datetime import datetime
from urllib.parse import unquote
# Import dependencies
from dependencies import get_vector_store, get_embedding_generator, get_semantic_search
from utils.duration import format_duration  # Add this import at the top

router = APIRouter()

# Add this as a constant at the top of the file with other imports
MINIMUM_SIMILARITY_THRESHOLD = 0.01  # This means results must be at least 1% similar

class SearchResult(BaseModel):
    id: str  # Change from single string to handle the first ID
    content: str  # Take first content
    metadata: Dict[str, Any]  # Take first metadata
    distance: Optional[float] = None  # Take first distance

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

def normalize_distance_score(distances: List[float]) -> float:
    """
    Convert an array of distance values into a single normalized score.
    Returns a value between 0 and 1, where 1 is the best match.
    """
    if not distances or not isinstance(distances, list):
        return None
        
    # Take the average distance if there are multiple values
    avg_distance = sum(distances) / len(distances)
    
    # Convert cosine distance (0-2 range) to similarity score (0-1 range)
    # where 1 is most similar and 0 is least similar
    similarity_score = (2 - avg_distance) / 2
    
    # Ensure the score stays within bounds
    return max(0, min(1, similarity_score))

def format_search_result(result: Dict[str, Any]) -> SearchResult:
    """Format a search result, including duration formatting"""
    metadata = result["metadata"][0]
    
    # Format duration if it exists and is a YouTube video
    if metadata.get("content_type") == "youtube" and "duration" in metadata:
        metadata = {**metadata, "duration": format_duration(metadata["duration"])}
    
    return SearchResult(
        id=result["id"][0],
        content=result["content"][0],
        metadata=metadata,
        distance=normalize_distance_score(result.get("distance"))
    )

@router.get("/single", response_model=SearchResponse)
async def search_single_collection(
    query: str,
    collection: str,
    limit: int = 5,
    min_similarity: float = Query(default=MINIMUM_SIMILARITY_THRESHOLD, ge=0.0, le=1.0, description="Minimum similarity threshold (0-1). Higher values return more relevant results."),
    semantic_search: SemanticSearch = Depends(get_semantic_search)
):
    """Search within a single collection"""
    print("SEARCHING SINGLE COLLECTION", query)
    print("COLLECTION", collection)
    try:
        results = await semantic_search.search(
            query=query,
            collection=collection,
            limit=limit
        )
        print("RESULTS", results)
        # Transform the results to match the SearchResult model
        processed_results = []
        for result in results["results"]:
            formatted_result = format_search_result(result)
            if formatted_result.distance >= min_similarity:
                processed_results.append(formatted_result)
        print("PROCESSED RESULTS", processed_results)
        return SearchResponse(query=query, results=processed_results)
        
    except Exception as e:
        print("Error processing search results:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/multi", response_model=SearchResponse)
async def search_multiple_collections(
    query: str,
    collections: List[str] = Query(...),
    limit_per_collection: int = 3,
    min_similarity: float = Query(default=MINIMUM_SIMILARITY_THRESHOLD, ge=0.0, le=1.0, description="Minimum similarity threshold (0-1). Higher values return more relevant results."),
    semantic_search: SemanticSearch = Depends(get_semantic_search)
):
    """Search across multiple collections"""
    try:
        # Split any collections that contain commas and flatten the list
        expanded_collections = []
        for collection in collections:
            expanded_collections.extend(collection.split(','))
        
        # Clean up the collection names
        cleaned_collections = [c.strip() for c in expanded_collections]
        
        print("Processing collections:", cleaned_collections)
        results = await semantic_search.multi_collection_search(
            query=query,
            collections=cleaned_collections,
            limit_per_collection=limit_per_collection
        )
        
        # Combine and process all results
        processed_results = []
        for collection_results in results["collections"].values():
            for result in collection_results:
                formatted_result = format_search_result(result)
                if formatted_result.distance >= min_similarity:
                    processed_results.append(formatted_result)
            
        return SearchResponse(query=query, results=processed_results)
        
    except Exception as e:
        print("Error processing multi-collection search results:", str(e))
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
        
        # Track seen content_ids to avoid duplicates
        seen_content_ids = set()
        items = []
        
        for item in contents["items"]:
            metadata = item.get("metadata", {})
            # Get the base content_id (remove chunk suffix if present)
            item_id = item["id"].split('_')[0] if item["id"] else None
            content_id = metadata.get("content_id", item_id)
            
            # Only process items that haven't been seen before
            if content_id and content_id not in seen_content_ids:
                seen_content_ids.add(content_id)
                items.append(ContentListItem(
                    id=content_id,
                    title=metadata.get("title", "Untitled"),
                    type=metadata.get("content_type", "unknown"),
                    source=metadata.get("source_url", ""),
                    metadata=metadata
                ))
            
        return ContentListResponse(
            total=len(items),
            items=items
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections/contents", response_model=MultiCollectionContentResponse)
async def get_multiple_collections_contents(
    collections: List[str] = Query(
        ..., 
        description="List of collections to fetch",
        example=["articles_content", "youtube_content"]
    ),
    offset: int = Query(0, description="Number of records to skip"),
    limit: int = Query(50, description="Maximum number of records per collection"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get paginated contents from multiple collections"""
    try:
        result = {"collections": {}}
        
        # Split any items that might contain commas
        expanded_collections = []
        for collection in collections:
            expanded_collections.extend(collection.split(','))
        
        # Remove any whitespace and convert to lowercase
        collections = [c.strip().lower() for c in expanded_collections]
        
        for collection_name in collections:
            seen_content_ids = set()
            contents = vector_store.get_collection_contents(
                collection_name=collection_name,
                offset=offset,
                limit=limit
            )
            
            # Format items for this collection
            items = []
            for item in contents["items"]:
                metadata = item.get("metadata", {})
                content_id = metadata.get("content_id")
                metadata["duration"] = format_duration(metadata.get("duration"))
                # Only process items that are the first chunk and haven't been seen
                if content_id and metadata.get("chunk_index") == 0 and content_id not in seen_content_ids:
                    seen_content_ids.add(content_id)
                    items.append(ContentListItem(
                        id=content_id,
                        title=metadata.get("title", "Untitled"),
                        type=metadata.get("content_type", "unknown"),
                        source=metadata.get("source_url", ""),
                        metadata=metadata
                    ))
                
            result["collections"][collection_name] = ContentListResponse(
                total=len(items),
                items=items
            )
            
        return MultiCollectionContentResponse(**result)
        
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content/{collection_name}/by-url")
async def get_content_by_url(
    collection_name: Annotated[str, Path(...)],
    source_url: Annotated[str, Query(...)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)]
):
    """
    Check if content exists by source URL in a specific collection.
    Returns an object indicating existence and content_id (if found).
    """
    try:
        decoded_url = unquote(source_url)
        collection = vector_store.get_collection(collection_name)
        
        results = collection.get(
            where={"source_url": decoded_url}
        )
        
        # Extract the base content_id from the chunk id (remove _0 suffix if present)
        content_id = results["ids"][0].split('_')[0] if (results and results["ids"]) else None
        
        return {
            "exists": bool(results and results["ids"]),
            "content_id": content_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/content/{collection_name}/{content_id}", response_model=ContentDetailResponse)
async def get_content_detail(
    collection_name: str,
    content_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get detailed content information"""
    try:
        print("\n=== GETTING CONTENT BY ID ===")
        print("Content ID:", content_id)
        print("Collection Name:", collection_name)
        print("Vector Store ID:", id(vector_store))
        
        # Get content from vector store
        content = vector_store.get_content_by_id(content_id, collection_name)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        # Get base metadata
        metadata = content["metadata"]
      
        # Format duration if it's a YouTube video
        if metadata.get("content_type") == "youtube" and "duration" in metadata:
            metadata = {**metadata, "duration": format_duration(metadata["duration"])}
        
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
        # Remove chunk suffix if present (e.g. "abc_0" -> "abc")
        base_content_id = content_id.split('_')[0]
        print("BASE CONTENT ID", base_content_id)
        print("COLLECTION NAME", collection_name)
        # Get content first to verify it exists
        content = vector_store.get_content_by_id(base_content_id, collection_name)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
            
        # Delete all chunks for this content
        collection = vector_store.get_collection(collection_name)
        collection.delete(where={"content_id": base_content_id})
        
        return None  # 204 No Content
        
    except Exception as e:
        print(f"Error deleting content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/vector-store")
async def debug_vector_store(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Debug endpoint to check vector store state"""
    try:
        collections_info = {
            "articles_content": {
                "exists": hasattr(vector_store, "articles"),
                "count": vector_store.articles.count() if hasattr(vector_store, "articles") else None,
                "peek": vector_store.articles.peek() if hasattr(vector_store, "articles") else None
            },
            "youtube_content": {
                "exists": hasattr(vector_store, "youtube"),
                "count": vector_store.youtube.count() if hasattr(vector_store, "youtube") else None,
                "peek": vector_store.youtube.peek() if hasattr(vector_store, "youtube") else None
            },
            "tutorials": {
                "exists": hasattr(vector_store, "tutorials"),
                "count": vector_store.tutorials.count() if hasattr(vector_store, "tutorials") else None,
                "peek": vector_store.tutorials.peek() if hasattr(vector_store, "tutorials") else None
            }
        }
        
        return {
            "vector_store_id": id(vector_store),
            "collections": collections_info,
            "client_type": type(vector_store.client).__name__
        }
    except Exception as e:
        print("\n=== DEBUG ENDPOINT ERROR ===")
        print("Error Type:", type(e))
        print("Error Message:", str(e))
        print("Error Details:", repr(e))
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@router.post("/debug/add-test-content")
async def add_test_content(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Debug endpoint to add test content"""
    try:
        # Add test article
        test_article = {
            "id": "test_article_1",
            "content": "This is a test article content",
            "metadata": {
                "title": "Test Article",
                "author": "Test Author",
                "source_url": "https://test.com/article1",
                "content_type": "article"
            }
        }
        
        # Generate a test embedding
        embedding = vector_store.embedding_generator.generate(test_article["content"])
        
        # Add to articles collection
        vector_store.articles.add(
            ids=[test_article["id"]],
            documents=[test_article["content"]],
            embeddings=[embedding],
            metadatas=[test_article["metadata"]]
        )
        
        return {
            "message": "Test content added successfully",
            "article_id": test_article["id"]
        }
        
    except Exception as e:
        print("\n=== TEST CONTENT ERROR ===")
        print("Error Type:", type(e))
        print("Error Message:", str(e))
        print("Error Details:", repr(e))
        raise HTTPException(status_code=500, detail=f"Error adding test content: {str(e)}")

@router.get("/debug/vector-store-status")
async def check_vector_store():
    """Debug endpoint to check vector store initialization"""
    try:
        from main import vector_store
        return {
            "status": "initialized",
            "id": id(vector_store),
            "collections": {
                "articles": hasattr(vector_store, "articles"),
                "youtube": hasattr(vector_store, "youtube"),
                "tutorials": hasattr(vector_store, "tutorials")
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": str(type(e)),
            "error_message": str(e)
        }

@router.get("/debug/url-test")
async def test_url_handling(
    test_url: Annotated[str, Query(...)],
):
    """Debug endpoint to test URL parameter handling"""
    try:
        print("\n=== URL TEST START ===")
        print("Raw URL:", test_url)
        decoded = unquote(test_url)
        print("Decoded URL:", decoded)
        return {
            "raw_url": test_url,
            "decoded_url": decoded
        }
    except Exception as e:
        print("\n=== URL TEST ERROR ===")
        print("Error Type:", type(e))
        print("Error Message:", str(e))
        return {
            "error": str(e),
            "error_type": str(type(e))
        }
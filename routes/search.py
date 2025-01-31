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
    # Add function entry logging before ANY other code
    import sys
    print("\n=== FUNCTION ENTRY LOGGING ===", file=sys.stderr)
    print(f"Python Version: {sys.version}", file=sys.stderr)
    print(f"Collection Name Type: {type(collection_name)}", file=sys.stderr)
    print(f"Source URL Type: {type(source_url)}", file=sys.stderr)
    print(f"Vector Store Type: {type(vector_store)}", file=sys.stderr)
    
    print("\n=== ENDPOINT START ===")
    print("Parameters received:")
    print("Collection Name:", collection_name)
    print("Source URL:", source_url)
    
    try:
        # Single decode is sufficient when UI handles encoding properly
        print("\n=== URL DECODING ===")
        decoded_url = unquote(source_url)
        print("Decoded URL:", decoded_url)
        
        try:
            print("\n=== GETTING COLLECTION ===")
            # Get collection
            collection = vector_store.get_collection(collection_name)
            print("Collection obtained successfully")
            print("Collection Type:", type(collection))
            
            try:
                print("\n=== EXECUTING QUERY ===")
                results = collection.get(
                    where={"source_url": decoded_url}
                )
                print("Query completed")
                print("Results:", results)
                
                return {
                    "exists": bool(results and results["ids"]),
                    "content_id": results["ids"][0] if (results and results["ids"]) else None
                }
                
            except Exception as query_error:
                print("\n=== QUERY ERROR ===")
                print("Error Type:", type(query_error))
                print("Error Message:", str(query_error))
                print("Error Details:", repr(query_error))
                raise HTTPException(status_code=500, detail=f"Query error: {str(query_error)}")
                
        except Exception as collection_error:
            print("\n=== COLLECTION ERROR ===")
            print("Error Type:", type(collection_error))
            print("Error Message:", str(collection_error))
            print("Error Details:", repr(collection_error))
            raise HTTPException(status_code=500, detail=f"Collection error: {str(collection_error)}")
            
    except Exception as e:
        print("\n=== UNEXPECTED ERROR ===")
        print("Error Type:", type(e))
        print("Error Message:", str(e))
        print("Error Details:", repr(e))
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

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
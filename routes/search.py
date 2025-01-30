from fastapi import APIRouter, Query, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from search.semantic_search import SemanticSearch

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

def get_semantic_search():
    # This should be properly initialized with dependencies
    from main import semantic_search
    return semantic_search

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
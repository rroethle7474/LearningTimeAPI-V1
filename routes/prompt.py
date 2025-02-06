from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from services.context_generation_service import ContextGenerationService
from search.semantic_search import SemanticSearch
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator
import logging
# Import dependencies
from dependencies import (
    get_vector_store,
    get_embedding_generator,
    get_semantic_search,
    get_llm_client
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prompts"])

class ContextRequest(BaseModel):
    query: str
    min_similarity: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0-1). Higher values return more relevant results."
    )

class ContextResponse(BaseModel):
    context: str
    error: Optional[str] = None

def get_context_service():
    """Dependency to get context generation service"""
    from main import context_generation_service
    return context_generation_service

@router.post("/generate", response_model=ContextResponse)
async def generate_context(
    request: ContextRequest,
    service: ContextGenerationService = Depends(get_context_service)
):
    """Generate context based on user query"""
    try:
        logger.debug(f"Received context generation request: {request.query}")
        context = await service.generate_context(
            query=request.query,
            min_similarity=request.min_similarity
        )
        
        # If no context was generated, return a more specific error
        if not context or context in [
            "No relevant context found for the given query.",
            "No sufficiently relevant content found for the given query."
        ]:
            return ContextResponse(
                context="No relevant context found for the given query.",
                error="No relevant content found. Try rephrasing your query or adjusting the similarity threshold."
            )
            
        return ContextResponse(context=context)
        
    except Exception as e:
        logger.error(f"Error generating context: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating context: {str(e)}"
        )
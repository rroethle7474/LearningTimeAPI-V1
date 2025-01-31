from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
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

class ContextResponse(BaseModel):
    context: str
    error: Optional[str] = None

def get_context_service(
    llm_client = Depends(get_llm_client),
    semantic_search: SemanticSearch = Depends(get_semantic_search)
):
    """Dependency to get context generation service"""
    return ContextGenerationService(llm_client, semantic_search)

@router.post("/generate", response_model=ContextResponse)
async def generate_context(
    request: ContextRequest,
    service: ContextGenerationService = Depends(get_context_service)
):
    """Generate context based on user query"""
    try:
        logger.debug(f"Received context generation request: {request.query}")
        context = await service.generate_context(request.query)
        return ContextResponse(context=context)
    except Exception as e:
        logger.error(f"Error generating context: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating context: {str(e)}"
        )
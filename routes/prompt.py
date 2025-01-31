from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from llm.factory import create_llm_client
from services.context_generation_service import ContextGenerationService
from search.semantic_search import SemanticSearch
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator
import logging

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prompts"])

class ContextRequest(BaseModel):
    query: str

class ContextResponse(BaseModel):
    context: str
    error: Optional[str] = None

def get_embedding_generator():
    """Dependency to get embedding generator instance"""
    from main import embedding_generator  # Import from main where it's initialized
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

def get_llm_client():
    """Dependency to get LLM client instance"""
    from main import llm_client  # Import from main where it's initialized
    return llm_client

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
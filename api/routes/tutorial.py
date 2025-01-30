from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from pydantic import BaseModel

from generators.tutorial import TutorialGenerator, ProcessedTutorial
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator
from llm.factory import LLMFactory
from config import settings
from app_types.tutorial import TutorialSectionType

router = APIRouter(prefix="/api/tutorials", tags=["tutorials"])

# Add dependency injection functions
def get_tutorial_generator() -> TutorialGenerator:
    """Dependency injection for TutorialGenerator"""
    vector_store = VectorStore()
    embedding_generator = EmbeddingGenerator()
    if settings.ANTHROPIC_API_KEY:
        llm_client = LLMFactory.create_client("anthropic", settings.ANTHROPIC_API_KEY)
    elif settings.OPENAI_API_KEY:
        llm_client = LLMFactory.create_client("openai", settings.OPENAI_API_KEY)
    else:
        raise ValueError("No LLM API keys configured")
    
    return TutorialGenerator(llm_client, vector_store, embedding_generator)

def get_vector_store() -> VectorStore:
    """Dependency injection for VectorStore"""
    return VectorStore()

class TutorialGenerationRequest(BaseModel):
    content_id: str
    content_type: Literal["article", "youtube"]  # Add type validation

class TutorialGenerationResponse(BaseModel):
    tutorial_id: str
    status: str
    message: str

class TutorialStatus(BaseModel):
    status: str
    tutorial_id: Optional[str] = None
    error: Optional[str] = None
    last_updated: datetime

# In-memory status tracking (consider using Redis for production)
generation_status: Dict[str, TutorialStatus] = {}

def update_generation_status(content_id: str, status: str, tutorial_id: Optional[str] = None, error: Optional[str] = None):
    generation_status[content_id] = TutorialStatus(
        status=status,
        tutorial_id=tutorial_id,
        error=error,
        last_updated=datetime.utcnow()
    )

async def generate_tutorial_background(
    content_id: str,
    content_type: str,
    tutorial_generator: TutorialGenerator,
    vector_store: VectorStore
):
    """Background task for tutorial generation"""
    try:
        tutorial = await tutorial_generator.generate_tutorial(content_id, content_type)
        update_generation_status(
            content_id=content_id,
            status="completed",
            tutorial_id=tutorial.metadata.content_id
        )
    except Exception as e:
        update_generation_status(
            content_id=content_id,
            status="failed",
            error=str(e)
        )

@router.post("/generate", response_model=TutorialGenerationResponse)
async def generate_tutorial(
    request: TutorialGenerationRequest,
    background_tasks: BackgroundTasks,
    tutorial_generator: TutorialGenerator = Depends(get_tutorial_generator),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Initiate tutorial generation for a piece of content
    """
    # Check if content exists
    try:
        content = vector_store.get_by_id(request.content_type, request.content_id)
        if not content or not content["documents"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error accessing content: {str(e)}"
        )

    # Initialize generation status
    update_generation_status(request.content_id, "pending")
    
    # Start background generation
    background_tasks.add_task(
        generate_tutorial_background,
        request.content_id,
        request.content_type,
        tutorial_generator,
        vector_store
    )
    
    return TutorialGenerationResponse(
        tutorial_id=request.content_id,
        status="pending",
        message="Tutorial generation started"
    )

@router.get("/status/{content_id}", response_model=TutorialStatus)
async def get_generation_status(content_id: str):
    """
    Get the status of a tutorial generation process
    """
    status = generation_status.get(content_id)
    if not status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No generation status found for this content"
        )
    return status

class TutorialResponse(BaseModel):
    metadata: Dict[str, Any]
    sections: List[Dict[str, Any]]

@router.get("/{tutorial_id}")
async def get_tutorial(
    tutorial_id: str, 
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Retrieve a generated tutorial
    """
    try:
        tutorial_data = vector_store.get_tutorial_with_sections(tutorial_id)
        return {
            "metadata": {
                "title": tutorial_data["metadata"]["title"],
                "content_id": tutorial_data["metadata"]["content_id"],
                "source_url": tutorial_data["metadata"]["source_url"],
                "content_type": tutorial_data["metadata"]["content_type"],
                "generated_date": tutorial_data["metadata"]["generated_date"]
            },
            "sections": tutorial_data["sections"]
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving tutorial: {str(e)}"
        )

@router.post("/{content_id}/retry")
async def retry_generation(
    content_id: str,
    background_tasks: BackgroundTasks,
    tutorial_generator: TutorialGenerator = Depends(get_tutorial_generator),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Retry tutorial generation for failed attempts
    """
    status = generation_status.get(content_id)
    if not status or status.status != "failed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only retry failed generations"
        )
    
    # Reset status and start new generation
    update_generation_status(content_id, "pending")
    
    background_tasks.add_task(
        generate_tutorial_background,
        content_id,
        "article",  # You might want to store the content type in the status
        tutorial_generator,
        vector_store
    )
    
    return {
        "tutorial_id": content_id,
        "status": "pending",
        "message": "Tutorial generation retry started"
    }

@router.get("/{tutorial_id}/sections/{section_type}")
async def get_tutorial_sections(
    tutorial_id: str,
    section_type: TutorialSectionType,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Get specific section types from a tutorial
    """
    try:
        tutorial_data = vector_store.get_tutorial_with_sections(tutorial_id)
        sections = [
            section for section in tutorial_data["sections"]
            if section["type"] == section_type
        ]
        return {"sections": sections}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sections: {str(e)}"
        ) 
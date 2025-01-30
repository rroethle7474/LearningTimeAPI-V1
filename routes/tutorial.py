from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from pydantic import BaseModel
import uuid
from generators.tutorial import TutorialGenerator, ProcessedTutorial
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator
from llm.factory import LLMFactory
from config import settings
from app_types.tutorial import TutorialSectionType

router = APIRouter(tags=["tutorials"])

# Update the dependency injection functions
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

def get_llm_client():
    """Dependency to get LLM client instance"""
    from main import llm_client
    return llm_client

def get_tutorial_generator(
    llm_client = Depends(get_llm_client),
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator)
):
    """Dependency injection for TutorialGenerator"""
    return TutorialGenerator(llm_client, vector_store, embedding_generator)

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

# In-memory task storage (replace with proper database in production)
tutorial_tasks = {}

class TutorialGenerationStatus(BaseModel):
    task_id: str
    status: str
    tutorial: Optional[ProcessedTutorial] = None
    error: Optional[str] = None

async def generate_tutorial_task(
    content_id: str,
    collection_name: str,
    task_id: str,
    tutorial_generator: TutorialGenerator
):
    """Background task for tutorial generation"""
    tutorial_tasks[task_id] = {"status": "processing"}
    
    try:
        tutorial = await tutorial_generator.generate_tutorial(
            content_id,
            collection_name
        )
        
        tutorial_tasks[task_id] = {
            "status": "completed",
            "tutorial": tutorial
        }
        
    except Exception as e:
        tutorial_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }

@router.post("/generate", response_model=TutorialGenerationStatus)
async def generate_tutorial(
    request: TutorialGenerationRequest,
    background_tasks: BackgroundTasks,
    tutorial_generator: TutorialGenerator = Depends(get_tutorial_generator)
):
    """Start tutorial generation"""
    task_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        generate_tutorial_task,
        request.content_id,
        request.content_type,
        task_id,
        tutorial_generator
    )
    
    return TutorialGenerationStatus(
        task_id=task_id,
        status="processing"
    )

@router.get("/status/{task_id}", response_model=TutorialGenerationStatus)
async def get_tutorial_status(task_id: str):
    """Get tutorial generation status"""
    if task_id not in tutorial_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tutorial_tasks[task_id]
    return TutorialGenerationStatus(
        task_id=task_id,
        status=task["status"],
        tutorial=task.get("tutorial"),
        error=task.get("error")
    )

@router.get("/content/{tutorial_id}", response_model=ProcessedTutorial)
async def get_tutorial_content(
    tutorial_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get a specific tutorial by ID"""
    try:
        tutorial_data = vector_store.get_by_id("tutorial", tutorial_id)
        if not tutorial_data or not tutorial_data["documents"]:
            raise HTTPException(status_code=404, detail="Tutorial not found")
            
        # Convert stored data back to ProcessedTutorial
        # This is a simplified example - you'll need to adjust based on your storage format
        metadata = tutorial_data["metadatas"][0]
        return ProcessedTutorial(
            title=metadata["title"],
            summary=["Summary from stored data"],
            key_points=["Key points from stored data"],
            code_examples=[],  # Parse from stored data
            practice_exercises=[],  # Parse from stored data
            source_url=metadata["source_url"],
            content_type=metadata["content_type"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
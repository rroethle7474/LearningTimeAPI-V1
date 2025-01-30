from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from typing import Optional, List
from pydantic import BaseModel, HttpUrl
from typing_extensions import Literal
import uuid
from processors.content_processor import ContentProcessor
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator
import os
from dotenv import load_dotenv
from config import settings
import logging

# Load environment variables from .env file
load_dotenv()

router = APIRouter()

# In-memory task storage (replace with proper database in production)
tasks = {}

logger = logging.getLogger(__name__)

class URLSubmission(BaseModel):
    url: HttpUrl
    content_type: Literal["article", "youtube"]

class TaskStatus(BaseModel):
    task_id: str
    status: str
    error: Optional[str] = None

class ContentMetadataResponse(BaseModel):
    title: str
    author: str
    source_url: str
    content_type: Literal["article", "youtube"]
    duration: Optional[str] = None
    published_date: Optional[str] = None
    view_count: Optional[int] = None

class ProcessedContent(BaseModel):
    metadata: ContentMetadataResponse
    chunks: List[str]

def generate_task_id() -> str:
    return str(uuid.uuid4())

async def process_content_task(
    url: str | HttpUrl,
    content_type: str,
    task_id: str,
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator
):
    """Background task for content processing"""
    logger.debug(f"Starting content processing task {task_id} for URL: {url}")
    tasks[task_id] = {"status": "processing"}
    
    try:
        async with ContentProcessor(youtube_api_key=settings.YOUTUBE_API_KEY) as processor:
            logger.debug("Created ContentProcessor instance")
            
            # Convert URL to string if it's a HttpUrl object
            url_str = str(url)
            
            # Process content
            metadata, chunks = await processor.process_content(url_str, content_type)
            logger.debug(f"Successfully processed content, got {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = processor.generate_embeddings(chunks)
            logger.debug("Generated embeddings")
            
            # Store in vector store
            collection = vector_store.get_collection(content_type)
            logger.debug(f"Storing in {content_type} collection")
            
            try:
                # Convert metadata to dict and ensure all values are primitive types
                # Filter out None values or replace with defaults
                metadata_dict = {
                    "title": metadata.title or "",  # Use empty string if None
                    "author": metadata.author or "Unknown",
                    "source_url": str(metadata.source_url),
                    "content_type": metadata.content_type,
                    "duration": metadata.duration or "",
                    "published_date": metadata.published_date or "",
                    "view_count": metadata.view_count or 0  # Use 0 for numeric field
                }
                logger.debug(f"Converted metadata to dict: {metadata_dict}")
                
                collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=[metadata_dict] * len(chunks),
                    ids=[f"{task_id}_{i}" for i in range(len(chunks))]
                )
                logger.debug("Successfully stored in vector store")
                
            except Exception as e:
                logger.error(f"Error storing in vector store: {str(e)}", exc_info=True)
                raise
            
            tasks[task_id] = {
                "status": "completed",
                "metadata": metadata_dict,
                "chunks": chunks
            }
            
    except Exception as e:
        logger.error(f"Error in process_content_task: {str(e)}", exc_info=True)
        tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }

# Move VectorStore initialization to a dependency
def get_vector_store():
    return VectorStore()

@router.post("/submit", response_model=TaskStatus)
async def submit_content(
    submission: URLSubmission,
    background_tasks: BackgroundTasks,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Submit content for processing"""
    try:
        task_id = str(uuid.uuid4())
        print(f"Task ID: {task_id}")
        background_tasks.add_task(
            process_content_task,
            str(submission.url),  # Convert to string here
            submission.content_type,
            task_id,
            vector_store,
            EmbeddingGenerator()
        )
        return TaskStatus(task_id=task_id, status="processing")
        
    except Exception as e:
        return TaskStatus(
            task_id=str(uuid.uuid4()),
            status="error",
            error=str(e)
        )

@router.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task = tasks[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        result=task.get("result"),
        error=task.get("error")
    )

@router.get("/{task_id}", response_model=ProcessedContent)
async def get_processed_content(task_id: str):
    """Get the processed content for a completed task"""
    if task_id not in tasks:
        logger.debug(f"Task {task_id} not found in tasks")
        raise HTTPException(status_code=404, detail="Content not found")
        
    task = tasks[task_id]
    logger.debug(f"Found task: {task}")
    
    if task["status"] != "completed":
        logger.debug(f"Task not completed. Current status: {task['status']}")
        raise HTTPException(
            status_code=400, 
            detail=f"Content processing not completed. Status: {task['status']}"
        )
    
    response = ProcessedContent(
        metadata=ContentMetadataResponse(**task["metadata"]),
        chunks=task["chunks"]
    )
    logger.debug(f"Returning response: {response.model_dump_json(indent=2)}")
    return response 
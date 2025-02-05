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
from datetime import datetime
from llm.base import LLMClient
from llm.factory import create_llm_client
# Import dependencies
from dependencies import (
    get_vector_store,
    get_embedding_generator,
    get_llm_client
)

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
    content_id: Optional[str] = None
    error: Optional[str] = None
    summary: Optional[str] = None

class ContentMetadataResponse(BaseModel):
    title: str
    author: str
    source_url: str
    content_type: Literal["article", "youtube"]
    duration: Optional[str] = None
    published_date: Optional[str] = None
    view_count: Optional[int] = None
    summary: Optional[str] = None
    processed_date: Optional[datetime] = None

class ProcessedContent(BaseModel):
    content_id: str
    metadata: ContentMetadataResponse
    chunks: List[str]

def generate_task_id() -> str:
    return str(uuid.uuid4())

async def process_content_task(
    url: str | HttpUrl,
    content_type: str,
    task_id: str,
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator,
    llm_client: LLMClient
):
    """Background task for content processing"""
    logger.debug(f"Starting content processing task {task_id} for URL: {url}")
    tasks[task_id] = {"status": "processing"}
    
    try:
        async with ContentProcessor(
            youtube_api_key=settings.YOUTUBE_API_KEY,
            llm_client=llm_client
        ) as processor:
            logger.debug("Created ContentProcessor instance")
            
            # Convert URL to string if it's a HttpUrl object
            url_str = str(url)
            
            # Process content and get metadata with summary
            metadata, chunks = await processor.process_content(url_str, content_type)
            logger.debug(f"Successfully processed content, got {len(chunks)} chunks")
            
            # Update task status with summary immediately after generation
            tasks[task_id].update({
                "status": "processing",
                "summary": metadata.summary  # Make summary available early
            })
            
            # Generate embeddings
            embeddings = processor.generate_embeddings(chunks)
            logger.debug("Generated embeddings")
            
            # Store in vector store
            print(f"Storing in {content_type}")
            if content_type == "article":
                collection_name = "articles_content"
            elif content_type == "youtube":
                collection_name = "youtube_content"
            else:
                raise ValueError(f"Invalid content type: {content_type}")
            
            logger.debug(f"Storing in {collection_name} collection")
            

            try:
                # Generate a content ID
                content_id = str(uuid.uuid4())
                logger.info(f"Generated content ID: {content_id}")
                
                metadata_dict = {
                    "content_id": content_id,
                    "title": metadata.title or "",
                    "author": metadata.author or "Unknown",
                    "source_url": str(metadata.source_url),
                    "content_type": metadata.content_type,
                    "duration": metadata.duration or "",
                    "published_date": metadata.published_date or "",
                    "view_count": metadata.view_count or 0,
                    "summary": metadata.summary,
                    "processed_date": metadata.processed_date.isoformat() if metadata.processed_date else None
                }
                
                # Store using new method with correct collection name
                vector_store.add_content(
                    content_id=content_id,
                    content_type=content_type,  # Use the original content_type instead of collection_name
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata=metadata_dict
                )
                logger.info(f"Successfully stored content with ID {content_id} in vector store")
                
                # Update final task status
                tasks[task_id] = {
                    "status": "completed",
                    "content_id": content_id,
                    "metadata": metadata_dict,
                    "chunks": chunks,
                    "summary": metadata.summary  # Keep summary in final status
                }
                logger.info(f"Updated task {task_id} with content_id {content_id}")
                
            except Exception as e:
                logger.error(f"Error storing content: {str(e)}", exc_info=True)
                raise
            
    except Exception as e:
        logger.error(f"Error in process_content_task: {str(e)}", exc_info=True)
        tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "summary": None  # Clear summary on error
        }

@router.post("/submit", response_model=TaskStatus)
async def submit_content(
    submission: URLSubmission,
    background_tasks: BackgroundTasks,
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator),
    llm_client: LLMClient = Depends(get_llm_client)
):
    """Submit content for processing"""
    try:
        task_id = str(uuid.uuid4())
        print(f"Task ID: {task_id}")
        background_tasks.add_task(
            process_content_task,
            str(submission.url),
            submission.content_type,
            task_id,
            vector_store,
            embedding_generator,
            llm_client
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
    """Get content processing task status"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task = tasks[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        content_id=task.get("content_id"),
        error=task.get("error"),
        summary=task.get("summary")
    )

@router.get("/{task_id}", response_model=ProcessedContent)
async def get_processed_content(
    task_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
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
        content_id=task["content_id"],
        metadata=ContentMetadataResponse(**task["metadata"]),
        chunks=task["chunks"]
    )
    logger.debug(f"Returning response: {response.model_dump_json(indent=2)}")
    return response 

@router.get("/debug/collection/{collection_name}")
async def inspect_collection(
    collection_name: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Debug endpoint to inspect collection contents"""
    try:
        inspection = vector_store.inspect_collection(collection_name)
        return inspection
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
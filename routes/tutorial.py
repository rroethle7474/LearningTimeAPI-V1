from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status, Path, Query
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from pydantic import BaseModel
import uuid
from generators.tutorial import TutorialGenerator, ProcessedTutorial
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator
from app_types.tutorial import TutorialSectionType
# Import dependencies
from dependencies import (
    get_vector_store,
    get_embedding_generator,
    get_llm_client,
    get_tutorial_generator
)
from urllib.parse import unquote

router = APIRouter(tags=["tutorials"])

# Remove all these local dependency definitions since we're importing them
# def get_embedding_generator():...
# def get_vector_store():...
# def get_llm_client():...
# def get_tutorial_generator():...

class TutorialGenerationRequest(BaseModel):
    content_id: str
    content_type: Literal["article", "youtube"]

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
    tutorial_generator: TutorialGenerator = Depends(get_tutorial_generator)  # Using imported dependency
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

class TutorialSection(BaseModel):
    section_id: str
    tutorial_id: str
    title: str
    content: str
    section_type: TutorialSectionType
    order: int
    metadata: Dict[str, Any]

class TutorialResponse(BaseModel):
    id: str
    title: str
    description: str
    source_content_id: Optional[str]
    source_type: Optional[str]  # "article" or "youtube"
    source_url: Optional[str]
    generated_date: datetime
    sections: List[TutorialSection]
    metadata: Dict[str, Any]

class TutorialListItem(BaseModel):
    id: str
    title: str
    description: str
    generated_date: datetime
    source_type: Optional[str]
    section_count: int
    metadata: Dict[str, Any]
    source_url: Optional[str] = ""

class TutorialListResponse(BaseModel):
    total: int
    items: List[TutorialListItem]

@router.get("/tutorials", response_model=TutorialListResponse)
async def list_tutorials(
    offset: int = Query(0, description="Number of records to skip"),
    limit: int = Query(50, description="Maximum number of records to return"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get paginated list of tutorials"""
    try:
        contents = vector_store.get_collection_contents(
            collection_name="tutorials",
            offset=offset,
            limit=limit
        )
        
        items = []
        for item in contents["items"]:
            tutorial_data = item.get("document")
            if isinstance(tutorial_data, str):
                import json
                tutorial_data = json.loads(tutorial_data)
            
            # Get metadata from the tutorial_data
            tutorial_metadata = tutorial_data.get("metadata", {})
            # Get the first section's content as description (usually the summary)
            description = ""
            sections = tutorial_data.get("sections", [])
            if sections and sections[0]["type"] == "summary":
                description = sections[0]["content"]

            items.append(TutorialListItem(
                id=item["id"],
                title=tutorial_metadata.get("title", "Untitled"),  # Title is in metadata
                description=description,  # Use summary section as description
                generated_date=datetime.fromisoformat(tutorial_metadata["generated_date"]),
                source_type=tutorial_metadata.get("content_type"),  # content_type is in metadata
                section_count=len(sections),
                metadata=item.get("metadata", {}),
                source_url=tutorial_metadata.get("source_url", "")
            ))
            


        tutorial_list_response = TutorialListResponse(
            total=contents["total"],
            items=items,
        )
        
        return tutorial_list_response
        
    except Exception as e:
        print(f"Error in list_tutorials: {str(e)}")  # Add debug logging
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tutorials/{tutorial_id}", response_model=TutorialResponse)
async def get_tutorial_detail(
    tutorial_id: str = Path(..., description="ID of the tutorial to retrieve"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get complete tutorial with all sections"""
    try:
        tutorial_data = vector_store.get_tutorial_with_sections(tutorial_id)
        if not tutorial_data:
            raise HTTPException(status_code=404, detail="Tutorial not found")
        
        # Convert the tutorial data into our response model
        sections = []
        for i, section in enumerate(tutorial_data["sections"]):
            sections.append(TutorialSection(
                section_id=section["id"],
                tutorial_id=tutorial_id,
                title=section["title"],
                content=section["content"],
                section_type=section["type"],
                order=i,  # Use the index as the order
                metadata=section["metadata"]
            ))
        
        # Get metadata fields from the correct location
        metadata = tutorial_data["metadata"]
        return TutorialResponse(
            id=tutorial_id,
            title=metadata["title"],
            description=sections[0].content if sections else "",  # Use first section (summary) as description
            source_content_id=metadata.get("content_id"),
            source_type=metadata.get("content_type"),
            source_url=metadata.get("source_url"),
            generated_date=metadata["generated_date"],
            sections=sorted(sections, key=lambda x: x.order),
            metadata=metadata
        )
        
    except Exception as e:
        print(f"Error in get_tutorial_detail: {str(e)}")  # Add debug logging
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/tutorials/{tutorial_id}", status_code=204)
async def delete_tutorial(
    tutorial_id: str = Path(..., description="ID of the tutorial to delete"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Delete a tutorial and its sections.
    Returns 204 on success with no content.
    """
    try:
        # First verify the tutorial exists
        tutorial = vector_store.get_tutorial_with_sections(tutorial_id)
        if not tutorial:
            raise HTTPException(status_code=404, detail="Tutorial not found")
            
        print(f"Deleting tutorial {tutorial_id} with title: {tutorial['metadata'].get('title', 'Unknown')}")
        
        # Delete the tutorial from tutorials collection
        vector_store.tutorials.delete(
            ids=[tutorial_id]
        )
        
        # Delete all associated sections
        section_ids = [section["id"] for section in tutorial["sections"]]
        if section_ids:
            print(f"Deleting {len(section_ids)} sections for tutorial {tutorial_id}")
            vector_store.tutorial_sections.delete(
                ids=section_ids
            )
        
        print(f"Successfully deleted tutorial {tutorial_id} and its sections")
        return None  # 204 No Content
        
    except Exception as e:
        print(f"Error deleting tutorial: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/tutorials/content/{content_id}", status_code=204)
async def delete_tutorial_by_content(
    content_id: str = Path(..., description="ID of the source content"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Delete a tutorial and its sections based on the source content ID.
    Returns 204 on success with no content.
    """
    try:
        tutorial_id = vector_store.delete_tutorial_by_content_id(content_id)
        print(f"Successfully deleted tutorial for content {content_id} (tutorial_id: {tutorial_id})")
        return None  # 204 No Content
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error deleting tutorial by content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/tutorials/url", status_code=204)
async def delete_tutorial_by_url(
    source_url: str = Query(..., description="Source URL of the content"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Delete a tutorial and its sections based on the source URL.
    Returns 204 on success with no content.
    """
    try:
        # URL decode the source_url parameter
        decoded_url = unquote(source_url)
        tutorial_id = vector_store.delete_tutorial_by_source_url(decoded_url)
        print(f"Successfully deleted tutorial for URL {decoded_url} (tutorial_id: {tutorial_id})")
        return None  # 204 No Content
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error deleting tutorial by URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 
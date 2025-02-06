from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, Depends, HTTPException
from typing import List, Optional
import uuid
from datetime import datetime
import os

from db.vector_store import VectorStore
from processors.document_processor import DocumentProcessor
from embeddings.generator import EmbeddingGenerator
from app_types.document import DocumentStatus, DocumentMetadata
from dependencies import get_vector_store, get_embedding_generator

router = APIRouter()

async def process_document_task(
    title: str,
    tags: str,
    file_content: bytes,
    filename: str,
    task_id: str,
    vector_store: VectorStore,
    document_processor: DocumentProcessor,
    embedding_generator: EmbeddingGenerator
) -> None:
    """Background task to process and store document"""
    try:
        # Extract content from document
        content = await document_processor.process_document(file_content, filename)
        
        # Generate embeddings
        embeddings = await embedding_generator.generate_embeddings(content)
        
        # Create document metadata
        metadata = DocumentMetadata(
            title=title,
            tags=tags.split(',') if tags else [],
            file_type=os.path.splitext(filename)[1],
            file_size=len(file_content),
            upload_date=datetime.utcnow(),
            source_file=filename
        )
        
        # Store in vector database
        document_id = str(uuid.uuid4())
        vector_store.add_document(
            document_id=document_id,
            content=content,
            metadata=metadata,
            embeddings=embeddings
        )
        
        # Update task status
        # Note: In a production system, you'd want to store task statuses in a database
        return document_id
        
    except Exception as e:
        raise Exception(f"Document processing failed: {str(e)}")

@router.post("/upload", response_model=DocumentStatus)
async def upload_document(
    background_tasks: BackgroundTasks,
    vector_store: VectorStore = Depends(get_vector_store),
    document_processor: DocumentProcessor = Depends(lambda: DocumentProcessor()),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator),
    title: str = Form(...),
    tags: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload and process a document file
    """
    # Validate file extension
    allowed_extensions = {'.txt', '.pdf', '.doc', '.docx'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Read file content
        file_content = await file.read()
        
        # Start background processing
        document_id = await process_document_task(
            title=title,
            tags=tags,
            file_content=file_content,
            filename=file.filename,
            task_id=task_id,
            vector_store=vector_store,
            document_processor=document_processor,
            embedding_generator=embedding_generator
        )
        
        return DocumentStatus(
            task_id=task_id,
            status="completed",
            document_id=document_id
        )
        
    except Exception as e:
        return DocumentStatus(
            task_id=task_id,
            status="failed",
            error=str(e)
        )

@router.get("/{document_id}")
async def get_document(
    document_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Retrieve a document by ID
    """
    document = vector_store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Delete a document by ID
    """
    try:
        vector_store.delete_document(document_id)
        return {"status": "success", "message": "Document deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/")
async def search_documents(
    query: str,
    limit: int = 5,
    tags: Optional[str] = None,
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator)
):
    """
    Search documents by content and optional tags
    """
    try:
        # Generate query embedding
        query_embedding = await embedding_generator.generate_embeddings(query)
        
        # Prepare filters
        filters = {}
        if tags:
            # Note: This is a simple implementation. In practice, you might want
            # more sophisticated tag matching
            filters["tags"] = {"$contains": tags}
        
        # Search documents
        results = vector_store.search_documents(
            query_embedding=query_embedding,
            limit=limit,
            filters=filters
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
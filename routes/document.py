from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, Depends, HTTPException
from typing import List, Optional
import uuid
from datetime import datetime
import os
from fastapi.responses import JSONResponse

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
        # Extract content and chunks from document
        full_text, chunks = await document_processor.process_document(file_content, filename)
        print("CHUNKS", chunks)
        # Generate embeddings for each chunk
        chunk_embeddings = []
        for chunk in chunks:
            embedding = embedding_generator.generate(chunk)
            # Ensure we get a flat 1D list by recursively flattening
            while isinstance(embedding, list) and isinstance(embedding[0], list):
                embedding = embedding[0]
            
            # Add some validation
            if not isinstance(embedding, list):
                raise ValueError(f"Expected list of floats, got {type(embedding)}")
            if not all(isinstance(x, (int, float)) for x in embedding):
                raise ValueError("All elements must be numbers")
                
            chunk_embeddings.append(embedding)
        print("CHUNK EMBEDDINGS", chunk_embeddings)
        # Create base metadata
        base_metadata = DocumentMetadata(
            title=title,
            tags=tags.split(',') if tags else [],
            file_type=os.path.splitext(filename)[1],
            file_size=len(file_content),
            upload_date=datetime.utcnow(),
            source_file=filename
        )
        print("BASE METADATA", base_metadata)
        # Generate document ID
        document_id = str(uuid.uuid4())
        print("DOCUMENT ID", document_id)
        # Store chunks with metadata
        chunk_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            # Create a new DocumentMetadata instance for the chunk
            chunk_metadata = DocumentMetadata(
                title=base_metadata.title,
                tags=base_metadata.tags,
                file_type=base_metadata.file_type,
                file_size=base_metadata.file_size,
                upload_date=base_metadata.upload_date,
                source_file=base_metadata.source_file
            )
            
            # Add chunk-specific fields to metadata dict after conversion
            chunk_metadata_dict = chunk_metadata.dict()
            chunk_metadata_dict.update({
                "document_id": document_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "is_chunk": True
            })
            print("CHUNK METADATA DICT", chunk_metadata_dict)
            vector_store.add_document(
                document_id=chunk_id,
                content=chunk,
                metadata=chunk_metadata,  # Pass the DocumentMetadata object
                embeddings=embedding
            )
        print("CHUNK METADATA", chunk_metadata)
        # Store full document metadata
        full_doc_metadata = DocumentMetadata(
            title=base_metadata.title,
            tags=base_metadata.tags,
            file_type=base_metadata.file_type,
            file_size=base_metadata.file_size,
            upload_date=base_metadata.upload_date,
            source_file=base_metadata.source_file
        )
        print("FULL DOC METADATA", full_doc_metadata)
        # Add full document specific fields
        full_metadata_dict = full_doc_metadata.dict()
        # Convert tags list to comma-separated string
        full_metadata_dict["tags"] = ",".join(full_metadata_dict["tags"])
        # Convert datetime to ISO format string
        full_metadata_dict["upload_date"] = full_metadata_dict["upload_date"].isoformat()
        # Convert chunk_ids list to comma-separated string
        full_metadata_dict.update({
            "chunk_ids": ",".join(chunk_ids),  # Convert list of chunk IDs to string
            "is_chunk": False
        })
        print("FULL METADATA DICT", full_metadata_dict)
        # Store full document (without embeddings)
        vector_store.store_full_document(
            document_id=document_id,
            content=full_text,
            metadata=full_metadata_dict
        )
        
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
    print("FILE EXTENSION", file_ext)
    if file_ext not in allowed_extensions:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "error": f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}",
                "task_id": None,
                "document_id": None
            }
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
        print("DOCUMENT ID", document_id)
        return JSONResponse(
            status_code=200,
            content={
                "status": "completed",
                "task_id": task_id,
                "document_id": document_id,
                "error": None
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "task_id": task_id,
                "document_id": None,
                "error": str(e)
            }
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

@router.get("/status/{task_id}", response_model=DocumentStatus)
async def get_document_status(
    task_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Check the status of a document processing task
    """
    try:
        # For now, since we're processing synchronously, we can just check if the document exists
        # In a production system, you'd want to check a task queue or database
        if hasattr(vector_store, 'task_statuses') and task_id in vector_store.task_statuses:
            return vector_store.task_statuses[task_id]
            
        return DocumentStatus(
            task_id=task_id,
            status="not_found",
            error="Task not found"
        )
    except Exception as e:
        return DocumentStatus(
            task_id=task_id,
            status="error",
            error=str(e)
        ) 
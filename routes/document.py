from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import List, Optional
import uuid
from datetime import datetime
import os
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from config import settings

from db.vector_store import VectorStore
from processors.document_processor import DocumentProcessor
from embeddings.generator import EmbeddingGenerator
from app_types.document import DocumentStatus, DocumentMetadata
from dependencies import get_vector_store, get_embedding_generator, get_document_processor

router = APIRouter()

def sanitize_filename(filename: str) -> str:
    """Convert filename to a safe version"""
    # Replace spaces with underscores and remove any non-alphanumeric characters except .-_
    safe_filename = "".join(c for c in filename.replace(' ', '_') if c.isalnum() or c in '.-_')
    return safe_filename

@router.post("/upload", response_model=DocumentStatus)
async def upload_document(
    vector_store: VectorStore = Depends(get_vector_store),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator),
    title: str = Form(...),
    tags: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload and process a document file
    """
    try:
        # Validate file extension
        allowed_extensions = {'.txt', '.pdf', '.doc', '.docx'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failed",
                    "error": f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}",
                    "document_id": None
                }
            )
        
        # Create upload directory if it doesn't exist
        upload_dir = Path(settings.DOCUMENT_UPLOAD_PATH)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate safe filename from original filename without timestamp
        original_filename = file.filename
        safe_filename = sanitize_filename(original_filename)
        
        # Check if file already exists
        file_path = upload_dir / safe_filename
        if file_path.exists():
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failed",
                    "error": "File already has been processed. Please see Explore Documents to remove if re-processing is needed.",
                    "document_id": None
                }
            )
        
        # Read and save file content
        file_content = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Process document
        full_text, chunks, embeddings = await document_processor.process_document(file_content, file.filename)
        
        # Create base metadata
        base_metadata = DocumentMetadata(
            title=title,
            tags=tags.split(',') if tags else [],
            file_type=file_ext,
            file_size=len(file_content),
            upload_date=datetime.utcnow(),
            source_file=str(file_path),
            original_filename=original_filename  # Make sure this is included
        )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Store chunks with metadata
        chunk_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            # Create chunk metadata - include all required fields
            chunk_metadata = DocumentMetadata(
                title=base_metadata.title,
                tags=base_metadata.tags,
                file_type=base_metadata.file_type,
                file_size=base_metadata.file_size,
                upload_date=base_metadata.upload_date,
                source_file=base_metadata.source_file,
                original_filename=base_metadata.original_filename  # Include this for chunks too
            )
            
            # Add chunk-specific fields
            chunk_metadata_dict = chunk_metadata.dict()
            chunk_metadata_dict.update({
                "document_id": document_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "is_chunk": True
            })
            
            vector_store.add_document(
                document_id=chunk_id,
                content=chunk,
                metadata=chunk_metadata,  # Pass the DocumentMetadata object directly
                embeddings=[embedding]
            )
        
        # Store full document
        full_doc_metadata = base_metadata.dict()
        full_doc_metadata["tags"] = ",".join(full_doc_metadata["tags"])
        full_doc_metadata["upload_date"] = full_doc_metadata["upload_date"].isoformat()
        full_doc_metadata.update({
            "chunk_ids": ",".join(chunk_ids),
            "is_chunk": False
        })
        
        vector_store.store_full_document(
            document_id=document_id,
            content=full_text,
            metadata=full_doc_metadata
        )
        
        return DocumentStatus(
            status="completed",
            document_id=document_id
        )
        
    except Exception as e:
        # If there's an error, clean up the saved file
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        return DocumentStatus(
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
        # Get document metadata first
        document = vector_store.get_document(document_id)
        if document and document.get("metadata"):
            # Delete local file if it exists
            file_path = Path(document["metadata"]["source_file"])
            if file_path.exists():
                file_path.unlink()
        
        # Delete from vector store
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
        query_embedding = embedding_generator.generate([query])[0]
        
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

@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Download the original document file by document ID
    """
    # Get document metadata from vector store
    document = vector_store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get source file path from metadata
    source_file = Path(document["metadata"]["source_file"])
    if not source_file.exists():
        raise HTTPException(
            status_code=404, 
            detail="Original file no longer exists on server"
        )
    
    # Return the file as a download response
    return FileResponse(
        path=source_file,
        filename=document["metadata"].get("original_filename", source_file.name),
        media_type="application/octet-stream"
    ) 
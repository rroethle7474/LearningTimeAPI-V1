from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import List, Optional
import uuid
from datetime import datetime
import os
from fastapi.responses import JSONResponse

from db.vector_store import VectorStore
from processors.document_processor import DocumentProcessor
from embeddings.generator import EmbeddingGenerator
from app_types.document import DocumentStatus, DocumentMetadata
from dependencies import get_vector_store, get_embedding_generator, get_document_processor

router = APIRouter()

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
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process document
        full_text, chunks, embeddings = await document_processor.process_document(file_content, file.filename)
        
        # Create base metadata
        base_metadata = DocumentMetadata(
            title=title,
            tags=tags.split(',') if tags else [],
            file_type=os.path.splitext(file.filename)[1],
            file_size=len(file_content),
            upload_date=datetime.utcnow(),
            source_file=file.filename
        )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Store chunks with metadata
        chunk_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            # Create chunk metadata
            chunk_metadata = DocumentMetadata(
                title=base_metadata.title,
                tags=base_metadata.tags,
                file_type=base_metadata.file_type,
                file_size=base_metadata.file_size,
                upload_date=base_metadata.upload_date,
                source_file=base_metadata.source_file
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
                metadata=chunk_metadata,
                embeddings=embedding
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
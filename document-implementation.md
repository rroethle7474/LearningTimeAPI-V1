# Document Collection Implementation Plan

## Overview
Implement document processing and storage in ChromaDB with async processing, metadata tracking, and tag-based relevance boosting.

## Initial Prompt
Hi Claude!

Can you help me develop a development plan to implement a new collection I'd like to add to my chromdb (vector_store.py file) called "documents".

The functionality I'd like to implement is have my front end pass in a document (either a .txt, .pdf, .doc, or .docx file) and have this processed and stored in chroma db for future processing. It should have the same functions as the search or content where i can retreive by id and delete.

I'll need a way to extract the data from the document based on the file extension and then save to chromadb.

The tricky other part I want to implement is to add a title and tags to the document so it can increase relevant scores when being searched.

Can you review the project knowledge, and help me think through a document project plan where we list out the tasks we need to implement and details for achieving each task? Let's focus first on the backend knowing that we will be getting a title, tags (seperated by comma's), and the file.

I think we need to build the endpoint, service for extracting content, methods to save, methods to retrieve based on search, add this collection to existing search methods (semantic_search and context_generation_service), and then allow for this to be returned via the SearchResponseDto.

Feel free to ask me any questions first before creating the markdown document.

let's have some fun!

## Type Definitions

```python
# models/document.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DocumentMetadata(BaseModel):
    title: str
    tags: List[str]
    file_type: str
    file_size: int
    upload_date: datetime
    source_file: str

class DocumentSubmission(BaseModel):
    title: str
    tags: str  # Comma-separated string
    file: bytes  # File content

class DocumentStatus(BaseModel):
    task_id: str
    status: str
    document_id: Optional[str] = None
    error: Optional[str] = None
```

## Implementation Tasks

### 1. Document Processor Implementation
```python
# processors/document_processor.py
class DocumentProcessor:
    def __init__(self):
        self.extractors = {
            '.txt': self._process_text,
            '.pdf': self._process_pdf,
            '.doc': self._process_doc,
            '.docx': self._process_docx
        }
    
    async def process_document(self, file_content: bytes, filename: str) -> str:
        ext = self._get_extension(filename)
        if ext not in self.extractors:
            raise ValueError(f"Unsupported file type: {ext}")
            
        return await self.extractors[ext](file_content)
    
    async def _process_text(self, content: bytes) -> str:
        return content.decode('utf-8')
    
    async def _process_pdf(self, content: bytes) -> str:
        # Use PyPDF2 or pdfplumber
        pass
    
    async def _process_doc(self, content: bytes) -> str:
        # Use python-docx2txt
        pass
    
    async def _process_docx(self, content: bytes) -> str:
        # Use python-docx
        pass
```

### 2. Update VectorStore
```python
# db/vector_store.py
class VectorStore:
    def _init_collections(self):
        self.documents = self.client.get_or_create_collection("documents")
        # ... existing collections ...
    
    def add_document(
        self,
        document_id: str,
        content: str,
        metadata: DocumentMetadata,
        embeddings: List[float]
    ):
        """Add document to vector store with metadata and tags"""
        self.documents.add(
            ids=[document_id],
            documents=[content],
            embeddings=[embeddings],
            metadatas=[{
                "title": metadata.title,
                "tags": ",".join(metadata.tags),
                "file_type": metadata.file_type,
                "file_size": metadata.file_size,
                "upload_date": metadata.upload_date.isoformat(),
                "source_file": metadata.source_file
            }]
        )
```

### 3. Document Routes
```python
# routes/document.py
@router.post("/submit", response_model=DocumentStatus)
async def submit_document(
    title: str = Form(...),
    tags: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks,
    vector_store: VectorStore = Depends(get_vector_store)
):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(
        process_document_task,
        title,
        tags,
        await file.read(),
        file.filename,
        task_id,
        vector_store
    )
    return DocumentStatus(task_id=task_id, status="processing")
```

### 4. Update Search Service
```python
# search/semantic_search.py
class SemanticSearch:
    async def search(
        self,
        query: str,
        collection: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # Add tag relevance boost
        if collection == "documents":
            results = await self._search_with_tags(query, limit, filters)
        else:
            # Existing search logic
            pass
    
    async def _search_with_tags(self, query: str, limit: int, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract potential tags from query
        query_tags = self._extract_tags(query)
        
        # Perform base search
        results = self.vector_store.search_collection(...)
        
        # Apply tag boost (simple constant boost)
        TAG_BOOST = 0.1
        for result in results:
            doc_tags = set(result["metadata"]["tags"].split(","))
            matching_tags = len(query_tags.intersection(doc_tags))
            result["distance"] *= (1 - (TAG_BOOST * matching_tags))
        
        return results
```

## Dependencies
Add to requirements.txt:
```
PyPDF2==3.0.1
python-docx==0.8.11
python-docx2txt==0.8
python-multipart==0.0.5
```

## Migration Steps
1. Add document collection to ChromaDB 
2. Implement DocumentProcessor
3. Update VectorStore with document methods
4. Add document routes
5. Update search service
6. Add tests
7. Update API documentation

## Testing Plan
1. Unit tests for each file type processor
2. Integration tests for document upload flow
3. Search relevance tests with/without tags
4. Error handling tests
5. Large file handling tests

## Next Steps
1. Implement document processors for each file type
2. Add the new collection to ChromaDB
3. Create routes for document upload and retrieval
4. Update search to handle document results
5. Add API documentation
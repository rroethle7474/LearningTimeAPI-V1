# Backend API Implementation Guide

## Overview
This guide outlines the implementation details for the FastAPI backend with ChromaDB integration, focusing on content processing, tutorial generation, and search functionality.

## System Architecture

### Core Components
1. FastAPI Application
2. ChromaDB Vector Store
3. Sentence Transformers for Embeddings
4. Content Processors (Playwright, YouTube API)
5. LLM Integration Layer

### ChromaDB Collections Structure
```
ChromaDB Instance
├── articles_content     # Stores article content and embeddings
├── youtube_content     # Stores video transcripts and embeddings
├── tutorials          # Stores generated tutorials
└── metadata          # Stores processing status and additional info
```

## Implementation Tasks

### Epic 1: Core API Setup
**Task 1.1: FastAPI Project Setup**
- Priority: High
- Dependencies: None
- Description: Initialize FastAPI project with required dependencies
```bash
pip install fastapi uvicorn chromadb sentence-transformers playwright python-youtube langchain pydantic python-dotenv
```

**Task 1.2: Environment Configuration**
- Priority: High
- Dependencies: 1.1
- Description: Setup environment configuration
```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    YOUTUBE_API_KEY: str
    OPENAI_API_KEY: str  # For OpenAI LLM (optional)
    ANTHROPIC_API_KEY: str  # For Claude LLM (optional)
    CHROMADB_PATH: str = "./chromadb"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Epic 2: Content Processing

**Task 2.1: URL Submission Endpoints**
- Priority: High
- Dependencies: 1.1, 1.2
```python
# routes/content.py
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, HttpUrl

router = APIRouter()

class URLSubmission(BaseModel):
    url: HttpUrl
    content_type: Literal["article", "youtube"]

@router.post("/submit")
async def submit_content(
    submission: URLSubmission,
    background_tasks: BackgroundTasks
):
    task_id = generate_task_id()
    background_tasks.add_task(
        process_content,
        submission.url,
        submission.content_type,
        task_id
    )
    return {"task_id": task_id, "status": "processing"}
```

**Task 2.2: Content Processor Implementation**
- Priority: High
- Dependencies: 2.1
- Description: Implement content extraction for articles and YouTube videos
```python
# processors/content_processor.py
class ContentProcessor:
    def __init__(self):
        self.playwright = None
        self.youtube_client = None

    async def process_article(self, url: str):
        # Initialize Playwright
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch()
        page = await browser.new_page()
        
        # Extract content
        await page.goto(url)
        content = await page.content()
        
        # Process with trafilaret or similar
        # Return structured content

    async def process_youtube(self, url: str):
        video_id = extract_video_id(url)
        # Use YouTube API to get transcript
        # Return structured content
```

### Epic 3: Vector Storage Integration

**Task 3.1: ChromaDB Setup**
- Priority: High
- Dependencies: 1.1
```python
# db/vector_store.py
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chromadb"
        ))
        self._init_collections()

    def _init_collections(self):
        self.articles = self.client.get_or_create_collection("articles_content")
        self.youtube = self.client.get_or_create_collection("youtube_content")
        self.tutorials = self.client.get_or_create_collection("tutorials")
```

**Task 3.2: Embedding Generation**
- Priority: High
- Dependencies: 3.1
```python
# embeddings/generator.py
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate(self, texts: list[str]):
        return self.model.encode(texts).tolist()
```

### Epic 4: Tutorial Generation

**Task 4.1: Tutorial Generator Implementation**
- Priority: Medium
- Dependencies: 3.1, 3.2
```python
# generators/tutorial.py
from typing import List
from pydantic import BaseModel

class CodeExample(BaseModel):
    code: str
    explanation: str
    language: str | None = None

class TutorialContent(BaseModel):
    summary: List[str]
    keyPoints: List[str]
    codeExamples: List[CodeExample]
    practiceExercises: List[dict]
    additionalNotes: List[str] | None = None

class TutorialGenerator:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def generate(self, content: str) -> TutorialContent:
        # Generate comprehensive tutorial using LLM
        # Return structured tutorial content
```

**Task 4.2: Tutorial Endpoints**
- Priority: Medium
- Dependencies: 4.1
```python
@router.get("/tutorial/{content_id}")
async def get_tutorial(content_id: str):
    # Retrieve tutorial from ChromaDB
    # Return structured tutorial content
```

### Epic 5: Search Functionality

**Task 5.1: Search Implementation**
- Priority: Medium
- Dependencies: 3.1, 3.2
```python
# search/semantic_search.py
class SemanticSearch:
    def __init__(self, vector_store, embedding_generator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

    async def search(self, query: str, collection: str, limit: int = 5):
        embedding = self.embedding_generator.generate([query])[0]
        results = self.vector_store.get_collection(collection).query(
            query_embeddings=[embedding],
            n_results=limit
        )
        return results
```

**Task 5.2: Search Endpoint**
- Priority: Medium
- Dependencies: 5.1
```python
@router.get("/search")
async def search_content(
    query: str,
    content_type: Literal["article", "youtube", "tutorial"],
    limit: int = 5
):
    results = await semantic_search.search(query, content_type, limit)
    return {"results": results}
```

### Epic 6: LLM Integration Layer

**Task 6.1: LLM Client Abstraction**
- Priority: High
- Dependencies: None
```python
# llm/base.py
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

# llm/openai.py
class OpenAIClient(LLMClient):
    async def generate(self, prompt: str) -> str:
        # Implement OpenAI generation

# llm/anthropic.py
class AnthropicClient(LLMClient):
    async def generate(self, prompt: str) -> str:
        # Implement Claude generation
```

## Getting Started

1. Clone the repository and create a virtual environment
2. Install dependencies from requirements.txt
3. Copy .env.example to .env and fill in required values
4. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

## API Documentation

Once running, access the OpenAPI documentation at:
- http://localhost:8000/docs
- http://localhost:8000/redoc

## Testing

Recommended testing approach:
1. Unit tests for each component
2. Integration tests for API endpoints
3. End-to-end tests for complete workflows

Create tests following this structure:
```
tests/
├── unit/
│   ├── test_processors.py
│   ├── test_generators.py
│   └── test_search.py
├── integration/
│   └── test_api.py
└── e2e/
    └── test_workflows.py
```

## Next Steps

1. Start with Epic 1 to set up the project structure
2. Implement content processing (Epic 2)
3. Add vector storage integration (Epic 3)
4. Add tutorial generation (Epic 4)
5. Implement search functionality (Epic 5)
6. Add LLM integration layer (Epic 6)

Monitor performance and adjust embeddings or search parameters as needed.

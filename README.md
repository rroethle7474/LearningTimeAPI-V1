# LearningTimeAPI

A powerful Python FastAPI-based application for content management, document processing, semantic search, tutorial generation, and prompt management. This API is to be used with the UI/Front end project found here: https://github.com/rroethle7474/MrALearningTime101

The items are stored in a vector db (ChromaDB).

The purpose of this API is to use various LLM API's to summarize and create recall documentation to help an individual learn and return knowledge they have gained over time (via article links, youtube videos, and other documents)

Also, using RAG from the ChromaDB, a user can enhance their prompt by injecting new information in this.

## Overview

LearningTimeAPI is a comprehensive backend service built with FastAPI that enables users to process, store, and retrieve various types of content with advanced semantic search capabilities. It integrates with ChromaDB for vector storage and supports multiple LLM providers for generating content and tutorials.

## Features

- **Content Management**: Upload, process, and manage various types of content
- **Document Processing**: Handle and extract information from documents
- **Semantic Search**: Perform advanced semantic searches across stored content
- **Tutorial Generation**: Generate tutorials based on processed content
- **Prompt Management**: Create and manage prompts for content generation
- **Vector Storage**: Efficient storage and retrieval using ChromaDB
- **Multiple LLM Support**: Integrates with OpenAI and Anthropic APIs

## Technologies

- **Python 3.x**
- **FastAPI**: High-performance web framework
- **ChromaDB**: Vector database for semantic search
- **Sentence Transformers**: For embedding generation
- **OpenAI/Anthropic**: LLM providers for content generation
- **Pydantic**: Data validation
- **CORS Middleware**: For cross-origin resource sharing
- **Python-multipart**: For handling file uploads
- **Uvicorn**: ASGI server for serving the API

## Technical Architecture

### Embedding Generation

The system uses Sentence Transformers to convert text content into vector embeddings:

- Default model: `all-MiniLM-L6-v2` for embedding generation
- Content is chunked before embedding to optimize search relevance
- Embeddings are stored alongside content in ChromaDB

### Vector Storage with ChromaDB

ChromaDB is used as the vector database for storing and retrieving embeddings:

- Collections are organized by content type (articles, youtube, documents)
- Each item in a collection includes:
  - Document text
  - Vector embedding
  - Metadata (title, author, source_url, etc.)
- Similarity search is performed using cosine distance

### Web Content Extraction with Playwright

The API uses Playwright for headless browser automation to extract content from web pages:

#### How Playwright Works in the System

1. **Browser Initialization**: When the ContentProcessor is initialized, it starts a headless Chromium browser.
2. **URL Processing**:
   - For each article URL, a new page is created in the browser
   - The system navigates to the URL and waits for the page to load completely
   - JavaScript executes in the context of the page to extract relevant content

3. **Content Extraction Strategy**:
   - Extracts metadata (title, author) using common selectors like meta tags
   - Removes unwanted elements (ads, navigation, footers) from the DOM
   - Identifies and extracts main content using common article container selectors
   - Falls back to body text if no specific content container is found

4. **Post-Processing**:
   - Removes excessive whitespace and normalizes text
   - Splits content into coherent chunks for embedding
   - Generates article summaries using LLM

#### Example Content Extraction Logic

```python
# Extract main content
content = await page.evaluate("""
    () => {
        // Remove unwanted elements
        const removeSelectors = [
            'header', 'footer', 'nav', 
            '.ads', '#ads', '.advertisement',
            '.social-share', '.comments', 
            'iframe', '.sidebar', '.related-articles'
        ];
        
        // Clean the page
        removeSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => el.remove());
        });
        
        // Try to find main content using common selectors
        const contentSelectors = [
            'article',
            '[role="main"]',
            '.post-content',
            '.article-content',
            '.entry-content',
            'main',
            '.main-content'
        ];
        
        // Find and return the content
        let content = null;
        for (const selector of contentSelectors) {
            const element = document.querySelector(selector);
            if (element) {
                content = element;
                break;
            }
        }
        
        return (content || document.body).innerText;
    }
""")
```

#### Resource Management

- The Playwright browser is managed as an async context manager
- Browser instances are properly closed to avoid memory leaks
- System uses browser arguments for optimal performance: `--disable-gpu`, `--no-sandbox`, `--disable-dev-shm-usage`

### Document Processing

The `DocumentProcessor` handles various document types:

- Supports PDF, DOCX, DOC, and TXT files
- Documents are processed in these steps:
  1. Text extraction based on file type
  2. Text chunking with RecursiveCharacterTextSplitter
  3. Embedding generation for each chunk
  4. Storage in ChromaDB with metadata

### Content Processing

Content from articles and YouTube videos is processed through:

- **Articles**: Playwright is used for web scraping to extract content
- **YouTube Videos**: YouTube API and transcript API to extract metadata and captions
- Content is chunked, summarized, and embedded before storage

### Semantic Search Implementation

The `SemanticSearch` class provides powerful search capabilities:

- **Single Collection Search**: Search within one collection (articles, documents, etc.)
- **Multi-Collection Search**: Search across multiple collections at once
- **Similarity Threshold**: Filter results based on minimum similarity score
- **Search Results Processing**: Results include metadata, content, and normalized similarity scores

### Context Generation for LLM Prompts

The `ContextGenerationService` enhances prompts with relevant context:

- User queries are used to search for related content in the vector database
- Retrieved content is filtered based on similarity threshold
- A system prompt formats the context for the LLM
- The LLM synthesizes a comprehensive response from the retrieved context

### Dependency Injection Pattern

The API uses FastAPI's dependency injection for service management:

- Singletons for database connections and embedding generators
- Services are initialized in main.py and accessed via dependencies
- This approach ensures efficient resource usage and simplified testing

## Prerequisites

- Python 3.8+
- ChromaDB account (for vector storage)
- OpenAI API key or Anthropic API key
- YouTube API key (for YouTube content)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LearningTimeAPI-V1.git
   cd LearningTimeAPI-V1
   ```

2. Create and activate a virtual environment: (It is recommended to place the virtual environment outside of the solution folder used by the IDE. If placed in the same folder, indexing could occur which may cause performance issues with the IDE.)
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Playwright browsers:
   ```bash
   playwright install
   ```

5. Create a `.env` file based on `.env.template`:
   ```bash
   cp .env.template .env
   ```

6. Edit the `.env` file with your API keys and configuration.

## Configuration

Edit the `.env` file to configure:

- **API Keys**:
  - `YOUTUBE_API_KEY`: For accessing YouTube content
  - `OPENAI_API_KEY`: For using OpenAI's models (optional)
  - `ANTHROPIC_API_KEY`: For using Anthropic's models (optional)
- **CORS**: `CORS_ORIGINS`: Allowed origins for CORS
- **Database**: `CHROMADB_PATH`: Path to ChromaDB storage
- **Environment**: Set to `development`, `test`, or `production`

## Running the API

Start the API using Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

## API Documentation

Once the API is running, you can access the auto-generated API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Content Management
- `POST /api/content/upload`: Upload content
- `GET /api/content/{content_id}`: Get content by ID
- `GET /api/content/`: Get all content items
- `DELETE /api/content/{content_id}`: Delete content

### Document Management
- `POST /api/document/upload`: Upload documents (PDF, DOCX, DOC, or TXT)
  - Parameters: title, tags, file
- `GET /api/document/{document_id}`: Get document by ID
- `GET /api/document/`: Get all documents
- `DELETE /api/document/{document_id}`: Delete document
- `GET /api/document/status/{task_id}`: Check document processing status
- `GET /api/document/search`: Search within documents with query parameters

### Search
- `GET /api/search/single`: Search within a single collection
  - Parameters: query, collection, limit, min_similarity
- `GET /api/search/multi`: Search across multiple collections
  - Parameters: query, collections (comma-separated), limit_per_collection, min_similarity
- `GET /api/search/similar/{content_id}`: Find similar content to a specific item
  - Parameters: content_id, collection, limit
- `GET /api/search/collection/{collection_name}/contents`: Get all contents of a collection
  - Parameters: collection_name, offset, limit
- `GET /api/search/collections/contents`: Get contents from multiple collections
  - Parameters: collections (list), offset, limit
- `GET /api/search/content/{collection_name}/by-url`: Find content by URL
  - Parameters: collection_name, source_url
- `GET /api/search/content/{collection_name}/{content_id}`: Get detailed content
  - Parameters: collection_name, content_id
- `DELETE /api/search/content/{collection_name}/{content_id}`: Delete content
  - Parameters: collection_name, content_id
- `GET /api/search/documents/collections/contents`: Get documents from multiple collections
  - Parameters: collections (list), offset, limit
- `GET /api/search/document/{collection_name}/{document_id}`: Get detailed document
  - Parameters: collection_name, document_id
- `DELETE /api/search/document/{collection_name}/{document_id}`: Delete document
  - Parameters: collection_name, document_id

### Tutorial Management
- `POST /api/tutorial/generate`: Generate a tutorial
- `GET /api/tutorial/{tutorial_id}`: Get tutorial by ID
- `GET /api/tutorial/`: Get all tutorials

### Prompt Management
- `POST /api/prompt/generate`: Generate context based on user query
  - Parameters: query, min_similarity (threshold for including results)

## Response Models

### Search Responses

```typescript
// Single collection search response
{
  query: string;
  results: Array<{
    id: string;
    content: string;
    metadata: Object;
    distance?: number;  // Similarity score
  }>;
}

// Multi-collection search response
{
  query: string;
  collections: {
    [collectionName: string]: Array<{
      id: string;
      content: string;
      metadata: Object;
      distance?: number;
    }>;
  };
}

// Document response
{
  document_id: string;
  content: string[];  // Document content broken into chunks
  metadata: {
    title: string;
    tags: string[];
    file_type: string;
    file_size: number;
    upload_date: string;  // ISO format date
    source_file: string;
  }
}
```

## Key Workflows

### Adding Content

1. Content is uploaded via the API (article URL, YouTube URL, or document file)
2. The appropriate processor extracts content based on type:
   - Articles: HTML is scraped and main content extracted
   - YouTube: Video metadata and transcript are retrieved
   - Documents: Text is extracted based on file format
3. Content is chunked into smaller segments
4. Each chunk is converted to vector embeddings
5. Chunks and metadata are stored in ChromaDB

### Searching Content

1. Search query is converted to a vector embedding
2. ChromaDB performs similarity search using the query embedding
3. Results are filtered based on similarity threshold
4. Metadata and content are retrieved for matching items
5. Results are formatted and returned to the client

### Context Generation for Prompts

1. User provides a query
2. System searches for relevant content across collections
3. Matching content is filtered by similarity threshold
4. LLM is prompted to create a synthesis of the relevant content
5. Generated context is returned for use in enhancing LLM prompts

## External Dependencies

This API relies on several external services:

1. **ChromaDB**: For vector storage and semantic search capabilities
   - Sign up at [ChromaDB](https://www.trychroma.com/)
   - Follow their documentation to set up your instance

2. **OpenAI API** or **Anthropic API**: At least one is required for LLM capabilities
   - For OpenAI: Sign up at [OpenAI Platform](https://platform.openai.com/)
   - For Anthropic: Sign up at [Anthropic](https://www.anthropic.com/)
   - Create an API key and add it to your `.env` file

3. **YouTube API**: Required for processing YouTube content
   - Create a project in [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the YouTube Data API v3
   - Create an API key and add it to your `.env` file

## Additional Requirements

### System Dependencies

- **Playwright**: For web content extraction (required for article processing)
  - After installing Python dependencies, run: `playwright install`

### Storage Requirements

- Ensure sufficient disk space for ChromaDB vector storage
- Consider backing up the ChromaDB directory periodically

### Memory Considerations

- The embedding models require approximately 500MB of RAM
- Processing large documents may require additional memory

## Development

### Project Structure
```
LearningTimeAPI-V1/
├── app_types/         # Type definitions
├── chromadb/          # ChromaDB related files
├── db/                # Database operations
├── embeddings/        # Embedding generation
├── generators/        # Content generators
├── llm/               # LLM integrations
├── processors/        # Content processors
├── routes/            # API routes
├── search/            # Search functionality
├── services/          # Business logic
├── tests/             # Test cases
├── uploads/           # Upload directory
├── utils/             # Utility functions
├── .env               # Environment variables
├── .env.template      # Template for environment variables
├── config.py          # Configuration
├── dependencies.py    # Dependency injection
├── main.py            # Application entry point
└── requirements.txt   # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request


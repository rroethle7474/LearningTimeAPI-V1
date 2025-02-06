# Document API Documentation

## Document Endpoints

### Upload Document
Upload and process a document file (PDF, DOCX, DOC, or TXT).

**Endpoint:** `POST /api/document/upload`  
**Content-Type:** `multipart/form-data`

**Request Parameters:**
```typescript
{
  title: string;          // Document title
  tags: string;          // Comma-separated tags (e.g., "report,finance,2024")
  file: File;            // Document file (PDF, DOCX, DOC, or TXT)
}
```

**Response:**
```typescript
{
  task_id: string;       // Processing task ID
  status: string;        // "completed" or "failed"
  document_id?: string;  // Unique document ID (if completed)
  error?: string;        // Error message (if failed)
}
```

### Get Document
Retrieve a document by ID.

**Endpoint:** `GET /api/document/{document_id}`

**Response:**
```typescript
{
  document_id: string;
  content: string;
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

### Delete Document
Delete a document by ID.

**Endpoint:** `DELETE /api/document/{document_id}`

**Response:**
```typescript
{
  status: string;        // "success"
  message: string;       // "Document deleted"
}
```

### Search Documents
Search through documents.

**Endpoint:** `GET /api/document/search`

**Query Parameters:**
```typescript
{
  query: string;         // Search query
  limit?: number;        // Maximum results (default: 5)
  tags?: string;         // Filter by tags (comma-separated)
}
```

**Response:**
```typescript
{
  document_id: string;
  content: string;
  metadata: {
    title: string;
    tags: string[];
    file_type: string;
    file_size: number;
    upload_date: string;
    source_file: string;
  }
  similarity: number;    // Relevance score (0-1)
}[]
```

## Multi-Collection Search

### Search Across All Collections
Search across articles, YouTube content, and documents.

**Endpoint:** `GET /api/search/multi`

**Query Parameters:**
```typescript
{
  query: string;                // Search query
  collections: string;          // Comma-separated collection names
                               // e.g., "articles_content,youtube_content,documents"
  limit_per_collection?: number;// Results per collection (default: 3)
  min_similarity?: number;      // Minimum similarity threshold (0-1, default: 0.01)
}
```

**Response:**
```typescript
{
  query: string;
  results: Array<{
    id: string;
    content: string;
    metadata: {
      title: string;
      content_type: string;     // "article", "youtube", or "document"
      tags?: string[];          // For documents
      file_type?: string;       // For documents
      author?: string;          // For articles/videos
      source_url?: string;      // For articles/videos
      duration?: string;        // For videos
      // ... other type-specific metadata
    };
    distance?: number;          // Similarity score (0-1)
  }>;
}
```

## Example Usage

### Upload Document
```typescript
const formData = new FormData();
formData.append('title', 'Annual Report 2024');
formData.append('tags', 'finance,report,annual');
formData.append('file', file);

const response = await api.post('/api/document/upload', formData);
```

### Search All Content
```typescript
const searchQuery = 'machine learning';
const collections = ['articles_content', 'youtube_content', 'documents'];
const params = {
  query: searchQuery,
  collections: collections.join(','),
  limit_per_collection: '3'
};

const searchResults = await api.get('search/multi', params);
```

## Notes

1. All dates are returned in ISO format
2. Search results are sorted by relevance (similarity score)
3. Document search includes tag-based relevance boosting
4. Supported file types: .pdf, .docx, .doc, .txt
5. Maximum file size limit should be set in your server configuration 
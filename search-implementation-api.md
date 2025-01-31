## Current Search Types
From the semantic search implementation, here are the key types and responses:

## Collections
You have 4 collections:
articles_content - Stores article content and embeddings
youtube_content - Stores video transcripts and embeddings
tutorials - Stores generated tutorials
tutorial_sections - Stores individual tutorial sections


## Single Collection Search Response:
    async def search(
        self,
        query: str,
        collection: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform semantic search across specified collection
        """
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate(query)
        
        # Search in vector store
        results = self.vector_store.search_collection(
            collection_name=collection,
            query_embeddings=query_embedding,
            n_results=limit,
            where=filters
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"])):
            formatted_results.append({
                "id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
                "distance": results["distances"][i] if "distances" in results else None
            })
        
        return {
            "query": query,
            "results": formatted_results
        }

The response structure is
interface SearchResult {
  id: string;
  content: string;
  metadata: {
    content_id: string;
    title: string;
    author: string;
    source_url: string;
    content_type: "article" | "youtube";
    duration?: string;
    published_date?: string;
    view_count?: number;
    summary?: string;
    processed_date: string;
  };
  distance?: number;
}

interface SearchResponse {
  query: string;
  results: SearchResult[];
}

## Multi-Collection Search Response
    async def multi_collection_search(
        self,
        query: str,
        collections: List[str],
        limit_per_collection: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across multiple collections
        """
        results = {}
        
        for collection in collections:
            collection_results = await self.search(
                query=query,
                collection=collection,
                limit=limit_per_collection,
                filters=filters
            )
            results[collection] = collection_results["results"]
        
        return {
            "query": query,
            "collections": results
        }

The response structure is:
interface MultiCollectionSearchResponse {
  query: string;
  collections: {
    [collectionName: string]: SearchResult[];
  };
}

## API Endpoints
The search endpoints are defined in:
@router.get("/single", response_model=SearchResponse)
async def search_single_collection(
    query: str,
    collection: str,
    limit: int = 5,
    semantic_search: SemanticSearch = Depends(get_semantic_search)
):
    """Search within a single collection"""
    try:
        results = await semantic_search.search(
            query=query,
            collection=collection,
            limit=limit
        )
        return SearchResponse(query=query, results=results["results"])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/multi", response_model=MultiCollectionSearchResponse)
async def search_multiple_collections(
    query: str,
    collections: List[str] = Query(...),
    limit_per_collection: int = 3,
    semantic_search: SemanticSearch = Depends(get_semantic_search)
):
    """Search across multiple collections"""
    try:
        results = await semantic_search.search_multi(
            query=query,
            collections=collections,
            limit_per_collection=limit_per_collection
        )
        return MultiCollectionSearchResponse(
            query=query,
            collections=results["collections"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

Usage Examples:
GET /api/search/single?query=react hooks&collection=article&limit=10
GET /api/search/multi?query=react hooks&collections=article,youtube&limit_per_collection=5

## Frontend DTO Recommendation

interface SearchResultDTO {
  id: string;
  content: string;
  metadata: {
    title: string;
    source_url: string;
    content_type: "article" | "youtube";
    summary?: string;
    author: string;
  };
  relevanceScore: number;  // This is the 'distance' field normalized
}

interface SearchResponseDTO {
  query: string;
  results: SearchResultDTO[];
}

## Explore Content Section

## Collection Content Endpoints
These endpoints allow you to retrieve paginated contents from collections without performing a search.

### Types
typescript
interface ContentListItem {
id: string;
title: string;
type: string;
source: string;
metadata: {
content_id: string;
title: string;
author: string;
source_url: string;
content_type: "article" | "youtube";
duration?: string;
published_date?: string;
view_count?: number;
summary?: string;
processed_date: string;
};
}
interface ContentListResponse {
total: number; // Total number of items in collection
items: ContentListItem[];
}
interface MultiCollectionContentResponse {
collections: {
[collectionName: string]: ContentListResponse;
};
}

### Single Collection Contents
typescript
GET /api/search/collection/{collection_name}/contents
Parameters:
collection_name: string (path parameter) - Name of collection to fetch (e.g. "articles_content")
offset: number (query parameter, default: 0) - Number of records to skip
limit: number (query parameter, default: 50) - Maximum number of records to return
Response: ContentListResponse
Example:
GET /api/search/collection/articles_content/contents?offset=0&limit=50
{
"total": 125,
"items": [
{
"id": "abc123_0",
"title": "Introduction to React Hooks",
"type": "article",
"source": "https://example.com/react-hooks",
"metadata": {
"content_id": "abc123",
"title": "Introduction to React Hooks",
"author": "Jane Doe",
"source_url": "https://example.com/react-hooks",
"content_type": "article",
"published_date": "2024-03-15",
"summary": "A comprehensive guide to React Hooks...",
"processed_date": "2024-03-16T10:30:00Z"
}
},
// ... more items
]
}

### Multiple Collections Contents
typescript
GET /api/search/collections/contents
Parameters:
collections: string (query parameter) - Comma-separated list of collection names
offset: number (query parameter, default: 0) - Number of records to skip
limit: number (query parameter, default: 50) - Maximum number of records per collection
Response: MultiCollectionContentResponse
Example:
GET /api/search/collections/contents?collections=articles_content,youtube_content&offset=0&limit=50
{
"collections": {
"articles_content": {
"total": 125,
"items": [
{
"id": "abc123_0",
"title": "Introduction to React Hooks",
"type": "article",
"source": "https://example.com/react-hooks",
"metadata": {
// ... metadata fields as above
}
}
// ... more items
]
},
"youtube_content": {
"total": 75,
"items": [
{
"id": "xyz789_0",
"title": "React Tutorial for Beginners",
"type": "youtube",
"source": "https://youtube.com/watch?v=abc123",
"metadata": {
"content_id": "xyz789",
"title": "React Tutorial for Beginners",
"author": "Code Channel",
"source_url": "https://youtube.com/watch?v=abc123",
"content_type": "youtube",
"duration": "1:23:45",
"view_count": 50000,
"published_date": "2024-02-01",
"processed_date": "2024-03-16T10:30:00Z"
}
}
// ... more items
]
}
}
}

### Implementation Notes

1. **Pagination**: Both endpoints support pagination through `offset` and `limit` parameters. The `total` field in the response can be used to implement pagination controls or infinite scrolling.

2. **Collection Names**: Valid collection names are:
   - `articles_content` - For processed articles
   - `youtube_content` - For processed video content
   - `tutorials` - For generated tutorials
   - `tutorial_sections` - For individual tutorial sections

3. **Error Handling**: The endpoints will return:
   - `404` if a collection is not found
   - `500` for server errors with an error message in the response

4. **Performance Considerations**:
   - Keep `limit` reasonable (50 is default, recommend not exceeding 100)
   - For multiple collections, the same offset/limit applies to each collection
   - Consider implementing virtual scrolling for large datasets

5. **Frontend Implementation Tips**:
   - Implement lazy loading or infinite scroll using the offset parameter
   - Show loading states while fetching data
   - Handle empty states when no items are returned

6. **Metadata Fields**: Not all metadata fields will be present for all items. Always handle optional fields appropriately in the frontend.

7. **Type vs Content Type**: The `type` field in `ContentListItem` is an alias for `content_type` from the metadata, provided at the top level for convenience.


### Content Detail Endpoint

GET /api/search/content/{collection_name}/{content_id}

Retrieves detailed content information in a normalized format regardless of content type.

Parameters:
- collection_name: string (path parameter) - Name of collection (e.g. "articles_content")
- content_id: string (path parameter) - ID of the content to retrieve

Response Type:
typescript
interface ContentDetailResponse {
id: string;
title: string;
content_type: string;
author: string;
source_url: string;
summary?: string;
published_date?: string;
processed_date: string;
tutorial_id?: string; // ID of associated tutorial if one exists
content_chunks: string[]; // Content broken into chunks
metadata: { // Additional type-specific metadata
[key: string]: any;
};
}

Example Response:

json
{
"id": "abc123",
"title": "Understanding React Hooks",
"content_type": "article",
"author": "Jane Doe",
"source_url": "https://example.com/react-hooks",
"summary": "A comprehensive guide to React Hooks...",
"published_date": "2024-03-15",
"processed_date": "2024-03-16T10:30:00Z",
"tutorial_id": "tut_xyz789",
"content_chunks": [
"React Hooks were introduced in React 16.8...",
"The useState hook is the most basic...",
// ... more chunks
],
"metadata": {
"content_type": "article",
"view_count": 1000,
"reading_time": "5 min",
// ... other type-specific metadata
}
}

Key Features:
1. Returns a consistent response structure regardless of content type
2. Includes both the content summary and full content (in chunks)
4. Maintains all original metadata for type-specific UI enhancements
5. Single endpoint for all content types

Error Responses:
- 404: Content not found
- 500: Server error with error message




## Tutorial Endpoints

These endpoints handle retrieving tutorials and their sections.

### Types
typescript
type TutorialSectionType =
| "introduction"
| "concept_explanation"
| "code_example"
| "practice_exercise"
| "summary";
interface TutorialSection {
section_id: string;
tutorial_id: string;
title: string;
content: string;
section_type: TutorialSectionType;
order: number;
metadata: Record<string, any>;
}
interface TutorialResponse {
id: string;
title: string;
description: string;
source_content_id?: string; // ID of original content if generated from article/video
source_type?: string; // "article" or "youtube"
source_url?: string; // Original content URL
generated_date: string; // ISO date string
sections: TutorialSection[];
metadata: Record<string, any>;
}
interface TutorialListItem {
id: string;
title: string;
description: string;
generated_date: string;
source_type?: string;
section_count: number;
metadata: Record<string, any>;
}
interface TutorialListResponse {
total: number;
items: TutorialListItem[];
}

### List Tutorials

GET /api/tutorials

Retrieves a paginated list of tutorials.

Parameters:
- offset: number (query parameter, default: 0) - Number of records to skip
- limit: number (query parameter, default: 50) - Maximum number of records to return

Response: TutorialListResponse

Example:
json
{
"total": 25,
"items": [
{
"id": "tut_123",
"title": "Understanding React Hooks",
"description": "A comprehensive tutorial on React Hooks",
"generated_date": "2024-03-15T10:30:00Z",
"source_type": "article",
"section_count": 5,
"metadata": {
"difficulty_level": "intermediate",
"estimated_time": "30 minutes"
}
}
// ... more tutorials
]
}

### Get Tutorial Detail

GET /api/tutorials/{tutorial_id}

Retrieves a complete tutorial with all its sections.

Parameters:
- tutorial_id: string (path parameter) - ID of the tutorial to retrieve

Response: TutorialResponse

Example:

json
{
"id": "tut_123",
"title": "Understanding React Hooks",
"description": "A comprehensive tutorial on React Hooks",
"source_content_id": "art_456",
"source_type": "article",
"source_url": "https://example.com/react-hooks",
"generated_date": "2024-03-15T10:30:00Z",
"sections": [
{
"section_id": "sec_1",
"tutorial_id": "tut_123",
"title": "Introduction to Hooks",
"content": "React Hooks were introduced in...",
"section_type": "introduction",
"order": 1,
"metadata": {
"reading_time": "2 minutes"
}
},
// ... more sections
],
"metadata": {
"difficulty_level": "intermediate",
"estimated_time": "30 minutes",
"prerequisites": ["Basic React knowledge"]
}
}

### Implementation Notes

1. **Pagination**: The list endpoint supports pagination through `offset` and `limit` parameters.

2. **Section Ordering**: Sections are returned in order based on their `order` field.

3. **Error Handling**:
   - 404: Tutorial not found
   - 500: Server error with error message

4. **Frontend Implementation Tips**:
   - Cache tutorial details when possible
   - Consider progressive loading for long tutorials
   - Handle optional fields (source content may not always be present)
   - Use section types to render appropriate UI components


### Delete Content Endpoint

DELETE /api/search/content/{collection_name}/{content_id}

Deletes content from a specified collection. This operation is permanent and cannot be undone.

Parameters:
- collection_name: string (path parameter) - Name of collection (e.g. "articles_content")
- content_id: string (path parameter) - ID of the content to delete

Response:
- Status: 204 No Content on successful deletion
- No response body

Example:
DELETE /api/search/content/articles_content/abc123

Error Responses:
- 404: Content not found
- 500: Server error with error message

Implementation Notes:
1. This is a permanent deletion - consider implementing soft delete if needed
2. All associated chunks and metadata will be removed
3. Consider implementing authorization before allowing deletion
4. Frontend should confirm with user before calling delete
5. Frontend should handle both success (204) and error cases appropriat

Example Frontend Implementation:

typescript
async function deleteContent(collectionName: string, contentId: string): Promise<boolean> {
try {
const response = await fetch(
/api/search/content/${collectionName}/${contentId},
{
method: 'DELETE',
}
);
if (response.status === 204) {
return true; // Deletion successful
}
if (response.status === 404) {
console.error('Content not found');
return false;
}
const error = await response.json();
console.error('Error deleting content:', error);
return false;
} catch (error) {
console.error('Error deleting content:', error);
return false;
}
}
// Usage example:
const deleteSuccessful = await deleteContent('articles_content', 'abc123');
if (deleteSuccessful) {
// Update UI to remove deleted item
// Show success message
} else {
// Show error message
// Keep item in UI
}
This implementation:
1. Provides a simple DELETE endpoint
2. Verifies content exists before attempting deletion
3. Returns appropriate status codes
4. Includes documentation for frontend implementation
5. Follows REST conventions for delete operations


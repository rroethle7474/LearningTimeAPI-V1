# Context Generation UI Implementation Guide

This guide explains how to implement the frontend UI for the context generation feature, which allows users to get relevant context from the knowledge base for their prompts.

## API Endpoint Details

**Endpoint:** `POST /api/prompt/generate`

**Request Format:**
typescript
interface ContextRequest {
query: string; // The user's query/prompt
}

**Response Format:**
typescript
interface ContextResponse {
context: string; // The generated context
error?: string; // Optional error message if something goes wrong
}

## Response Format

The context returned from the API will be formatted as follows:
### ARTICLES_CONTENT SOURCES:
Source: https://example.com/article1
[Content from article...]
YOUTUBE_CONTENT SOURCES:
Source: https://youtube.com/watch?v=xyz
[Content from video transcript...]
[Synthesized summary and key points...]

## Best Practices

1. **Query Guidelines:**
   - Encourage users to be specific in their queries
   - Suggest including relevant keywords
   - Recommend mentioning specific topics or concepts they want to learn about

2. **UI/UX Considerations:**
   - Show a loading state during generation
   - Make the generated context easily copyable
   - Provide clear error messages if something goes wrong
   - Consider adding a character limit to the query input
   - Add a word count or character count display

3. **Context Usage:**
   - Suggest ways to incorporate the generated context into prompts
   - Consider adding example queries
   - Provide tips for effective prompt construction

4. **Error Handling:**
   - Handle network errors gracefully
   - Provide retry functionality
   - Show user-friendly error messages
   - Consider adding a timeout for long-running requests

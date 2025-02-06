from typing import Optional, Dict, Any, List
from db.vector_store import VectorStore
from search.semantic_search import SemanticSearch
from llm.base import LLMClient
import logging

logger = logging.getLogger(__name__)

class ContextGenerationService:
    def __init__(
        self,
        llm_client: LLMClient,
        semantic_search: SemanticSearch,
    ):
        self.llm_client = llm_client
        self.semantic_search = semantic_search
        
    async def generate_context(self, query: str, min_similarity: float = 0.5) -> str:
        """
        Generate context based on query using semantic search and LLM summarization
        
        Args:
            query: The user's query string
            min_similarity: Minimum similarity threshold (0-1) for including results
        """
        logger.debug(f"Generating context for query: {query}")
        print("Similarity threshold: ", min_similarity)
        try:
            # Search across all relevant collections
            search_results = await self.semantic_search.multi_collection_search(
                query=query,
                collections=["articles_content", "youtube_content"],
                limit_per_collection=3
            )
            print("Search results retrieved")
            if not any(search_results["collections"].values()):
                logger.warning("No relevant content found in vector store")
                return "No relevant context found for the given query."

            # Format search results into context sections
            context_sections = []
            relevant_content_found = False
            
            for collection_name, results in search_results["collections"].items():
                if not results:
                    continue
                    
                result = results[0]  # Get the first (and only) result dict
                contents = result['content']
                metadatas = result['metadata']
                distances = result.get('distance', [])
                
                # Filter and format relevant content
                relevant_items = []
                for i in range(len(contents)):
                    # Calculate similarity score (1 - distance)
                    # Distance of 0 means perfect match (similarity = 1)
                    # Distance of 2 means completely different (similarity = 0)
                    similarity = 1 - (distances[i] / 2) if i < len(distances) else 0
                    print("Similarity: ", similarity)
                    if similarity >= min_similarity:
                        relevant_items.append({
                            'content': contents[i],
                            'metadata': metadatas[i],
                            'similarity': similarity
                        })
                        relevant_content_found = True
                
                if relevant_items:
                    # Sort by similarity
                    relevant_items.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    # Add collection header
                    context_sections.append(f"\n### {collection_name.upper()} SOURCES:")
                    
                    # Add each relevant item
                    for item in relevant_items:
                        source_info = f"Source: {item['metadata'].get('title', 'Unknown source')} (Relevance: {item['similarity']:.2f})"
                        context_sections.append(f"\n{source_info}\n{item['content']}\n")

            if not relevant_content_found:
                logger.warning("No content met the minimum similarity threshold")
                return "No sufficiently relevant content found for the given query."

            context = "\n".join(context_sections)

            # Create prompt for LLM to synthesize the context
            system_prompt = """
            Based on the following knowledge base excerpts, generate a comprehensive context 
            summary that addresses the user's query. Focus on:
            1. Most relevant information to the query
            2. Key technical details and concepts
            3. Practical examples or use cases
            4. Important considerations or limitations
            
            Format the response in clear sections with bullet points where appropriate.
            Include source references when citing specific information.
            
            Knowledge base excerpts:
            {context}
            
            Query:
            {query}
            
            Synthesize a clear, well-organized response that directly addresses the query 
            while incorporating relevant details from the knowledge base.
            """

            # Generate enhanced context using LLM
            response = await self.llm_client.generate(
                prompt=system_prompt.format(
                    context=context,
                    query=query
                ),
                temperature=0.7
            )

            logger.debug("Successfully generated context response")
            return response.text

        except Exception as e:
            logger.error(f"Error generating context: {str(e)}")
            raise
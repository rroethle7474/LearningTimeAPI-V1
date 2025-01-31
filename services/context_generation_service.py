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
        
    async def generate_context(self, query: str) -> str:
        """Generate context based on query using semantic search and LLM summarization"""
        logger.debug(f"Generating context for query: {query}")
        
        try:
            # Search across all relevant collections
            search_results = await self.semantic_search.multi_collection_search(
                query=query,
                collections=["articles_content", "youtube_content"],
                limit_per_collection=3
            )
            
            if not any(search_results["collections"].values()):
                logger.warning("No relevant content found in vector store")
                return "No relevant context found for the given query."

            # Format search results into context sections
            context_sections = []
            for collection_name, results in search_results["collections"].items():
                if results:  # Only add section if there are results
                    context_sections.append(f"\n### {collection_name.upper()} SOURCES:")
                    for result in results:
                        # Add source metadata if available
                        source_info = f"Source: {result.get('metadata', {}).get('source_url', 'Unknown source')}"
                        context_sections.append(f"\n{source_info}\n{result['content']}\n")

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
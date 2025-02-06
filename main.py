import asyncio
import sys
import logging

if sys.platform == "win32":
    # Set up policy for Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from routes import content, tutorial, search, prompt
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator
from generators.tutorial import TutorialGenerator
from search.semantic_search import SemanticSearch
from llm.factory import LLMFactory
from config import settings
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from services.context_generation_service import ContextGenerationService
from processors.document_processor import DocumentProcessor

# Load environment variables at startup
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
embedding_generator = EmbeddingGenerator()
vector_store = VectorStore(
    persist_directory=settings.CHROMADB_PATH,
    embedding_generator=embedding_generator
)

# Initialize document processor
document_processor = DocumentProcessor()

# Initialize LLM client (prioritize OpenAI over Anthropic)
if settings.OPENAI_API_KEY:
    llm_client = LLMFactory.create_client("openai", settings.OPENAI_API_KEY)
elif settings.ANTHROPIC_API_KEY:
    llm_client = LLMFactory.create_client("anthropic", settings.ANTHROPIC_API_KEY)
else:
    raise ValueError("No LLM API keys configured")

# Initialize other services that depend on the above
tutorial_generator = TutorialGenerator(llm_client, vector_store, embedding_generator)
semantic_search = SemanticSearch(vector_store, embedding_generator)
context_generation_service = ContextGenerationService(llm_client, semantic_search)

# Include routers
app.include_router(content.router, prefix="/api/content", tags=["content"])
app.include_router(tutorial.router, prefix="/api/tutorial")
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(prompt.router, prefix="/api/prompt", tags=["prompts"]) 
from typing import Optional, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    YOUTUBE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None  # For OpenAI LLM (optional)
    ANTHROPIC_API_KEY: Optional[str] = None  # For Claude LLM (optional)
    CHROMADB_PATH: str = "./chromadb"
    
    # Add CORS settings
    CORS_ORIGINS: str = "http://localhost:3000"
    
    # Recommended additions
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    MAX_CONTENT_LENGTH: int = 1000  # for chunking
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    ENABLE_TELEMETRY: bool = False
    
    # Add document upload path with type annotation
    DOCUMENT_UPLOAD_PATH: str = "./uploads/documents"  # You can change this path as needed
    
    class Config:
        env_file = ".env"

    # Parse comma-separated string into list
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

settings = Settings()
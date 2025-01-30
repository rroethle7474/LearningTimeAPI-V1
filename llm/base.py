from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel

class LLMResponse(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt"""
        pass
    
    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """
        Generate structured output using the LLM
        
        Args:
            prompt: Input text prompt
            response_model: Pydantic model for response structure
            **kwargs: Additional generation parameters
            
        Returns:
            Structured response as a Pydantic model
        """
        pass 
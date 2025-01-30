from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel

class LLMResponse(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[list[str]] = None
    ) -> LLMResponse:
        """
        Generate text using the LLM
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            stop_sequences: Optional list of strings to stop generation
            
        Returns:
            LLMResponse with generated text and metadata
        """
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
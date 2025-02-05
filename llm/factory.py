from typing import Optional, Literal
from .base import LLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from config import settings

class LLMFactory:
    @staticmethod
    def create_client(
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMClient:
        """
        Create an LLM client based on provider
        
        Args:
            provider: 'openai' or 'anthropic'
            api_key: API key for the provider
            model: Optional model name
            **kwargs: Additional provider-specific parameters
        """
        if provider.lower() == "openai":
            return OpenAIClient(
                api_key=api_key,
                model=model or "gpt-3.5-turbo",
                **kwargs
            )
        elif provider.lower() == "anthropic":
            return AnthropicClient(
                api_key=api_key,
                model=model or "claude-3-opus-20240229",
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

def create_llm_client(
    provider: Literal["openai", "anthropic"] = "openai"
) -> LLMClient:
    """Create an LLM client based on the specified provider"""
    if provider == "openai":
        return OpenAIClient(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-3.5-turbo-0125"
        )
    elif provider == "anthropic":
        return AnthropicClient(
            api_key=settings.ANTHROPIC_API_KEY,
            model="claude-3-sonnet-20240229"
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}") 
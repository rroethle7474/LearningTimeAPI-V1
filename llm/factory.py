from typing import Optional
from .base import LLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient

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
                model=model or "gpt-4",
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
from typing import Optional, List
from .base import LLMClient, LLMResponse
from openai import AsyncOpenAI
from pydantic import BaseModel
import json

class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        **kwargs
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> LLMResponse:
        """Generate text using OpenAI"""
        try:
            # Only add response_format for supported models
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that generates structured tutorials. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop_sequences,
            }
            
            # Add response_format only for supported models
            if self.model in ["gpt-4-turbo-preview", "gpt-4-1106-preview", "gpt-3.5-turbo-1106"]:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = await self.client.chat.completions.create(**kwargs)
            
            return LLMResponse(
                text=response.choices[0].message.content,
                metadata={
                    "model": self.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": response.usage.model_dump()
                }
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """Generate structured output using OpenAI"""
        response = await self.generate(prompt, **kwargs)
        try:
            return response_model.model_validate_json(response.text)
        except Exception as e:
            raise Exception(f"Failed to parse structured response: {str(e)}") 
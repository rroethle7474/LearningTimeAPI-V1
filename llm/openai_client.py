from typing import Optional, Dict, Any
from openai import AsyncOpenAI
from pydantic import BaseModel
import json
from .base import LLMClient, LLMResponse
import logging

logger = logging.getLogger(__name__)

class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo"
    ):
        """Initialize OpenAI client"""
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None
    ) -> LLMResponse:
        """Generate text using OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop
            )
            
            return LLMResponse(
                text=response.choices[0].message.content,
                metadata={
                    "model": self.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {str(e)}")
            raise
    
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """Generate structured output using OpenAI"""
        structured_prompt = f"""
        {prompt}
        
        Provide the response in the following JSON structure:
        {response_model.model_json_schema()}
        
        Response must be valid JSON. Only provide the JSON response, no additional text.
        """
        
        try:
            response = await self.generate(structured_prompt, **kwargs)
            json_response = json.loads(response.text)
            return response_model.model_validate(json_response)
            
        except Exception as e:
            raise Exception(f"Failed to parse structured response: {str(e)}") 
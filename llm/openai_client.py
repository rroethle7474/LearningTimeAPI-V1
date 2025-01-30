from typing import Optional, Dict, Any
import openai
from pydantic import BaseModel
import json
from .base import LLMClient, LLMResponse

class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        organization: Optional[str] = None
    ):
        """Initialize OpenAI client"""
        openai.api_key = api_key
        if organization:
            openai.organization = organization
        self.model = model
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[list[str]] = None
    ) -> LLMResponse:
        """Generate text using OpenAI"""
        try:
            response = await openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences
            )
            
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
        # Add structure requirements to prompt
        structured_prompt = f"""
        {prompt}
        
        Provide the response in the following JSON structure:
        {response_model.model_json_schema()}
        
        Response must be valid JSON.
        """
        
        # Generate response
        response = await self.generate(structured_prompt, **kwargs)
        
        try:
            # Parse JSON response
            json_response = json.loads(response.text)
            # Convert to Pydantic model
            return response_model.model_validate(json_response)
            
        except Exception as e:
            raise Exception(f"Failed to parse structured response: {str(e)}") 
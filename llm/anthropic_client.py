from typing import Optional, Dict, Any
import anthropic
from pydantic import BaseModel
import json
from .base import LLMClient, LLMResponse

class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229"
    ):
        """Initialize Anthropic client"""
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[list[str]] = None
    ) -> LLMResponse:
        """Generate text using Anthropic Claude"""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                stop_sequences=stop_sequences
            )
            
            return LLMResponse(
                text=response.content[0].text,
                metadata={
                    "model": self.model,
                    "stop_reason": response.stop_reason,
                    "usage": response.usage
                }
            )
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """Generate structured output using Claude"""
        # Add structure requirements to prompt
        structured_prompt = f"""
        {prompt}
        
        Provide the response in the following JSON structure:
        {response_model.model_json_schema()}
        
        Response must be valid JSON. Only provide the JSON response, no additional text.
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
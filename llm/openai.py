from typing import Optional
from .base import LLMClient
import openai
from openai import OpenAI

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" if you prefer
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates structured tutorials. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={ "type": "json_object" }  # Ensure JSON response
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}") 
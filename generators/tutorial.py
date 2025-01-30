from typing import List, Optional
from pydantic import BaseModel
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator
import uuid

class CodeExample(BaseModel):
    code: str
    explanation: str
    language: Optional[str] = None

class TutorialContent(BaseModel):
    title: str
    summary: List[str]
    key_points: List[str]
    code_examples: List[CodeExample]
    practice_exercises: List[dict]
    additional_notes: Optional[List[str]] = None
    source_url: str
    content_type: str

class TutorialGenerator:
    def __init__(
        self,
        llm_client,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator
    ):
        self.llm = llm_client
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    async def _generate_tutorial_content(self, content: str, metadata: dict) -> TutorialContent:
        """Generate tutorial content using LLM"""
        # Construct prompt for the LLM
        prompt = f"""
        Create a comprehensive tutorial based on the following content:
        {content}

        The tutorial should include:
        1. A clear summary of the main concepts
        2. Key learning points
        3. Code examples with explanations (if applicable)
        4. Practice exercises
        5. Additional notes or resources

        Format the response as a structured tutorial.
        """
        
        # Get LLM response
        tutorial_text = await self.llm.generate(prompt)
        
        # Parse and structure the response (implementation depends on LLM output format)
        # This is a simplified example - you'll need to adjust based on your LLM's output
        tutorial = TutorialContent(
            title=metadata.get("title", "Tutorial"),
            summary=["Main concept 1", "Main concept 2"],  # Parse from LLM response
            key_points=["Key point 1", "Key point 2"],    # Parse from LLM response
            code_examples=[
                CodeExample(
                    code="example code",
                    explanation="code explanation",
                    language="python"
                )
            ],
            practice_exercises=[
                {
                    "question": "Practice question 1",
                    "hint": "Hint 1",
                    "solution": "Solution 1"
                }
            ],
            additional_notes=["Note 1", "Note 2"],
            source_url=metadata["url"],
            content_type=metadata["type"]
        )
        
        return tutorial
    
    async def generate_tutorial(
        self,
        content_id: str,
        collection_name: str
    ) -> TutorialContent:
        """Generate a tutorial from stored content"""
        # Get content from vector store
        content_data = self.vector_store.get_by_id(collection_name, content_id)
        
        if not content_data or not content_data["documents"]:
            raise ValueError(f"Content not found for ID: {content_id}")
        
        content = content_data["documents"][0]
        metadata = content_data["metadatas"][0]
        
        # Generate tutorial
        tutorial = await self._generate_tutorial_content(content, metadata)
        
        # Store tutorial in vector store
        tutorial_dict = tutorial.model_dump()
        tutorial_id = str(uuid.uuid4())
        
        # Generate embeddings for the tutorial content
        tutorial_text = f"{tutorial.title} {' '.join(tutorial.summary)} {' '.join(tutorial.key_points)}"
        embeddings = self.embedding_generator.generate([tutorial_text])
        
        # Store in tutorials collection
        self.vector_store.add_to_collection(
            collection_name="tutorial",
            documents=[tutorial_text],
            embeddings=embeddings,
            metadatas=[{
                "title": tutorial.title,
                "source_url": tutorial.source_url,
                "content_type": tutorial.content_type,
                "original_content_id": content_id
            }],
            ids=[tutorial_id]
        )
        
        return tutorial 
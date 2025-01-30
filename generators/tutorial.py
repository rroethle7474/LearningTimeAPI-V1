from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator
from app_types.tutorial import TutorialSectionType
import uuid
import json
import re
from typing import Literal

class TutorialSection(BaseModel):
    id: str
    type: TutorialSectionType
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class TutorialMetadata(BaseModel):
    title: str
    content_id: str
    source_url: str
    content_type: Literal["article", "youtube"]
    generated_date: datetime

class ProcessedTutorial(BaseModel):
    metadata: TutorialMetadata
    sections: List[TutorialSection]

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
    
    async def _generate_tutorial_content(self, content: str, metadata: dict) -> ProcessedTutorial:
        """Generate tutorial content using LLM"""
        # Updated prompt for structured output
        prompt = f"""
        Create a comprehensive tutorial based on the following content. Format your response as a JSON object with specific sections.
        
        Content to process:
        {content}

        Required JSON structure:
        {{
            "sections": [
                {{
                    "type": "summary",
                    "title": "Overview",
                    "content": "Clear overview of main concepts"
                }},
                {{
                    "type": "key_points",
                    "title": "Key Points",
                    "content": "Important takeaways and learning points"
                }},
                {{
                    "type": "code_example",
                    "title": "Code Examples",
                    "content": "Code samples with explanations",
                    "metadata": {{
                        "language": "programming language used"
                    }}
                }},
                {{
                    "type": "practice",
                    "title": "Practice Exercises",
                    "content": "Exercise description",
                    "metadata": {{
                        "difficulty": "beginner|intermediate|advanced"
                    }}
                }},
                {{
                    "type": "notes",
                    "title": "Additional Notes",
                    "content": "Additional resources or notes"
                }}
            ]
        }}

        Guidelines:
        - Ensure each section is properly formatted and contains relevant information
        - For code examples, include both the code and explanations
        - Make practice exercises actionable and clear
        - Include relevant metadata for code and practice sections
        - Keep the content focused and well-structured
        """
        
        # Get LLM response
        tutorial_text = await self.llm.generate(prompt)
        
        # Parse the LLM response
        try:
            # Extract JSON from the response (in case LLM includes additional text)
            json_match = re.search(r'\{[\s\S]*\}', tutorial_text)
            if not json_match:
                raise ValueError("No JSON found in LLM response")
            
            tutorial_data = json.loads(json_match.group())
            
            # Create ProcessedTutorial object
            tutorial = ProcessedTutorial(
                metadata=TutorialMetadata(
                    title=metadata.get("title", "Tutorial"),
                    content_id=metadata.get("id"),
                    source_url=metadata.get("url"),
                    content_type=metadata.get("type"),
                    generated_date=datetime.utcnow()
                ),
                sections=[
                    TutorialSection(
                        id=str(uuid.uuid4()),
                        type=section["type"],
                        title=section["title"],
                        content=section["content"],
                        metadata=section.get("metadata")
                    )
                    for section in tutorial_data["sections"]
                ]
            )
            
            return tutorial
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to basic structure if parsing fails
            return ProcessedTutorial(
                metadata=TutorialMetadata(
                    title=metadata.get("title", "Tutorial"),
                    content_id=metadata.get("id"),
                    source_url=metadata.get("url"),
                    content_type=metadata.get("type"),
                    generated_date=datetime.utcnow()
                ),
                sections=[
                    TutorialSection(
                        id=str(uuid.uuid4()),
                        type="summary",
                        title="Overview",
                        content=tutorial_text[:1000]  # Use first 1000 chars as summary
                    )
                ]
            )
    
    async def generate_tutorial(
        self,
        content_id: str,
        collection_name: str
    ) -> ProcessedTutorial:
        """Generate a tutorial from stored content"""
        # Get content from vector store
        content_data = self.vector_store.get_by_id(collection_name, content_id)
        
        if not content_data or not content_data["documents"]:
            raise ValueError(f"Content not found for ID: {content_id}")
        
        content = content_data["documents"][0]
        metadata = content_data["metadatas"][0]
        
        # Generate tutorial
        tutorial = await self._generate_tutorial_content(content, metadata)
        
        # Generate embedding for the entire tutorial
        tutorial_text = f"{tutorial.metadata.title} " + " ".join(
            f"{section.title} {section.content}" for section in tutorial.sections
        )
        tutorial_embedding = self.embedding_generator.generate([tutorial_text])[0]
        
        # Store tutorial using the new schema
        tutorial_id = str(uuid.uuid4())
        self.vector_store.add_tutorial(
            tutorial_id=tutorial_id,
            tutorial_data=tutorial.dict(),
            embeddings=tutorial_embedding
        )
        
        return tutorial

    def validate_section_types(self, tutorial: ProcessedTutorial) -> bool:
        """Validate that all section types are valid"""
        valid_types = {'summary', 'key_points', 'code_example', 'practice', 'notes'}
        return all(section.type in valid_types for section in tutorial.sections)

    def validate_section_metadata(self, tutorial: ProcessedTutorial) -> bool:
        """Validate section-specific metadata"""
        for section in tutorial.sections:
            if section.type == 'code_example':
                if not section.metadata or 'language' not in section.metadata:
                    return False
            elif section.type == 'practice':
                if not section.metadata or 'difficulty' not in section.metadata:
                    return False
        return True 
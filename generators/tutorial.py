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
import logging

logger = logging.getLogger(__name__)

class TutorialSection(BaseModel):
    id: str
    type: TutorialSectionType
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}

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
    
    def create_tutorial_section(self, section: dict) -> TutorialSection:
        """Helper method to create a tutorial section with validation and logging"""
        logger.debug(f"Creating TutorialSection with type: {section['type']}")
        
        if not self.validate_section_type(section["type"]):
            logger.error(f"Invalid section type: {section['type']}")
            raise ValueError(f"Invalid section type: {section['type']}")
        
        try:
            # Ensure metadata is always a dict, never None
            metadata = section.get("metadata", {})
            if metadata is None:
                metadata = {}
            
            tutorial_section = TutorialSection(
                id=str(uuid.uuid4()),
                type=section["type"],
                title=section["title"],
                content=self.format_content(section["content"]),
                metadata=metadata  # Use the sanitized metadata
            )
            logger.debug(f"Successfully created TutorialSection with type: {tutorial_section.type}")
            return tutorial_section
        except Exception as e:
            logger.error(f"Error creating TutorialSection: {str(e)}")
            raise

    async def _generate_tutorial_content(self, content: str, metadata: dict) -> ProcessedTutorial:
        """Generate tutorial content using LLM"""
        logger.debug(f"Received metadata: {metadata}")
        # Updated prompt for OpenAI
        prompt = f"""
        Create a comprehensive tutorial based on the following content. Your response must be a valid JSON object.

        Content to process:
        {content}

        Follow this exact JSON structure:
        {{
            "sections": [
                {{
                    "type": "summary",
                    "title": "Overview",
                    "content": "<single string with paragraphs separated by newlines>"
                }},
                {{
                    "type": "key_points",
                    "title": "Key Points",
                    "content": "<single string with bullet points separated by newlines>"
                }},
                {{
                    "type": "code_example",
                    "title": "Code Examples",
                    "content": "<code samples with explanations>",
                    "metadata": {{
                        "language": "<programming language used>"
                    }}
                }},
                {{
                    "type": "practice",
                    "title": "Practice Exercises",
                    "content": "<exercises or application activities for readers>",
                    "metadata": {{
                        "difficulty": "beginner|intermediate|advanced"
                    }}
                }},
                {{
                    "type": "notes",
                    "title": "Additional Notes",
                    "content": "<any additional information>"
                }}
            ]
        }}

        Important:
        1. Ensure all JSON is properly formatted
        2. Include summary, key_points, practice, and notes sections
        3. Only include code_example section if the content is programming-related
        4. If the content is not programming-related, omit the code_example section entirely
        5. Make content clear and educational
        6. Practice exercises should be relevant to the content type (could be coding exercises, comprehension questions, or practical applications)
        7. All content fields must be strings (use newlines for formatting)
        """
        
        try:
            # Get LLM response - extract the text from LLMResponse
            response = await self.llm.generate(prompt)
            tutorial_text = response.text
            tutorial_data = json.loads(tutorial_text)
            print("IS THIS THE RESPONSE", tutorial_data)
            
            # Move format_content to be a method of the class
            tutorial = ProcessedTutorial(
                metadata=TutorialMetadata(
                    title=metadata.get("title", "Tutorial"),
                    content_id=metadata.get("content_id"),
                    source_url=metadata.get("source_url"),
                    content_type=metadata.get("content_type"),
                    generated_date=datetime.utcnow()
                ),
                sections=[
                    self.create_tutorial_section(section)
                    for section in tutorial_data["sections"]
                ]
            )
            
            return tutorial
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to basic structure if parsing fails
            return ProcessedTutorial(
                metadata=TutorialMetadata(
                    title=metadata.get("title", "Tutorial"),
                    content_id=metadata.get("content_id"),
                    source_url=metadata.get("source_url"),
                    content_type=metadata.get("content_type"),
                    generated_date=datetime.utcnow()
                ),
                sections=[
                    TutorialSection(
                        id=str(uuid.uuid4()),
                        type="summary",
                        title="Overview",
                        content=tutorial_text[:1000],
                        metadata={}  # Use empty dict instead of None
                    )
                ]
            )

    def format_content(self, content_data: Any) -> str:
        """Convert content to proper string format"""
        if isinstance(content_data, list):
            # Convert list to bullet-point string
            return "\n".join(f"• {item}" for item in content_data)
        elif isinstance(content_data, str):
            return content_data
        else:
            # Convert any other type to string
            return str(content_data)

    async def generate_tutorial(
        self,
        content_id: str,
        content_type: str
    ) -> ProcessedTutorial:
        """Generate a tutorial from stored content"""
        logger.debug(f"Generating tutorial for content_id: {content_id}, type: {content_type}")
        
        content_data = self.vector_store.get_content_by_id(content_id, content_type)
        logger.debug(f"Retrieved content data with metadata: {content_data['metadata']}")
        
        if not content_data:
            raise ValueError(f"Content not found for ID: {content_id}")
        
        # Combine chunks into full content
        full_content = " ".join(content_data["documents"])
        
        # Add content_id to metadata
        metadata = content_data["metadata"]
        metadata["content_id"] = content_id
        logger.debug(f"Updated metadata with content_id: {metadata}")
        
        # Generate tutorial with updated metadata
        tutorial = await self._generate_tutorial_content(full_content, metadata)
        
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
                # Only validate code_example metadata if the section exists
                if not section.metadata or 'language' not in section.metadata:
                    return False
            elif section.type == 'practice':
                if not section.metadata or 'difficulty' not in section.metadata:
                    return False
        return True

    def validate_section_type(self, section_type: str) -> bool:
        """Validate that a section type is valid"""
        valid_types = {'summary', 'key_points', 'code_example', 'practice', 'notes'}
        return section_type in valid_types 
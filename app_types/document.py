from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DocumentMetadata(BaseModel):
    title: str
    tags: List[str]
    file_type: str
    file_size: int
    upload_date: datetime
    source_file: str

class DocumentSubmission(BaseModel):
    title: str
    tags: str  # Comma-separated string
    file: bytes  # File content

class DocumentStatus(BaseModel):
    task_id: str
    status: str
    document_id: Optional[str] = None
    error: Optional[str] = None
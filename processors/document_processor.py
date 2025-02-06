import io
from typing import Dict, Callable, Coroutine, Any, List, Tuple
import PyPDF2
import docx
import docx2txt
import subprocess
import tempfile
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self):
        self.extractors: Dict[str, Callable[[bytes], Coroutine[Any, Any, str]]] = {
            '.txt': self._process_text,
            '.pdf': self._process_pdf,
            '.doc': self._process_doc,
            '.docx': self._process_docx
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    async def process_document(self, file_content: bytes, filename: str) -> Tuple[str, List[str]]:
        """
        Process a document file and extract its text content.
        
        Returns:
            Tuple[str, List[str]]: (full_text, chunks)
        """
        ext = self._get_extension(filename.lower())
        if ext not in self.extractors:
            raise ValueError(f"Unsupported file type: {ext}")
            
        try:
            full_text = await self.extractors[ext](file_content)
            chunks = self.text_splitter.split_text(full_text)
            return full_text, chunks
        except Exception as e:
            raise Exception(f"Failed to process {ext} file: {str(e)}")

    def _get_extension(self, filename: str) -> str:
        """Extract file extension including the dot"""
        if '.' not in filename:
            raise ValueError("Filename has no extension")
        return filename[filename.rindex('.'):]

    async def _process_text(self, content: bytes) -> str:
        """Process .txt files"""
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback to a more lenient encoding if UTF-8 fails
            return content.decode('latin-1')

    async def _process_pdf(self, content: bytes) -> str:
        """Process .pdf files"""
        text = []
        print("PROCESSING PDF")
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            print("TEXT", text)
            return "\n".join(text)
        finally:
            pdf_file.close()

    async def _process_doc(self, content: bytes) -> str:
        """Process .doc files"""
        try:
            return docx2txt.process(io.BytesIO(content))
        except Exception as e:
            raise Exception(f"Failed to process DOC file: {str(e)}")

    async def _process_docx(self, content: bytes) -> str:
        """Process .docx files"""
        try:
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Failed to process DOCX file: {str(e)}") 
import io
from typing import Dict, Callable, Coroutine, Any
import PyPDF2
import docx
import docx2txt
import subprocess
import tempfile
import os

class DocumentProcessor:
    def __init__(self):
        self.extractors: Dict[str, Callable[[bytes], Coroutine[Any, Any, str]]] = {
            '.txt': self._process_text,
            '.pdf': self._process_pdf,
            '.doc': self._process_doc,
            '.docx': self._process_docx
        }
    
    async def process_document(self, file_content: bytes, filename: str) -> str:
        """
        Process a document file and extract its text content.
        
        Args:
            file_content (bytes): Raw file content
            filename (str): Original filename with extension
            
        Returns:
            str: Extracted text content
            
        Raises:
            ValueError: If file type is unsupported
            Exception: If processing fails
        """
        ext = self._get_extension(filename.lower())
        if ext not in self.extractors:
            raise ValueError(f"Unsupported file type: {ext}")
            
        try:
            return await self.extractors[ext](file_content)
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
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            
            return "\n".join(text)
        finally:
            pdf_file.close()

    async def _process_doc(self, content: bytes) -> str:
        """
        Process .doc files using antiword
        Falls back to docx2txt if antiword fails
        """
        try:
            # Create a temporary file to store the .doc content
            with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name

            try:
                # Try antiword first (more reliable for .doc files)
                result = subprocess.run(
                    ['antiword', temp_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                extracted_text = result.stdout.strip()
                
                # If antiword extracted content successfully, return it
                if extracted_text:
                    return extracted_text
                    
            except (subprocess.SubprocessError, FileNotFoundError):
                # If antiword fails or isn't installed, fall back to docx2txt
                logger.warning("Antiword failed, falling back to docx2txt")
                return await self._fallback_doc_process(content)
                
        except Exception as e:
            logger.error(f"Error processing .doc file: {str(e)}")
            # Try fallback method
            return await self._fallback_doc_process(content)
            
        finally:
            # Clean up temporary file
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    async def _fallback_doc_process(self, content: bytes) -> str:
        """Fallback method using docx2txt"""
        try:
            return docx2txt.process(io.BytesIO(content))
        except Exception as e:
            raise Exception(f"All .doc processing methods failed. Last error: {str(e)}")

    async def _process_docx(self, content: bytes) -> str:
        """Process .docx files"""
        try:
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Failed to process DOCX file: {str(e)}") 
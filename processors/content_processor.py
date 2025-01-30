from playwright.async_api import async_playwright
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, HttpUrl
import asyncio
import re
from urllib.parse import urlparse, parse_qs
import logging

# At the top of the file, after imports
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ContentMetadata(BaseModel):
    """Metadata for processed content"""
    title: str
    author: str
    source_url: HttpUrl
    content_type: Literal["article", "youtube"]
    duration: Optional[str] = None
    published_date: Optional[str] = None
    view_count: Optional[int] = None

    class Config:
        json_encoders = {
            HttpUrl: str
        }

class ProcessingError(Exception):
    """Custom exception for content processing errors"""
    pass

class ContentProcessor:
    def __init__(self, youtube_api_key: str):
        """Initialize content processor with necessary clients"""
        self.playwright = None
        self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    async def __aenter__(self):
        """Async context manager entry"""
        print("Starting Playwright")
        self.playwright = await async_playwright().start()
        print("Started Playwright")
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-gpu',
                '--no-sandbox',
                '--disable-dev-shm-usage'
            ]
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    def extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL"""
        # Handle different YouTube URL formats
        parsed_url = urlparse(url)
        if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
            return parsed_url.path[1:]
        if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            if parsed_url.path.startswith(('/embed/', '/v/')):
                return parsed_url.path.split('/')[2]
        raise ValueError("Invalid YouTube URL")

    async def process_article(self, url: str) -> Tuple[ContentMetadata, List[str]]:
        """Process article content using Playwright"""
        page = await self.browser.new_page()
        try:
            logger.debug(f"Attempting to process article: {url}")
            await page.goto(url)
            
            # Extract metadata
            title = await page.title()
            logger.debug(f"Extracted title: {title}")
            
            # Try to extract author using common patterns
            author = await page.evaluate("""
                () => {
                    const selectors = [
                        'meta[name="author"]',
                        'meta[property="article:author"]',
                        '.author',
                        '.article-author',
                        '[rel="author"]'
                    ];
                    
                    for (const selector of selectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            if (element.tagName.toLowerCase() === 'meta') {
                                return element.content;
                            }
                            return element.textContent.trim();
                        }
                    }
                    
                    return 'Unknown Author';
                }
            """)
            
            logger.debug(f"Extracted author: {author}")
            
            # Extract main content
            content = await page.evaluate("""
                () => {
                    // Remove unwanted elements
                    const removeSelectors = [
                        'header', 'footer', 'nav', 
                        '.ads', '#ads', '.advertisement',
                        '.social-share', '.comments', 
                        'iframe', '.sidebar', '.related-articles'
                    ];
                    
                    removeSelectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(el => el.remove());
                    });
                    
                    // Try to find main content
                    const contentSelectors = [
                        'article',
                        '[role="main"]',
                        '.post-content',
                        '.article-content',
                        '.entry-content',
                        'main',
                        '.main-content'
                    ];
                    
                    let content = null;
                    for (const selector of contentSelectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            content = element;
                            break;
                        }
                    }
                    
                    // Fallback to body if no content found
                    return (content || document.body).innerText;
                }
            """)
            
            logger.debug("Creating ContentMetadata object")
            try:
                metadata = ContentMetadata(
                    title=title,
                    author=author,
                    source_url=url,
                    content_type="article"
                )
                logger.debug("Successfully created ContentMetadata object")
            except Exception as e:
                logger.error(f"Error creating ContentMetadata: {str(e)}")
                raise
            
            # Process content into chunks
            logger.debug("Processing content into chunks")
            chunks = self._chunk_text(content)
            logger.debug(f"Created {len(chunks)} chunks")
            
            return metadata, chunks
            
        except Exception as e:
            logger.error(f"Error in process_article: {str(e)}", exc_info=True)
            raise ProcessingError(f"Error processing article: {str(e)}")
        finally:
            await page.close()

    async def process_youtube(self, url: str) -> Tuple[ContentMetadata, List[str]]:
        """Process YouTube video content"""
        video_id = self.extract_video_id(url)
        
        try:
            # Get video metadata using YouTube API
            video_response = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            ).execute()

            if not video_response['items']:
                raise ProcessingError("Video not found")

            video_data = video_response['items'][0]
            
            # Create metadata object
            metadata = ContentMetadata(
                title=video_data['snippet']['title'],
                author=video_data['snippet']['channelTitle'],
                source_url=str(url),
                content_type="youtube",
                duration=video_data['contentDetails']['duration'],
                published_date=video_data['snippet']['publishedAt'],
                view_count=int(video_data['statistics'].get('viewCount', 0))
            )

            # Try to get transcript
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # Try to get English transcript first
                try:
                    transcript = transcript_list.find_transcript(['en'])
                except NoTranscriptFound:
                    # If no English, get first available and translate
                    transcript = transcript_list.find_transcript(['en-US', 'en-GB'])
                    if not transcript:
                        transcript = transcript_list.find_transcript()
                        transcript = transcript.translate('en')
                
                transcript_data = transcript.fetch()
                
                # Process transcript into chunks with timestamps
                chunks = []
                current_chunk = []
                current_chunk_text = ""
                
                for entry in transcript_data:
                    text = entry['text'].strip()
                    timestamp = entry['start']
                    
                    # Format timestamp as MM:SS
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    timestamp_str = f"{minutes:02d}:{seconds:02d}"
                    
                    # Add timestamp to text
                    formatted_text = f"[{timestamp_str}] {text}"
                    
                    if len(current_chunk_text) + len(text) > 1000:  # Chunk size limit
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_chunk_text = ""
                    
                    current_chunk.append(formatted_text)
                    current_chunk_text += " " + text
                
                # Add the last chunk if it exists
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                return metadata, chunks

            except (TranscriptsDisabled, NoTranscriptFound) as e:
                raise ProcessingError(f"No transcript available: {str(e)}")
                
        except HttpError as e:
            raise ProcessingError(f"YouTube API error: {str(e)}")
        except Exception as e:
            raise ProcessingError(f"Error processing YouTube video: {str(e)}")

    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks of approximately equal size"""
        # Remove extra whitespace and split into sentences
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        return self.embedding_model.encode(texts).tolist()

    async def process_content(
        self,
        url: str,
        content_type: Literal["article", "youtube"]
    ) -> Tuple[ContentMetadata, List[str]]:
        """Main method to process content based on type"""
        try:
            logger.debug(f"Processing content of type {content_type} from URL: {url}")
            if content_type == "article":
                result = await self.process_article(url)
                logger.debug("Successfully processed article")
                return result
            elif content_type == "youtube":
                result = await self.process_youtube(url)
                logger.debug("Successfully processed YouTube video")
                return result
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
        except Exception as e:
            logger.error(f"Error in process_content: {str(e)}", exc_info=True)
            raise ProcessingError(f"Error processing content: {str(e)}")
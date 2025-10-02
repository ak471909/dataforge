"""
Web content loader.

Handles loading and extracting content from web pages using HTTP requests.
"""

from pathlib import Path
from typing import Union
from urllib.parse import urlparse

from training_data_bot.core import Document, DocumentType, WebLoadError, LogContext
from training_data_bot.sources.base import BaseLoader


class WebLoader(BaseLoader):
    """
    Loader for web content.
    
    Fetches and extracts text content from web pages.
    """
    
    def __init__(self, timeout: float = 30.0, user_agent: str = "TrainingDataBot/0.1.0"):
        """
        Initialize the web loader.
        
        Args:
            timeout: Request timeout in seconds
            user_agent: User agent string for requests
        """
        super().__init__()
        self.supported_formats = [DocumentType.URL]
        self.timeout = timeout
        self.user_agent = user_agent
    
    async def load_single(
        self,
        source: Union[str, Path],
        **kwargs
    ) -> Document:
        """
        Load content from a web URL.
        
        Args:
            source: URL to load
            **kwargs: Additional parameters
            
        Returns:
            Document object with extracted content
            
        Raises:
            WebLoadError: If loading fails
        """
        url = str(source)
        
        with LogContext("load_single", component="WebLoader"):
            self.logger.info(f"Loading URL: {url}")
            
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                raise WebLoadError(
                    f"Invalid URL (must start with http:// or https://): {url}",
                    url=url
                )
            
            # Fetch content
            try:
                content = await self._fetch_url_content(url)
                title = self._extract_title(url, content)
                
                # Create document
                document = self.create_document(
                    title=title,
                    content=content,
                    source=url,
                    doc_type=DocumentType.URL,
                    extraction_method="httpx",
                )
                
                self.logger.info(
                    f"Successfully loaded URL: {url}",
                    word_count=document.word_count,
                    char_count=document.char_count
                )
                
                return document
                
            except Exception as e:
                if isinstance(e, WebLoadError):
                    raise
                raise WebLoadError(
                    f"Failed to load URL: {url}",
                    url=url,
                    cause=e
                )
    
    async def _fetch_url_content(self, url: str) -> str:
        """
        Fetch content from URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            Extracted text content
            
        Raises:
            WebLoadError: If request fails
        """
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Make request with custom headers
                headers = {
                    'User-Agent': self.user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                }
                
                response = await client.get(url, headers=headers, follow_redirects=True)
                
                # Check for HTTP errors
                if response.status_code != 200:
                    raise WebLoadError(
                        f"HTTP {response.status_code} error",
                        url=url,
                        status_code=response.status_code
                    )
                
                # Get content type
                content_type = response.headers.get('content-type', '').lower()
                
                # Extract text based on content type
                if 'text/html' in content_type:
                    return self._extract_html_text(response.text)
                else:
                    # For plain text or other types
                    return response.text
                    
        except ImportError:
            raise WebLoadError(
                "httpx package required for web loading. "
                "Install with: pip install httpx",
                url=url
            )
        except httpx.TimeoutException:
            raise WebLoadError(
                f"Request timeout after {self.timeout} seconds",
                url=url
            )
        except httpx.HTTPError as e:
            raise WebLoadError(
                f"HTTP error: {e}",
                url=url,
                cause=e
            )
    
    def _extract_html_text(self, html: str) -> str:
        """
        Extract text content from HTML.
        
        Args:
            html: HTML string
            
        Returns:
            Extracted text
        """
        try:
            from bs4 import BeautifulSoup
            
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            self.logger.warning(
                "BeautifulSoup not installed, returning raw HTML"
            )
            return html
    
    def _extract_title(self, url: str, content: str) -> str:
        """
        Extract title from URL or content.
        
        Args:
            url: Source URL
            content: Page content (may be HTML)
            
        Returns:
            Extracted or generated title
        """
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Try to find title tag
            title_tag = soup.find('title')
            if title_tag and title_tag.text.strip():
                return title_tag.text.strip()
                
        except ImportError:
            pass
        
        # Fallback: use URL path
        parsed = urlparse(url)
        path_title = parsed.netloc + parsed.path
        
        # Clean up the title
        path_title = path_title.rstrip('/')
        if not path_title:
            path_title = url
        
        return path_title
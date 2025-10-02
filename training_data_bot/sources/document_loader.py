"""
Document loader for text-based formats.

Handles TXT, MD, HTML, JSON, CSV, and DOCX files with appropriate
text extraction for each format.
"""

import asyncio
import csv
import json
from pathlib import Path
from typing import Optional, Union

from training_data_bot.core import (
    Document,
    DocumentType,
    DocumentLoadError,
    LogContext,
)
from training_data_bot.sources.base import BaseLoader


class DocumentLoader(BaseLoader):
    """
    Loader for various text-based document formats.
    
    Supports: TXT, MD, HTML, JSON, CSV, DOCX
    """
    
    def __init__(self):
        """Initialize the document loader."""
        super().__init__()
        self.supported_formats = [
            DocumentType.TXT,
            DocumentType.MD,
            DocumentType.HTML,
            DocumentType.JSON,
            DocumentType.CSV,
            DocumentType.DOCX,
        ]
    
    async def load_single(
        self,
        source: Union[str, Path],
        encoding: str = "utf-8",
        **kwargs
    ) -> Document:
        """
        Load a single text-based document.
        
        Args:
            source: Path to the document file
            encoding: Text encoding (default: utf-8)
            **kwargs: Additional parameters
            
        Returns:
            Document object with loaded content
            
        Raises:
            DocumentLoadError: If loading fails
        """
        source_path = Path(source)
        
        with LogContext("load_single", component="DocumentLoader"):
            self.logger.info(f"Loading document: {source_path.name}")
            
            # Validate source
            if not source_path.exists():
                raise DocumentLoadError(
                    f"File not found: {source}",
                    file_path=str(source)
                )
            
            # Determine document type
            doc_type = self.get_document_type(source)
            
            # Load content based on type
            try:
                if doc_type == DocumentType.TXT:
                    content = await self._load_text(source_path, encoding)
                elif doc_type == DocumentType.MD:
                    content = await self._load_markdown(source_path, encoding)
                elif doc_type == DocumentType.HTML:
                    content = await self._load_html(source_path, encoding)
                elif doc_type == DocumentType.JSON:
                    content = await self._load_json(source_path, encoding)
                elif doc_type == DocumentType.CSV:
                    content = await self._load_csv(source_path, encoding)
                elif doc_type == DocumentType.DOCX:
                    content = await self._load_docx(source_path)
                else:
                    raise DocumentLoadError(
                        f"Unsupported document type: {doc_type}",
                        file_type=doc_type.value
                    )
                
                # Create document object
                document = self.create_document(
                    title=source_path.stem,
                    content=content,
                    source=source_path,
                    doc_type=doc_type,
                    encoding=encoding,
                    file_size=source_path.stat().st_size,
                )
                
                self.logger.info(
                    f"Successfully loaded: {source_path.name}",
                    word_count=document.word_count,
                    char_count=document.char_count
                )
                
                return document
                
            except Exception as e:
                if isinstance(e, DocumentLoadError):
                    raise
                raise DocumentLoadError(
                    f"Failed to load document: {source_path.name}",
                    file_path=str(source),
                    file_type=doc_type.value,
                    cause=e
                )
    
    async def _load_text(self, path: Path, encoding: str) -> str:
        """
        Load plain text file.
        
        Args:
            path: File path
            encoding: Text encoding
            
        Returns:
            File content as string
        """
        def _read():
            return path.read_text(encoding=encoding)
        
        return await asyncio.to_thread(_read)
    
    async def _load_markdown(self, path: Path, encoding: str) -> str:
        """
        Load Markdown file.
        
        Args:
            path: File path
            encoding: Text encoding
            
        Returns:
            Markdown content as string (preserves formatting)
        """
        # For now, treat as plain text
        # Could add Markdown parsing in the future
        return await self._load_text(path, encoding)
    
    async def _load_html(self, path: Path, encoding: str) -> str:
        """
        Load HTML file and extract text content.
        
        Args:
            path: File path
            encoding: Text encoding
            
        Returns:
            Extracted text content
        """
        def _extract():
            try:
                from bs4 import BeautifulSoup
                
                with open(path, 'r', encoding=encoding) as f:
                    html_content = f.read()
                
                # Parse HTML
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(['script', 'style']):
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
                    "BeautifulSoup not installed, reading HTML as plain text"
                )
                return path.read_text(encoding=encoding)
        
        return await asyncio.to_thread(_extract)
    
    async def _load_json(self, path: Path, encoding: str) -> str:
        """
        Load JSON file and convert to text representation.
        
        Args:
            path: File path
            encoding: Text encoding
            
        Returns:
            JSON content as formatted text
        """
        def _extract():
            with open(path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            # Convert JSON to readable text format
            if isinstance(data, dict):
                lines = [f"{key}: {value}" for key, value in data.items()]
                return "\n".join(lines)
            elif isinstance(data, list):
                lines = [f"Item {i+1}: {item}" for i, item in enumerate(data)]
                return "\n".join(lines)
            else:
                return str(data)
        
        return await asyncio.to_thread(_extract)
    
    async def _load_csv(self, path: Path, encoding: str) -> str:
        """
        Load CSV file and convert to text representation.
        
        Args:
            path: File path
            encoding: Text encoding
            
        Returns:
            CSV content as formatted text
        """
        def _extract():
            lines = []
            
            with open(path, 'r', encoding=encoding, newline='') as f:
                reader = csv.reader(f)
                
                # Get headers
                headers = next(reader, None)
                if headers:
                    # Strip whitespace from headers
                    headers = [h.strip() for h in headers]
                    lines.append("Headers: " + ", ".join(headers))
                    lines.append("")
                
                # Process rows
                for row_num, row in enumerate(reader, 1):
                    if headers and len(row) == len(headers):
                        # Create key-value pairs
                        row_data = [
                            f"{header}: {value.strip()}"
                            for header, value in zip(headers, row)
                        ]
                        lines.append(f"Row {row_num}: {' | '.join(row_data)}")
                    else:
                        # Just list values if no headers or mismatch
                        lines.append(f"Row {row_num}: {', '.join(row)}")
                    
                    # Limit number of rows to prevent huge files
                    if row_num > 1000:
                        lines.append("... (truncated, too many rows)")
                        break
            
            return "\n".join(lines)
        
        return await asyncio.to_thread(_extract)
    
    async def _load_docx(self, path: Path) -> str:
        """
        Load Microsoft Word document and extract text.
        
        Args:
            path: File path
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentLoadError: If python-docx is not installed
        """
        def _extract():
            try:
                from docx import Document as DocxDocument
                
                doc = DocxDocument(path)
                
                # Extract text from all paragraphs
                text_parts = [
                    paragraph.text
                    for paragraph in doc.paragraphs
                    if paragraph.text.strip()
                ]
                
                return "\n".join(text_parts)
                
            except ImportError:
                raise DocumentLoadError(
                    "python-docx package required for DOCX files. "
                    "Install with: pip install python-docx",
                    file_path=str(path),
                    file_type="docx"
                )
        
        return await asyncio.to_thread(_extract)
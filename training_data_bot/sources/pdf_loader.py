"""
PDF document loader.

Handles PDF file loading and text extraction using PyMuPDF (fitz).
"""

import asyncio
from pathlib import Path
from typing import Union

from training_data_bot.core import Document, DocumentType, DocumentLoadError, LogContext
from training_data_bot.sources.base import BaseLoader


class PDFLoader(BaseLoader):
    """
    Loader for PDF documents.
    
    Uses PyMuPDF (fitz) for robust PDF text extraction.
    """
    
    def __init__(self):
        """Initialize the PDF loader."""
        super().__init__()
        self.supported_formats = [DocumentType.PDF]
    
    async def load_single(
        self,
        source: Union[str, Path],
        **kwargs
    ) -> Document:
        """
        Load a single PDF document.
        
        Args:
            source: Path to the PDF file
            **kwargs: Additional parameters
            
        Returns:
            Document object with extracted text
            
        Raises:
            DocumentLoadError: If loading fails
        """
        source_path = Path(source)
        
        with LogContext("load_single", component="PDFLoader"):
            self.logger.info(f"Loading PDF: {source_path.name}")
            
            # Validate source
            if not source_path.exists():
                raise DocumentLoadError(
                    f"PDF file not found: {source}",
                    file_path=str(source),
                    file_type="pdf"
                )
            
            # Extract text from PDF
            try:
                content = await self._extract_pdf_text(source_path)
                
                if not content or not content.strip():
                    self.logger.warning(
                        f"No text extracted from PDF: {source_path.name}"
                    )
                    content = "[Empty PDF or no extractable text]"
                
                # Create document
                document = self.create_document(
                    title=source_path.stem,
                    content=content,
                    source=source_path,
                    doc_type=DocumentType.PDF,
                    extraction_method="PyMuPDF",
                    file_size=source_path.stat().st_size,
                )
                
                self.logger.info(
                    f"Successfully loaded PDF: {source_path.name}",
                    word_count=document.word_count,
                    char_count=document.char_count
                )
                
                return document
                
            except Exception as e:
                if isinstance(e, DocumentLoadError):
                    raise
                raise DocumentLoadError(
                    f"Failed to load PDF: {source_path.name}",
                    file_path=str(source),
                    file_type="pdf",
                    cause=e
                )
    
    async def _extract_pdf_text(self, path: Path) -> str:
        """
        Extract text from PDF file.
        
        Args:
            path: Path to PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentLoadError: If PyMuPDF is not installed or extraction fails
        """
        def _extract():
            try:
                import fitz  # PyMuPDF
                
                text_parts = []
                
                # Open PDF
                doc = fitz.open(path)
                
                try:
                    # Extract text from each page
                    for page_num in range(doc.page_count):
                        page = doc[page_num]
                        text = page.get_text()
                        
                        if text.strip():
                            # Add page marker
                            text_parts.append(f"--- Page {page_num + 1} ---")
                            text_parts.append(text)
                    
                    return "\n\n".join(text_parts)
                    
                finally:
                    # Always close the document
                    doc.close()
                    
            except ImportError:
                raise DocumentLoadError(
                    "PyMuPDF package required for PDF files. "
                    "Install with: pip install PyMuPDF",
                    file_path=str(path),
                    file_type="pdf"
                )
            except Exception as e:
                raise DocumentLoadError(
                    f"Failed to extract text from PDF: {e}",
                    file_path=str(path),
                    file_type="pdf",
                    cause=e
                )
        
        return await asyncio.to_thread(_extract)
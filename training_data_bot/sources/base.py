"""
Base loader class for document loading. 

This module provides the abstract base class that all document loaders must inherit from, ensuring consistent interface behavior. 
"""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union
from uuid import uuid4

from training_data_bot.core import (
    Document,
    DocumentType,
    DocumentLoadError,
    get_logger,
    LogContext,
)


class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.
    
    All loaders must implement the load_single method and define
    their supported formats.
    """
    
    def __init__(self):
        """Initialize the base loader."""
        self.logger = get_logger(f"loader.{self.__class__.__name__}")
        self.supported_formats: List[DocumentType] = []
    
    @abstractmethod
    async def load_single(
        self,
        source: Union[str, Path],
        **kwargs
    ) -> Document:
        """
        Load a single document from a source.
        
        Args:
            source: Path or URL to the document
            **kwargs: Additional loader-specific parameters
            
        Returns:
            Document object with loaded content
            
        Raises:
            DocumentLoadError: If loading fails
        """
        pass
    
    async def load_multiple(
        self,
        sources: List[Union[str, Path]],
        max_workers: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        Load multiple documents in parallel.
        
        Args:
            sources: List of paths or URLs to documents
            max_workers: Maximum number of parallel loading operations
            **kwargs: Additional loader-specific parameters
            
        Returns:
            List of successfully loaded Document objects
        """
        with LogContext("load_multiple", component=self.__class__.__name__):
            self.logger.info(
                f"Loading {len(sources)} documents",
                total_sources=len(sources),
                max_workers=max_workers
            )
            
            # Create semaphore to limit concurrent operations
            semaphore = asyncio.Semaphore(max_workers)
            
            async def load_with_semaphore(source):
                """Load a single document with semaphore control."""
                async with semaphore:
                    try:
                        return await self.load_single(source, **kwargs)
                    except Exception as e:
                        self.logger.error(
                            f"Failed to load document: {source}",
                            source=str(source),
                            error=str(e)
                        )
                        return None
            
            # Create tasks for all sources
            tasks = [load_with_semaphore(source) for source in sources]
            
            # Execute all tasks and gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out failed loads
            documents = []
            failed_count = 0
            
            for result in results:
                if isinstance(result, Document):
                    documents.append(result)
                elif isinstance(result, Exception):
                    failed_count += 1
                    self.logger.warning(f"Document load failed: {result}")
                elif result is None:
                    failed_count += 1
            
            self.logger.info(
                "Batch loading complete",
                total_sources=len(sources),
                successful_loads=len(documents),
                failed_loads=failed_count
            )
            
            return documents
    
    def get_document_type(self, source: Union[str, Path]) -> DocumentType:
        """
        Determine document type from source.
        
        Args:
            source: Path or URL to the document
            
        Returns:
            DocumentType enum value
            
        Raises:
            DocumentLoadError: If type cannot be determined
        """
        # Check if it's a URL
        if isinstance(source, str) and source.startswith(('http://', 'https://')):
            return DocumentType.URL
        
        # Convert to Path and get extension
        source_path = Path(source)
        suffix = source_path.suffix.lower().lstrip('.')
        
        # Try to match to DocumentType
        try:
            return DocumentType(suffix)
        except ValueError:
            raise DocumentLoadError(
                f"Unsupported file type: {suffix}",
                file_path=str(source),
                file_type=suffix
            )
    
    def validate_source(self, source: Union[str, Path]) -> bool:
        """
        Validate that a source can be loaded.
        
        Args:
            source: Path or URL to validate
            
        Returns:
            True if source is valid and can be loaded
        """
        # Check if it's a URL
        if isinstance(source, str) and source.startswith(('http://', 'https://')):
            return DocumentType.URL in self.supported_formats
        
        # Check file existence
        source_path = Path(source)
        if not source_path.exists():
            return False
        
        # Check if file type is supported
        try:
            doc_type = self.get_document_type(source)
            return doc_type in self.supported_formats
        except DocumentLoadError:
            return False
    
    def create_document(
        self,
        title: str,
        content: str,
        source: Union[str, Path],
        doc_type: DocumentType,
        **kwargs
    ) -> Document:
        """
        Create a Document object with standard fields.
        
        Args:
            title: Document title
            content: Document text content
            source: Source path or URL
            doc_type: Type of document
            **kwargs: Additional document metadata
            
        Returns:
            Document object
        """
        # Calculate word and character counts
        word_count = len(content.split())
        char_count = len(content)
        
        # Create document
        document = Document(
            id=uuid4(),
            title=title,
            content=content,
            source=str(source),
            doc_type=doc_type,
            word_count=word_count,
            char_count=char_count,
            **kwargs
        )
        
        self.logger.debug(
            f"Created document: {title}",
            document_id=str(document.id),
            doc_type=doc_type.value,
            word_count=word_count,
            char_count=char_count
        )
        
        return document
    
    async def load_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        max_workers: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            max_workers: Maximum parallel loading operations
            **kwargs: Additional loader-specific parameters
            
        Returns:
            List of loaded Document objects
        """
        directory_path = Path(directory)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise DocumentLoadError(
                f"Directory not found or not a directory: {directory}",
                file_path=str(directory)
            )
        
        with LogContext("load_directory", component=self.__class__.__name__):
            self.logger.info(
                f"Scanning directory: {directory}",
                directory=str(directory),
                recursive=recursive
            )
            
            # Find all supported files
            sources = self._find_supported_files(directory_path, recursive)
            
            self.logger.info(
                f"Found {len(sources)} supported files",
                file_count=len(sources)
            )
            
            # Load all files
            return await self.load_multiple(sources, max_workers, **kwargs)
    
    def _find_supported_files(
        self,
        directory: Path,
        recursive: bool
    ) -> List[Path]:
        """
        Find all supported files in a directory.
        
        Args:
            directory: Directory to search
            recursive: Whether to search subdirectories
            
        Returns:
            List of file paths
        """
        files = []
        
        # Create patterns for supported formats
        patterns = [f"*.{fmt.value}" for fmt in self.supported_formats]
        
        for pattern in patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))
        
        # Sort files for consistent ordering
        return sorted(set(files))
    
    def __repr__(self) -> str:
        """String representation of the loader."""
        formats = ", ".join(fmt.value for fmt in self.supported_formats)
        return f"{self.__class__.__name__}(formats=[{formats}])"
"""
Unified document loader.

This module provides a single interface for loading any supported document
type by automatically detecting the format and routing to the appropriate
specialized loader.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union

from training_data_bot.core import (
    Document,
    DocumentType,
    DocumentLoadError,
    UnsupportedFormatError,
    get_logger,
    LogContext,
)
from training_data_bot.sources.base import BaseLoader
from training_data_bot.sources.document_loader import DocumentLoader
from training_data_bot.sources.pdf_loader import PDFLoader
from training_data_bot.sources.web_loader import WebLoader


class UnifiedLoader(BaseLoader):
    """
    Unified loader that automatically detects document types and routes
    to the appropriate specialized loader.
    
    This is the main entry point for document loading in the system.
    """
    
    def __init__(self):
        """Initialize the unified loader with all specialized loaders."""
        super().__init__()
        
        # Initialize specialized loaders
        self.document_loader = DocumentLoader()
        self.pdf_loader = PDFLoader()
        self.web_loader = WebLoader()
        
        # Build loader registry
        self._loader_registry: Dict[DocumentType, BaseLoader] = {}
        self._register_loaders()
        
        # All supported formats
        self.supported_formats = list(DocumentType)
        
        self.logger.info("UnifiedLoader initialized with all specialized loaders")
    
    def _register_loaders(self):
        """Register all specialized loaders for their supported formats."""
        # Register document loader for text-based formats
        for doc_type in self.document_loader.supported_formats:
            self._loader_registry[doc_type] = self.document_loader
        
        # Register PDF loader
        for doc_type in self.pdf_loader.supported_formats:
            self._loader_registry[doc_type] = self.pdf_loader
        
        # Register web loader
        for doc_type in self.web_loader.supported_formats:
            self._loader_registry[doc_type] = self.web_loader
        
        self.logger.debug(
            f"Registered loaders for {len(self._loader_registry)} document types"
        )
    
    def _get_loader_for_source(self, source: Union[str, Path]) -> BaseLoader:
        """
        Determine the appropriate loader for a source.
        
        Args:
            source: Path or URL to the document
            
        Returns:
            Specialized loader for the document type
            
        Raises:
            UnsupportedFormatError: If no loader supports the format
        """
        # Detect document type
        try:
            doc_type = self.get_document_type(source)
        except DocumentLoadError as e:
            raise UnsupportedFormatError(
                f"Cannot determine document type for: {source}",
                file_path=str(source),
                cause=e
            )
        
        # Get appropriate loader
        loader = self._loader_registry.get(doc_type)
        
        if loader is None:
            raise UnsupportedFormatError(
                f"No loader available for document type: {doc_type.value}",
                file_type=doc_type.value,
                file_path=str(source)
            )
        
        return loader
    
    async def load_single(
        self,
        source: Union[str, Path],
        **kwargs
    ) -> Document:
        """
        Load a single document, automatically detecting type and using
        the appropriate loader.
        
        Args:
            source: Path or URL to the document
            **kwargs: Additional parameters passed to specialized loader
            
        Returns:
            Document object with loaded content
            
        Raises:
            DocumentLoadError: If loading fails
            UnsupportedFormatError: If format is not supported
        """
        with LogContext("unified_load_single", component="UnifiedLoader"):
            self.logger.info(f"Loading document: {source}")
            
            # Get appropriate loader
            loader = self._get_loader_for_source(source)
            
            self.logger.debug(
                f"Routing to {loader.__class__.__name__} for {source}"
            )
            
            # Load using specialized loader
            try:
                document = await loader.load_single(source, **kwargs)
                
                self.logger.info(
                    f"Successfully loaded document: {document.title}",
                    document_id=str(document.id),
                    doc_type=document.doc_type.value,
                    word_count=document.word_count
                )
                
                return document
                
            except Exception as e:
                self.logger.error(
                    f"Failed to load document: {source}",
                    error=str(e)
                )
                raise
    
    async def load_multiple(
        self,
        sources: List[Union[str, Path]],
        max_workers: int = 4,
        skip_errors: bool = True,
        **kwargs
    ) -> List[Document]:
        """
        Load multiple documents with automatic type detection.
        
        Args:
            sources: List of paths or URLs to documents
            max_workers: Maximum number of parallel loading operations
            skip_errors: If True, continue loading other documents if one fails
            **kwargs: Additional parameters passed to specialized loaders
            
        Returns:
            List of successfully loaded Document objects
        """
        with LogContext("unified_load_multiple", component="UnifiedLoader"):
            self.logger.info(
                f"Loading {len(sources)} documents",
                total_sources=len(sources),
                max_workers=max_workers,
                skip_errors=skip_errors
            )
            
            # Create semaphore to limit concurrent operations
            semaphore = asyncio.Semaphore(max_workers)
            
            async def load_with_semaphore(source):
                """Load a single document with semaphore control."""
                async with semaphore:
                    try:
                        return await self.load_single(source, **kwargs)
                    except Exception as e:
                        if skip_errors:
                            self.logger.warning(
                                f"Skipping failed document: {source}",
                                source=str(source),
                                error=str(e)
                            )
                            return None
                        else:
                            raise
            
            # Create tasks for all sources
            tasks = [load_with_semaphore(source) for source in sources]
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=not skip_errors)
            
            # Filter successful loads
            documents = [doc for doc in results if isinstance(doc, Document)]
            failed_count = len(sources) - len(documents)
            
            self.logger.info(
                "Batch loading complete",
                total_sources=len(sources),
                successful_loads=len(documents),
                failed_loads=failed_count,
                success_rate=f"{(len(documents)/len(sources)*100):.1f}%"
            )
            
            return documents
    
    async def load_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        max_workers: int = 4,
        file_patterns: Optional[List[str]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            max_workers: Maximum parallel loading operations
            file_patterns: Optional list of glob patterns to filter files
            **kwargs: Additional parameters passed to specialized loaders
            
        Returns:
            List of loaded Document objects
        """
        directory_path = Path(directory)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise DocumentLoadError(
                f"Directory not found or not a directory: {directory}",
                file_path=str(directory)
            )
        
        with LogContext("unified_load_directory", component="UnifiedLoader"):
            self.logger.info(
                f"Scanning directory: {directory}",
                directory=str(directory),
                recursive=recursive
            )
            
            # Find all supported files
            sources = self._find_all_supported_files(
                directory_path,
                recursive,
                file_patterns
            )
            
            self.logger.info(
                f"Found {len(sources)} supported files",
                file_count=len(sources)
            )
            
            # Load all files
            return await self.load_multiple(sources, max_workers, **kwargs)
    
    def _find_all_supported_files(
        self,
        directory: Path,
        recursive: bool,
        file_patterns: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Find all supported files in a directory.
        
        Args:
            directory: Directory to search
            recursive: Whether to search subdirectories
            file_patterns: Optional list of glob patterns
            
        Returns:
            List of file paths
        """
        files = []
        
        if file_patterns:
            # Use custom patterns
            patterns = file_patterns
        else:
            # Use patterns for all supported formats
            patterns = [f"*.{fmt.value}" for fmt in self.supported_formats if fmt != DocumentType.URL]
        
        for pattern in patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))
        
        # Sort files for consistent ordering
        return sorted(set(files))
    
    async def load_from_urls(
        self,
        urls: List[str],
        max_workers: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        Load content from multiple URLs.
        
        Args:
            urls: List of URLs to load
            max_workers: Maximum parallel loading operations
            **kwargs: Additional parameters passed to web loader
            
        Returns:
            List of loaded Document objects
        """
        with LogContext("load_from_urls", component="UnifiedLoader"):
            self.logger.info(f"Loading {len(urls)} URLs")
            return await self.load_multiple(urls, max_workers, **kwargs)
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of all supported document formats.
        
        Returns:
            List of format extensions (e.g., ['txt', 'pdf', 'html'])
        """
        return [fmt.value for fmt in self.supported_formats if fmt != DocumentType.URL]
    
    def get_loader_info(self) -> Dict[str, List[str]]:
        """
        Get information about registered loaders and their formats.
        
        Returns:
            Dictionary mapping loader names to their supported formats
        """
        info = {}
        
        for loader in [self.document_loader, self.pdf_loader, self.web_loader]:
            loader_name = loader.__class__.__name__
            formats = [fmt.value for fmt in loader.supported_formats]
            info[loader_name] = formats
        
        return info
    
    async def load_mixed_batch(
        self,
        sources: List[Union[str, Path]],
        max_workers: int = 4,
        group_by_type: bool = False,
        **kwargs
    ) -> Union[List[Document], Dict[DocumentType, List[Document]]]:
        """
        Load a mixed batch of different document types.
        
        Args:
            sources: List of paths and URLs
            max_workers: Maximum parallel loading operations
            group_by_type: If True, return documents grouped by type
            **kwargs: Additional parameters passed to loaders
            
        Returns:
            Either a flat list of documents or a dictionary grouped by type
        """
        documents = await self.load_multiple(sources, max_workers, **kwargs)
        
        if not group_by_type:
            return documents
        
        # Group documents by type
        grouped: Dict[DocumentType, List[Document]] = {}
        for doc in documents:
            if doc.doc_type not in grouped:
                grouped[doc.doc_type] = []
            grouped[doc.doc_type].append(doc)
        
        self.logger.info(
            "Documents grouped by type",
            type_counts={
                doc_type.value: len(docs)
                for doc_type, docs in grouped.items()
            }
        )
        
        return grouped
    
    def __repr__(self) -> str:
        """String representation of the unified loader."""
        return f"UnifiedLoader(formats={len(self.supported_formats)}, loaders={len(set(self._loader_registry.values()))})"
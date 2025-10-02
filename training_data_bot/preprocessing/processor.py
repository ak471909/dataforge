"""
Text preprocessing and chunking for document processing. 

This module handles splitting documents into manageable chunks for AI processing, with configurable size, overlap and metadata 
chunking
"""


import re
from typing import List, Optional
from uuid import uuid4

from training_data_bot.core import (
    Document,
    TextChunk,
    ProcessingConfig,
    ProcessingError,
    get_logger,
    LogContext,
)


class TextPreprocessor:
    """
    Text preprocessor for chunking documents.
    
    Splits documents into overlapping chunks suitable for AI processing
    while preserving context and tracking metadata.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: Optional[int] = None,
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            chunk_size: Target number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.logger = get_logger("preprocessor.TextPreprocessor")
        
        # Validate parameters
        if chunk_overlap >= chunk_size:
            raise ProcessingError(
                "Chunk overlap must be less than chunk size",
                details={
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            )
        
        # Set default min_chunk_size if not provided
        if min_chunk_size is None:
            min_chunk_size = min(100, chunk_size // 10)
        
        if min_chunk_size > chunk_size:
            raise ProcessingError(
                "Minimum chunk size cannot exceed chunk size",
                details={
                    "chunk_size": chunk_size,
                    "min_chunk_size": min_chunk_size
                }
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        self.logger.info(
            "TextPreprocessor initialized",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )
    
    @classmethod
    def from_config(cls, config: ProcessingConfig) -> "TextPreprocessor":
        """
        Create preprocessor from configuration.
        
        Args:
            config: Processing configuration object
            
        Returns:
            Configured TextPreprocessor instance
        """
        return cls(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_size=getattr(config, 'min_chunk_size', 100)
        )
    
    def process_document(self, document: Document) -> List[TextChunk]:
        """
        Process a document into chunks.
        
        Args:
            document: Document to process
            
        Returns:
            List of TextChunk objects
        """
        with LogContext("process_document", component="TextPreprocessor"):
            self.logger.info(
                f"Processing document: {document.title}",
                document_id=str(document.id),
                content_length=len(document.content),
                word_count=document.word_count
            )
            
            # Clean the text
            cleaned_text = self._clean_text(document.content)
            
            # Split into chunks
            chunks = self._create_chunks(document, cleaned_text)
            
            self.logger.info(
                f"Created {len(chunks)} chunks",
                document_id=str(document.id),
                chunk_count=len(chunks),
                avg_chunk_size=sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
            )
            
            return chunks
    
    def process_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        Process multiple documents into chunks.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of all TextChunk objects from all documents
        """
        all_chunks = []
        
        for document in documents:
            chunks = self.process_document(document)
            all_chunks.extend(chunks)
        
        self.logger.info(
            f"Processed {len(documents)} documents into {len(all_chunks)} chunks"
        )
        
        return all_chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove control characters except newlines
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        return text
    
    def _create_chunks(self, document: Document, text: str) -> List[TextChunk]:
        """
        Create overlapping chunks from text.
        
        Args:
            document: Source document
            text: Cleaned text to chunk
            
        Returns:
            List of TextChunk objects
        """
        # Split text into words (simple tokenization)
        words = text.split()
        
        if not words:
            self.logger.warning(
                f"Document has no words after cleaning: {document.title}"
            )
            return []
        
        chunks = []
        chunk_index = 0
        start_word_idx = 0
        
        while start_word_idx < len(words):
            # Calculate end index for this chunk
            end_word_idx = min(start_word_idx + self.chunk_size, len(words))
            
            # Extract chunk words
            chunk_words = words[start_word_idx:end_word_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions in original text
            # This is approximate since we've normalized the text
            chars_before = len(' '.join(words[:start_word_idx]))
            start_char = chars_before + (1 if start_word_idx > 0 else 0)
            end_char = start_char + len(chunk_text)
            
            # Only keep chunks that meet minimum size
            if len(chunk_words) >= self.min_chunk_size or end_word_idx >= len(words):
                chunk = TextChunk(
                    id=uuid4(),
                    document_id=document.id,
                    content=chunk_text,
                    start_index=start_char,
                    end_index=end_char,
                    chunk_index=chunk_index,
                    token_count=len(chunk_words),
                    overlap_tokens=self.chunk_overlap if chunk_index > 0 else 0
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move to next chunk with overlap
            if end_word_idx >= len(words):
                break
            
            start_word_idx = end_word_idx - self.chunk_overlap
            
            # Prevent infinite loop
            if start_word_idx <= 0:
                start_word_idx = end_word_idx
        
        return chunks
    
    def get_chunk_context(
        self,
        chunks: List[TextChunk],
        chunk_index: int,
        context_chunks: int = 1
    ) -> str:
        """
        Get context around a specific chunk.
        
        Args:
            chunks: List of chunks from same document
            chunk_index: Index of target chunk
            context_chunks: Number of chunks before/after to include
            
        Returns:
            Combined text with context
        """
        if not chunks or chunk_index < 0 or chunk_index >= len(chunks):
            return ""
        
        start_idx = max(0, chunk_index - context_chunks)
        end_idx = min(len(chunks), chunk_index + context_chunks + 1)
        
        context_texts = [chunks[i].content for i in range(start_idx, end_idx)]
        return ' '.join(context_texts)
    
    def merge_small_chunks(
        self,
        chunks: List[TextChunk],
        merge_threshold: Optional[int] = None
    ) -> List[TextChunk]:
        """
        Merge chunks that are too small.
        
        Args:
            chunks: List of chunks to process
            merge_threshold: Size threshold for merging (default: min_chunk_size)
            
        Returns:
            List of chunks with small ones merged
        """
        if not chunks:
            return []
        
        threshold = merge_threshold or self.min_chunk_size
        merged_chunks = []
        current_merge = None
        
        for chunk in chunks:
            if chunk.token_count < threshold:
                # This chunk is too small
                if current_merge is None:
                    current_merge = chunk
                else:
                    # Merge with previous small chunk
                    current_merge = TextChunk(
                        id=uuid4(),
                        document_id=chunk.document_id,
                        content=current_merge.content + ' ' + chunk.content,
                        start_index=current_merge.start_index,
                        end_index=chunk.end_index,
                        chunk_index=current_merge.chunk_index,
                        token_count=current_merge.token_count + chunk.token_count,
                        overlap_tokens=0
                    )
            else:
                # This chunk is large enough
                if current_merge is not None:
                    # Save the merged chunk
                    merged_chunks.append(current_merge)
                    current_merge = None
                merged_chunks.append(chunk)
        
        # Don't forget the last merge if any
        if current_merge is not None:
            merged_chunks.append(current_merge)
        
        return merged_chunks
    
    def split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be enhanced with NLTK)
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def create_sentence_chunks(
        self,
        document: Document,
        sentences_per_chunk: int = 5
    ) -> List[TextChunk]:
        """
        Create chunks based on sentences rather than word count.
        
        Args:
            document: Document to process
            sentences_per_chunk: Number of sentences per chunk
            
        Returns:
            List of sentence-based chunks
        """
        cleaned_text = self._clean_text(document.content)
        sentences = self.split_by_sentences(cleaned_text)
        
        if not sentences:
            return []
        
        chunks = []
        chunk_index = 0
        
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = ' '.join(chunk_sentences)
            word_count = len(chunk_text.split())
            
            chunk = TextChunk(
                id=uuid4(),
                document_id=document.id,
                content=chunk_text,
                start_index=0,  # Approximate
                end_index=len(chunk_text),
                chunk_index=chunk_index,
                token_count=word_count,
                overlap_tokens=0
            )
            chunks.append(chunk)
            chunk_index += 1
        
        return chunks
    
    def get_statistics(self, chunks: List[TextChunk]) -> dict:
        """
        Get statistics about chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens_per_chunk": 0,
                "min_tokens": 0,
                "max_tokens": 0,
            }
        
        token_counts = [chunk.token_count for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "documents_processed": len(set(chunk.document_id for chunk in chunks))
        }
    
    def __repr__(self) -> str:
        """String representation of the preprocessor."""
        return (
            f"TextPreprocessor(chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, "
            f"min_size={self.min_chunk_size})"
        )
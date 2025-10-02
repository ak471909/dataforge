"""
Document loading sources module.

This module provides loaders for various document types including
text files, PDFs, web content, and more.
"""

from training_data_bot.sources.base import BaseLoader
from training_data_bot.sources.document_loader import DocumentLoader
from training_data_bot.sources.pdf_loader import PDFLoader
from training_data_bot.sources.web_loader import WebLoader
from training_data_bot.sources.unified import UnifiedLoader

__all__ = [
    "BaseLoader",
    "DocumentLoader",
    "PDFLoader",
    "WebLoader",
    "UnifiedLoader",
]
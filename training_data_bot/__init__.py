"""
Training Data Curation Bot

A bot that curates training data for fine-tuning large language models (LLMs) 
using AI automation and Python.
"""

__version__ = "0.1.0"
__author__ = "Abhinandan"
__email__ = "abhinandan19909@gmail.com"
__description__ = "A bot that curates training data for fine-tuning large language models (LLMs) using user-provided prompts and responses."

# Core imports for easy access
from .core.config import settings
from .core.logging import get_logger
from .core.exceptions import TrainingDataBotError

# Main bot class
from training_data_bot.bot import TrainingDataBot

# Document loading
from training_data_bot.sources import (
    PDFLoader,           # Worker for reading PDF files
    WebLoader,           # Worker for reading web pages
    DocumentLoader,      # Worker for reading documents
    UnifiedLoader,       # Manager who decides which worker to use
)

# Task generation
from training_data_bot.tasks import (
    QAGenerator,                    # Generates Q&A pairs
    ClassificationGenerator,        # Generates classifications
    SummarizationGenerator,         # Generates summaries
    TaskTemplate,                   # Task template definition
    TaskManager,                    # Task orchestration manager
)

# Support services
from .preprocessing import TextPreprocessor    # Text chunking and preprocessing
from .evaluation import QualityEvaluator       # Quality assessment
from .storage import DatasetExporter           # Dataset export to various formats

# Core models (optional - for advanced users)
from training_data_bot.core import (
    Document,
    TextChunk,
    TrainingExample,
    Dataset,
    TaskType,
    DocumentType,
    ExportFormat,
    QualityReport,
)

__all__ = [
    # Core
    "TrainingDataBot",
    "settings",
    "get_logger",
    "TrainingDataBotError",
    
    # Sources
    "PDFLoader",
    "WebLoader",
    "DocumentLoader",
    "UnifiedLoader",
    
    # Tasks
    "QAGenerator",
    "ClassificationGenerator",
    "SummarizationGenerator",
    "TaskTemplate",
    "TaskManager",
    
    # Services
    "TextPreprocessor",
    "QualityEvaluator",
    "DatasetExporter",
    
    # Models (for advanced usage)
    "Document",
    "TextChunk",
    "TrainingExample",
    "Dataset",
    "TaskType",
    "DocumentType",
    "ExportFormat",
    "QualityReport",
]
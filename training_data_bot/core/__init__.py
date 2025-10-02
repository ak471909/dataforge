"""
Core module for the Training Data Bot.

This module provides the foundational data models, exceptions, and utilities
used throughout the system.
"""

# Data models
from training_data_bot.core.models import (
    # Base classes
    BaseEntity,
    
    # Enums
    DocumentType,
    TaskType,
    QualityMetric,
    ProcessingStatus,
    ExportFormat,
    
    # Document family
    Document,
    TextChunk,
    
    # Task family
    TaskTemplate,
    TaskResult,
    
    # Training family
    TrainingExample,
    Dataset,
    
    # Quality family
    QualityReport,
    
    # Operations family
    ProcessingJob,
    
    # Configuration models
    AIProviderConfig,
    ProcessingConfig,
    
    # Type aliases
    DocumentSource,
    TaskParameters,
    QualityScores,
    MetricScores,
)

# Exceptions
from training_data_bot.core.exceptions import (
    # Base exceptions
    TrainingDataBotError,
    
    # Configuration errors
    ConfigurationError,
    InitializationError,
    
    # Document loading errors
    DocumentLoadError,
    UnsupportedFormatError,
    FileNotFoundError,
    FileCorruptedError,
    WebLoadError,
    
    # AI and task errors
    AIClientError,
    AIProviderError,
    RateLimitError,
    AuthenticationError,
    TaskProcessingError,
    TaskTemplateError,
    TaskTimeoutError,
    
    # Data processing errors
    ValidationError,
    ProcessingError,
    QualityError,
    
    # Storage errors
    StorageError,
    ExportError,
    ImportError,
    DatabaseError,
    
    # Resource errors
    ResourceError,
    MemoryError,
    TimeoutError,
    
    # Utility functions
    handle_exception,
    is_recoverable_error,
    get_retry_delay,
)

# Configuration and logging (Session 2 additions)
from training_data_bot.core.config import (
    Settings,
    LogLevel,
    Environment,
    create_settings,
    get_settings,
    update_settings,
    load_settings_from_env,
    validate_configuration,
    settings,  # Global settings instance
)

from training_data_bot.core.logging import (
    TrainingDataBotLogger,
    LogContext,
    PerformanceLogger,
    setup_logging,
    get_logger,
    get_performance_logger,
    setup_logging_from_settings,
)

# Version info
__version__ = "0.1.0"

# Public API for this module
__all__ = [
    # Models
    "BaseEntity",
    "DocumentType",
    "TaskType", 
    "QualityMetric",
    "ProcessingStatus",
    "ExportFormat",
    "Document",
    "TextChunk",
    "TaskTemplate",
    "TaskResult",
    "TrainingExample",
    "Dataset",
    "QualityReport",
    "ProcessingJob",
    "AIProviderConfig",
    "ProcessingConfig",
    "DocumentSource",
    "TaskParameters",
    "QualityScores",
    "MetricScores",
    
    # Configuration
    "Settings",
    "LogLevel",
    "Environment", 
    "create_settings",
    "get_settings",
    "update_settings",
    "load_settings_from_env",
    "validate_configuration",
    "settings",
    
    # Logging
    "TrainingDataBotLogger",
    "LogContext",
    "PerformanceLogger",
    "setup_logging",
    "get_logger",
    "get_performance_logger",
    "setup_logging_from_settings",
    
    # Exceptions
    "TrainingDataBotError",
    "ConfigurationError",
    "InitializationError",
    "DocumentLoadError",
    "UnsupportedFormatError", 
    "FileNotFoundError",
    "FileCorruptedError",
    "WebLoadError",
    "AIClientError",
    "AIProviderError",
    "RateLimitError",
    "AuthenticationError",
    "TaskProcessingError",
    "TaskTemplateError",
    "TaskTimeoutError",
    "ValidationError",
    "ProcessingError",
    "QualityError",
    "StorageError",
    "ExportError",
    "ImportError",
    "DatabaseError", 
    "ResourceError",
    "MemoryError",
    "TimeoutError",
    "handle_exception",
    "is_recoverable_error",
    "get_retry_delay",
]
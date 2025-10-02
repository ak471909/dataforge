"""
Custom exceptions for the Training Data Bot.

This module defines all custom exception types used throughout the system
for better error handling and debugging.
"""

from typing import Any, Dict, Optional


class TrainingDataBotError(Exception):
    """Base exception for all Training Data Bot errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"{self.error_code}: {self.message}"]
        
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")
        
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
            "type": self.__class__.__name__
        }


# Configuration and initialization errors
class ConfigurationError(TrainingDataBotError):
    """Raised when there are configuration issues."""
    pass


class InitializationError(TrainingDataBotError):
    """Raised when component initialization fails."""
    pass


# Document loading errors
class DocumentLoadError(TrainingDataBotError):
    """Raised when document loading fails."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = file_path
        if file_type:
            details['file_type'] = file_type
        
        super().__init__(message, details=details, **kwargs)


class UnsupportedFormatError(DocumentLoadError):
    """Raised when trying to load an unsupported file format."""
    pass


class FileNotFoundError(DocumentLoadError):
    """Raised when a file cannot be found."""
    pass


class FileCorruptedError(DocumentLoadError):
    """Raised when a file is corrupted or unreadable."""
    pass


class WebLoadError(DocumentLoadError):
    """Raised when web content loading fails."""
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if url:
            details['url'] = url
        if status_code:
            details['status_code'] = status_code
        
        super().__init__(message, details=details, **kwargs)


# AI and task processing errors
class AIClientError(TrainingDataBotError):
    """Base class for AI client errors."""
    pass


class AIProviderError(AIClientError):
    """Raised when AI provider returns an error."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if provider:
            details['provider'] = provider
        if model:
            details['model'] = model
        
        super().__init__(message, details=details, **kwargs)


class RateLimitError(AIProviderError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if retry_after:
            details['retry_after'] = retry_after
        
        super().__init__(message, details=details, **kwargs)


class AuthenticationError(AIProviderError):
    """Raised when API authentication fails."""
    pass


class TaskProcessingError(TrainingDataBotError):
    """Raised when task processing fails."""
    
    def __init__(
        self,
        message: str,
        task_type: Optional[str] = None,
        chunk_id: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if task_type:
            details['task_type'] = task_type
        if chunk_id:
            details['chunk_id'] = chunk_id
        
        super().__init__(message, details=details, **kwargs)


class TaskTemplateError(TaskProcessingError):
    """Raised when task template is invalid or malformed."""
    pass


class TaskTimeoutError(TaskProcessingError):
    """Raised when task processing times out."""
    pass


# Data processing and validation errors
class ValidationError(TrainingDataBotError):
    """Raised when data validation fails."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if field_name:
            details['field_name'] = field_name
        if invalid_value is not None:
            details['invalid_value'] = str(invalid_value)
        
        super().__init__(message, details=details, **kwargs)


class ProcessingError(TrainingDataBotError):
    """Raised when data processing fails."""
    pass


class QualityError(TrainingDataBotError):
    """Raised when quality assessment fails or quality is too low."""
    
    def __init__(
        self,
        message: str,
        quality_score: Optional[float] = None,
        threshold: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if quality_score is not None:
            details['quality_score'] = quality_score
        if threshold is not None:
            details['threshold'] = threshold
        
        super().__init__(message, details=details, **kwargs)


# Storage and export errors
class StorageError(TrainingDataBotError):
    """Base class for storage-related errors."""
    pass


class ExportError(StorageError):
    """Raised when data export fails."""
    
    def __init__(
        self,
        message: str,
        export_format: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if export_format:
            details['export_format'] = export_format
        if output_path:
            details['output_path'] = output_path
        
        super().__init__(message, details=details, **kwargs)


class ImportError(StorageError):
    """Raised when data import fails."""
    pass


class DatabaseError(StorageError):
    """Raised when database operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table_name: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        if table_name:
            details['table_name'] = table_name
        
        super().__init__(message, details=details, **kwargs)


# Resource and system errors
class ResourceError(TrainingDataBotError):
    """Raised when system resources are insufficient."""
    pass


class MemoryError(ResourceError):
    """Raised when memory limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        required_memory: Optional[int] = None,
        available_memory: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if required_memory:
            details['required_memory'] = required_memory
        if available_memory:
            details['available_memory'] = available_memory
        
        super().__init__(message, details=details, **kwargs)


class TimeoutError(ResourceError):
    """Raised when operations timeout."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        if operation:
            details['operation'] = operation
        
        super().__init__(message, details=details, **kwargs)


# Utility functions for error handling
def handle_exception(
    exception: Exception,
    context: Optional[str] = None,
    reraise_as: Optional[type] = None,
    **details
) -> TrainingDataBotError:
    """
    Handle and wrap exceptions with additional context.
    
    Args:
        exception: The original exception
        context: Additional context about where the error occurred
        reraise_as: Exception class to reraise as
        **details: Additional details to include
    
    Returns:
        TrainingDataBotError or specified exception type
    """
    message = str(exception)
    if context:
        message = f"{context}: {message}"
    
    exception_class = reraise_as or TrainingDataBotError
    
    return exception_class(
        message=message,
        cause=exception,
        details=details
    )


def is_recoverable_error(error: Exception) -> bool:
    """
    Determine if an error is recoverable (should retry).
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error is recoverable, False otherwise
    """
    recoverable_types = (
        RateLimitError,
        TimeoutError,
        WebLoadError,
    )
    
    # Check if it's a recoverable type
    if isinstance(error, recoverable_types):
        return True
    
    # Check if it's a temporary network issue
    if isinstance(error, AIProviderError):
        # 5xx status codes are typically temporary
        status_code = error.details.get('status_code')
        if status_code and 500 <= status_code < 600:
            return True
    
    return False


def get_retry_delay(error: Exception, attempt: int) -> float:
    """
    Calculate retry delay based on error type and attempt number.
    
    Args:
        error: The exception that occurred
        attempt: The current attempt number (1-based)
        
    Returns:
        Delay in seconds before retry
    """
    base_delay = 1.0
    max_delay = 60.0
    
    # Exponential backoff
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    
    # Special handling for rate limit errors
    if isinstance(error, RateLimitError):
        retry_after = error.details.get('retry_after')
        if retry_after:
            return max(delay, retry_after)
    
    return delay
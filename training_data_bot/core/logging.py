"""
Logging infrastrucutre for the Training Data Bot. 

This module provides structured logging with context management, performance tracking, and configurable output formats. 
"""


import json
import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import UUID, uuid4


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""
    
    def __init__(self):
        super().__init__()
        self.context_stack = []
    
    def filter(self, record):
        """Add context information to the log record."""
        # Add context information if available
        if self.context_stack:
            current_context = self.context_stack[-1]
            record.context_id = current_context.get('context_id', 'unknown')
            record.operation = current_context.get('operation', 'unknown')
            record.component = current_context.get('component', 'unknown')
            
            # Add any additional context data
            for key, value in current_context.items():
                if key not in ['context_id', 'operation', 'component']:
                    setattr(record, f"ctx_{key}", value)
        else:
            record.context_id = 'root'
            record.operation = 'system'
            record.component = 'core'
        
        # Add timestamp
        record.timestamp = datetime.utcnow().isoformat()
        
        return True
    
    def push_context(self, context: Dict[str, Any]):
        """Push a new context onto the stack."""
        self.context_stack.append(context)
    
    def pop_context(self):
        """Pop the current context from the stack."""
        if self.context_stack:
            return self.context_stack.pop()
        return None


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record):
        """Format the log record as structured JSON."""
        log_data = {
            'timestamp': getattr(record, 'timestamp', datetime.utcnow().isoformat()),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'context_id': getattr(record, 'context_id', 'unknown'),
            'operation': getattr(record, 'operation', 'unknown'),
            'component': getattr(record, 'component', 'unknown'),
        }
        
        # Add context data
        for attr_name in dir(record):
            if attr_name.startswith('ctx_'):
                key = attr_name[4:]  # Remove 'ctx_' prefix
                log_data[key] = getattr(record, attr_name)
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """Formatter that outputs human-readable logs with context."""
    
    def __init__(self):
        super().__init__(
            fmt='%(timestamp)s [%(levelname)s] %(component)s.%(operation)s (%(context_id)s) - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record):
        """Format the log record in a human-readable format."""
        # Ensure required attributes exist
        if not hasattr(record, 'timestamp'):
            record.timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        if not hasattr(record, 'context_id'):
            record.context_id = 'root'
        if not hasattr(record, 'operation'):
            record.operation = 'system'
        if not hasattr(record, 'component'):
            record.component = 'core'
        
        return super().format(record)


class TrainingDataBotLogger:
    """Main logger class for the Training Data Bot."""
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        structured: bool = True,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Create context filter
        self.context_filter = ContextFilter()
        self.logger.addFilter(self.context_filter)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(HumanReadableFormatter())
        self.logger.addHandler(console_handler)
        
        # Create file handler if log file specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _log_with_extra(self, level: int, message: str, **extra):
        """Log a message with extra context fields."""
        if extra:
            # Create a custom LogRecord with extra fields
            record = self.logger.makeRecord(
                self.logger.name,
                level,
                __file__,
                0,
                message,
                (),
                None
            )
            record.extra_fields = extra
            self.logger.handle(record)
        else:
            self.logger.log(level, message)
    
    def debug(self, message: str, **extra):
        """Log debug message."""
        self._log_with_extra(logging.DEBUG, message, **extra)
    
    def info(self, message: str, **extra):
        """Log info message."""
        self._log_with_extra(logging.INFO, message, **extra)
    
    def warning(self, message: str, **extra):
        """Log warning message."""
        self._log_with_extra(logging.WARNING, message, **extra)
    
    def error(self, message: str, **extra):
        """Log error message."""
        self._log_with_extra(logging.ERROR, message, **extra)
    
    def critical(self, message: str, **extra):
        """Log critical message."""
        self._log_with_extra(logging.CRITICAL, message, **extra)
    
    def exception(self, message: str, **extra):
        """Log exception with traceback."""
        self._log_with_extra(logging.ERROR, message, **extra)
        # Let the logging framework handle the exception info
        self.logger.exception("")
    
    @contextmanager
    def context(
        self,
        operation: str,
        component: Optional[str] = None,
        context_id: Optional[Union[str, UUID]] = None,
        **kwargs
    ):
        """Create a logging context for grouping related operations."""
        context_data = {
            'context_id': str(context_id or uuid4()),
            'operation': operation,
            'component': component or 'unknown',
            **kwargs
        }
        
        self.context_filter.push_context(context_data)
        start_time = time.time()
        
        try:
            self.info(f"Starting operation: {operation}")
            yield context_data
        except Exception as e:
            self.exception(f"Operation failed: {operation}", error=str(e))
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.info(
                f"Completed operation: {operation}",
                duration_seconds=duration
            )
            self.context_filter.pop_context()


class LogContext:
    """Context manager for logging operations."""
    
    def __init__(
        self,
        operation: str,
        component: Optional[str] = None,
        logger: Optional[TrainingDataBotLogger] = None,
        **kwargs
    ):
        self.operation = operation
        self.component = component
        self.logger = logger or get_logger()
        self.kwargs = kwargs
        self.context_data = None
    
    def __enter__(self):
        """Enter the logging context."""
        self.context_manager = self.logger.context(
            self.operation,
            self.component,
            **self.kwargs
        )
        self.context_data = self.context_manager.__enter__()
        return self.context_data
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the logging context."""
        return self.context_manager.__exit__(exc_type, exc_val, exc_tb)


class PerformanceLogger:
    """Logger specifically for performance metrics."""
    
    def __init__(self, logger: TrainingDataBotLogger):
        self.logger = logger
    
    def log_processing_stats(
        self,
        operation: str,
        total_items: int,
        processed_items: int,
        failed_items: int,
        duration: float,
        **extra
    ):
        """Log processing performance statistics."""
        success_rate = (processed_items / total_items * 100) if total_items > 0 else 0
        items_per_second = processed_items / duration if duration > 0 else 0
        
        self.logger.info(
            f"Processing complete: {operation}",
            total_items=total_items,
            processed_items=processed_items,
            failed_items=failed_items,
            success_rate_percent=round(success_rate, 2),
            duration_seconds=round(duration, 2),
            items_per_second=round(items_per_second, 2),
            **extra
        )
    
    def log_api_call(
        self,
        provider: str,
        model: str,
        tokens_used: int,
        response_time: float,
        success: bool,
        **extra
    ):
        """Log AI API call metrics."""
        self.logger.info(
            f"AI API call: {provider}/{model}",
            provider=provider,
            model=model,
            tokens_used=tokens_used,
            response_time_seconds=round(response_time, 3),
            success=success,
            **extra
        )
    
    def log_document_processing(
        self,
        document_id: str,
        document_type: str,
        word_count: int,
        chunks_created: int,
        processing_time: float,
        **extra
    ):
        """Log document processing metrics."""
        words_per_second = word_count / processing_time if processing_time > 0 else 0
        
        self.logger.info(
            f"Document processed: {document_type}",
            document_id=document_id,
            document_type=document_type,
            word_count=word_count,
            chunks_created=chunks_created,
            processing_time_seconds=round(processing_time, 2),
            words_per_second=round(words_per_second, 2),
            **extra
        )


# Global logger instance
_global_logger: Optional[TrainingDataBotLogger] = None


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
    **kwargs
) -> TrainingDataBotLogger:
    """
    Setup global logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        structured: Whether to use structured JSON logging
        **kwargs: Additional logging configuration
    
    Returns:
        Configured logger instance
    """
    global _global_logger
    _global_logger = TrainingDataBotLogger(
        name="training_data_bot",
        level=level,
        log_file=log_file,
        structured=structured,
        **kwargs
    )
    return _global_logger


def get_logger(name: Optional[str] = None) -> TrainingDataBotLogger:
    """
    Get a logger instance.
    
    Args:
        name: Optional logger name, uses global logger if not specified
    
    Returns:
        Logger instance
    """
    if name and name != "training_data_bot":
        # Create a new logger with the specified name
        return TrainingDataBotLogger(
            name=name,
            level="INFO",
            structured=True
        )
    
    # Return global logger or create default if not initialized
    if _global_logger is None:
        return setup_logging()
    
    return _global_logger


def get_performance_logger() -> PerformanceLogger:
    """Get a performance logger instance."""
    return PerformanceLogger(get_logger())


# Convenience function for quick logging setup from settings
def setup_logging_from_settings(settings):
    """Setup logging from settings configuration."""
    from .config import Settings
    
    if isinstance(settings, Settings):
        return setup_logging(
            level=settings.log_level.value,
            log_file=settings.log_file,
            structured=settings.enable_structured_logging,
            max_bytes=settings.log_max_bytes,
            backup_count=settings.log_backup_count
        )
    else:
        raise ValueError("Invalid settings object provided")
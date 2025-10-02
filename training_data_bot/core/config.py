"""
Configuration management for the Training Data Bot.

This module handles all configuration loading from environment variables,
config files, and provides default settings for the system.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from enum import Enum

from training_data_bot.core.models import AIProviderConfig, ProcessingConfig


class LogLevel(str, Enum):
    """Available logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application info
    app_name: str = "Training Data Bot"
    app_version: str = "0.1.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = Field(default=False)
    
    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_max_bytes: int = 10485760  # 10MB
    log_backup_count: int = 5
    enable_structured_logging: bool = True
    
    # File processing settings
    max_file_size_mb: int = Field(default=100, gt=0)
    supported_file_types: List[str] = Field(default=[
        "pdf", "txt", "md", "html", "json", "csv", "docx"
    ])
    default_encoding: str = "utf-8"
    
    # Text processing configuration
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    min_chunk_size: int = Field(default=100, gt=0)
    max_chunks_per_document: int = Field(default=1000, gt=0)
    
    # Performance settings
    max_workers: int = Field(default=4, gt=0, le=32)
    batch_size: int = Field(default=10, gt=0)
    request_timeout: float = Field(default=30.0, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, gt=0)
    enable_parallel_processing: bool = True
    
    # Quality control
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_quality_filtering: bool = True
    min_quality_examples: int = Field(default=10, ge=0)
    
    # AI Provider settings
    default_ai_provider: str = "openai"
    ai_providers: Dict[str, AIProviderConfig] = Field(default_factory=dict)
    
    # OpenAI specific settings
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 4000
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    # Anthropic specific settings  
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-sonnet-20240229"
    anthropic_max_tokens: int = 4000
    
    # Web scraping settings
    web_timeout: float = Field(default=30.0, gt=0)
    web_max_redirects: int = Field(default=5, ge=0)
    web_user_agent: str = "TrainingDataBot/0.1.0"
    enable_web_caching: bool = True
    web_cache_ttl: int = Field(default=3600, gt=0)  # 1 hour
    
    # Database settings (optional)
    database_url: Optional[str] = None
    database_echo: bool = False
    database_pool_size: int = Field(default=5, gt=0)
    database_max_overflow: int = Field(default=10, ge=0)
    
    # Storage settings
    output_directory: str = "output"
    temp_directory: str = "temp"
    enable_compression: bool = True
    default_export_format: str = "jsonl"
    
    # Security settings
    max_request_size_mb: int = Field(default=50, gt=0)
    allowed_domains: List[str] = Field(default_factory=list)
    blocked_domains: List[str] = Field(default_factory=list)
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, gt=0)
    requests_per_hour: int = Field(default=3600, gt=0)
    requests_per_day: int = Field(default=86400, gt=0)
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        """Ensure chunk overlap is less than chunk size."""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    @validator('ai_providers', pre=True)
    def parse_ai_providers(cls, v):
        """Parse AI provider configurations from environment or dict."""
        if isinstance(v, str):
            # If it's a string, try to parse as YAML/JSON
            try:
                import json
                return json.loads(v)
            except json.JSONDecodeError:
                try:
                    return yaml.safe_load(v)
                except yaml.YAMLError:
                    return {}
        return v or {}
    
    @validator('supported_file_types')
    def normalize_file_types(cls, v):
        """Normalize file types to lowercase without dots."""
        return [ext.lower().lstrip('.') for ext in v]
    
    def get_ai_provider_config(self, provider_name: str) -> Optional[AIProviderConfig]:
        """Get configuration for a specific AI provider."""
        if provider_name in self.ai_providers:
            return AIProviderConfig(**self.ai_providers[provider_name])
        
        # Create default configs for known providers
        if provider_name == "openai":
            return AIProviderConfig(
                provider_name="openai",
                api_key=self.openai_api_key,
                api_url=self.openai_api_base,
                model_name=self.openai_model,
                max_tokens=self.openai_max_tokens,
                temperature=self.openai_temperature,
                timeout=self.request_timeout,
                retry_attempts=self.retry_attempts,
            )
        elif provider_name == "anthropic":
            return AIProviderConfig(
                provider_name="anthropic",
                api_key=self.anthropic_api_key,
                model_name=self.anthropic_model,
                max_tokens=self.anthropic_max_tokens,
                temperature=0.7,
                timeout=self.request_timeout,
                retry_attempts=self.retry_attempts,
            )
        
        return None
    
    def get_processing_config(self) -> ProcessingConfig:
        """Get text processing configuration."""
        return ProcessingConfig(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            max_workers=self.max_workers,
            batch_size=self.batch_size,
            quality_threshold=self.quality_threshold,
            enable_parallel_processing=self.enable_parallel_processing,
            max_file_size_mb=self.max_file_size_mb,
        )
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [self.output_directory, self.temp_directory]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION
    
    class Config:
        """Pydantic settings configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "TDB_"  # Training Data Bot prefix
        case_sensitive = False
        extra = "allow"
        
        # Field aliases for environment variables
        fields = {
            "openai_api_key": {"env": ["TDB_OPENAI_API_KEY", "OPENAI_API_KEY"]},
            "anthropic_api_key": {"env": ["TDB_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"]},
            "database_url": {"env": ["TDB_DATABASE_URL", "DATABASE_URL"]},
            "log_level": {"env": ["TDB_LOG_LEVEL", "LOG_LEVEL"]},
            "environment": {"env": ["TDB_ENVIRONMENT", "ENVIRONMENT"]},
        }


def load_config_from_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f) or {}
        elif config_path.suffix.lower() == '.json':
            import json
            return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def create_settings(
    config_file: Optional[Union[str, Path]] = None,
    **overrides
) -> Settings:
    """
    Create settings instance with optional configuration file and overrides.
    
    Args:
        config_file: Optional path to configuration file
        **overrides: Additional configuration overrides
        
    Returns:
        Configured Settings instance
    """
    config_data = {}
    
    # Load from file if provided
    if config_file:
        config_data.update(load_config_from_file(config_file))
    
    # Apply overrides
    config_data.update(overrides)
    
    # Create settings instance
    settings = Settings(**config_data)
    
    # Create necessary directories
    settings.create_directories()
    
    return settings


# Global settings instance
settings = create_settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def update_settings(**overrides) -> Settings:
    """Update global settings with new values."""
    global settings
    current_dict = settings.dict()
    current_dict.update(overrides)
    settings = Settings(**current_dict)
    settings.create_directories()
    return settings


def load_settings_from_env() -> Settings:
    """Load settings primarily from environment variables."""
    return Settings()


def validate_configuration(config: Settings) -> List[str]:
    """
    Validate configuration and return list of issues.
    
    Args:
        config: Settings instance to validate
        
    Returns:
        List of validation error messages
    """
    issues = []
    
    # Check for required API keys based on default provider
    if config.default_ai_provider == "openai" and not config.openai_api_key:
        issues.append("OpenAI API key is required when using OpenAI as default provider")
    
    if config.default_ai_provider == "anthropic" and not config.anthropic_api_key:
        issues.append("Anthropic API key is required when using Anthropic as default provider")
    
    # Check chunk configuration
    if config.chunk_overlap >= config.chunk_size:
        issues.append("Chunk overlap must be less than chunk size")
    
    if config.min_chunk_size >= config.chunk_size:
        issues.append("Minimum chunk size must be less than chunk size")
    
    # Check quality threshold
    if not (0.0 <= config.quality_threshold <= 1.0):
        issues.append("Quality threshold must be between 0.0 and 1.0")
    
    # Check worker limits
    if config.max_workers > 32:
        issues.append("Max workers should not exceed 32 for stability")
    
    # Check file size limits
    if config.max_file_size_mb > 1000:  # 1GB
        issues.append("Max file size should not exceed 1GB")
    
    return issues
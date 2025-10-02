"""
Core data models for the Training Data Bot.

This module defines all the data strucutures used throughout the system,
including documents, tasks, training examples, and datasets.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
import pathlib
from pydantic import BaseModel, Field, field_validator

class BaseEntity(BaseModel):
    """Base class for all entities with common fields."""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None 
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration.""" 
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }

# Enums for categorization
class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    URL = "url"

class TaskType(str, Enum):
    """Types of tasks that can be performed."""
    QA_GENERATION = "qa_generation"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    NER = "named_entity_recognition"
    RED_TEAMING = "red_teaming"
    INSTRUCTION_RESPONSE = "instruction_response"

class QualityMetric(str, Enum):
    """Quality assessment metrics"""
    TOXICITY = "toxicity"
    BIAS = "bias"
    DIVERSITY = "diversity"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"


class ProcessingStatus(str, Enum):
    """Status of processing jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


# Document family - Input data
class Document(BaseEntity):
    """A source document that contains content to be processed."""
    title: str
    content: str
    source: str
    doc_type: DocumentType
    word_count: int = 0
    char_count: int = 0
    language: Optional[str] = None
    encoding: Optional[str] = None
    
    @field_validator('word_count')
    @classmethod
    def calculate_word_count(cls, v, info):
        if v == 0 and info.data.get('content'):
            return len(info.data['content'].split())
        return v

    @field_validator('char_count') 
    @classmethod
    def calculate_char_count(cls, v, info):
        if v == 0 and info.data.get('content'):
            return len(info.data['content'])
        return v


class TextChunk(BaseEntity):
    """A chunk of text from a document for processing."""
    document_id: UUID
    content: str
    start_index: int
    end_index: int
    chunk_index: int
    token_count: int = 0
    overlap_tokens: int = 0


# Task family - Work instructions
class TaskTemplate(BaseEntity):
    """Template for generating tasks."""
    name: str
    task_type: TaskType
    description: str
    prompt_template: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0"
    enabled: bool = True


class TaskResult(BaseEntity):
    """Result of executing a task on a text chunk."""
    task_id: UUID
    input_chunk_id: UUID
    output: str
    confidence: float = Field(ge=0.0, le=1.0)
    quality_scores: Dict[str, float] = Field(default_factory=dict)
    processing_time: float = 0.0
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None


# Training family - Final products
class TrainingExample(BaseEntity):
    """A single training example for machine learning."""
    input_text: str
    output_text: str
    task_type: TaskType
    source_document_id: UUID
    source_chunk_id: Optional[UUID] = None
    quality_scores: Dict[str, float] = Field(default_factory=dict)
    difficulty: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class Dataset(BaseEntity):
    """A collection of training examples."""
    name: str
    description: str
    examples: List[Any] = Field(default_factory=list)
    total_examples: int = 0
    train_split: float = Field(default=0.8, ge=0.0, le=1.0)
    validation_split: float = Field(default=0.1, ge=0.0, le=1.0)
    test_split: float = Field(default=0.1, ge=0.0, le=1.0)
    task_distribution: Dict[TaskType, int] = Field(default_factory=dict)
    version: str = "1.0"
    

# Quality family - Inspectors
class QualityReport(BaseEntity):
    """Quality assessment report for data."""
    target_id: UUID
    target_type: str  # "document", "example", "dataset"
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool = False
    metric_scores: Dict[QualityMetric, float] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    evaluation_model: Optional[str] = None


# Operations family - Factory managers
class ProcessingJob(BaseEntity):
    """A processing job that tracks work progress."""
    name: str
    job_type: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    
    def update_progress(self):
        """Update progress percentage based on processed items."""
        if self.total_items > 0:
            self.progress_percentage = (self.processed_items / self.total_items) * 100.0
        self.updated_at = datetime.utcnow()


# Configuration and settings models
class AIProviderConfig(BaseModel):
    """Configuration for AI providers."""
    provider_name: str
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    model_name: str
    max_tokens: int = 4000
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: float = 30.0
    retry_attempts: int = 3
    rate_limit_rpm: Optional[int] = None


class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    max_workers: int = Field(default=4, gt=0)
    batch_size: int = Field(default=10, gt=0)
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_parallel_processing: bool = True
    max_file_size_mb: int = Field(default=100, gt=0)


# Type aliases for complex types
DocumentSource = Union[str, "pathlib.Path"]
TaskParameters = Dict[str, Any]
QualityScores = Dict[str, float]
MetricScores = Dict[QualityMetric, float]

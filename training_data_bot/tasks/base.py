"""
Base task generator interface.

This module defines the abstract base class that all task generators must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from uuid import uuid4

from training_data_bot.core import (
    TextChunk,
    TaskTemplate,
    TaskResult,
    TrainingExample,
    TaskType,
    get_logger,
    LogContext,
)
from training_data_bot.ai import AIClient, AIResponse


class BaseTaskGenerator(ABC):
    """
    Abstract base class for task generators.
    
    All task generators (QA, Classification, Summarization, etc.) 
    must implement this interface.
    """
    
    def __init__(
        self,
        ai_client: AIClient,
        task_type: TaskType,
        **kwargs
    ):
        """
        Initialize the task generator.
        
        Args:
            ai_client: AI client for generation
            task_type: Type of task this generator handles
            **kwargs: Additional generator-specific parameters
        """
        self.ai_client = ai_client
        self.task_type = task_type
        self.logger = get_logger(f"tasks.{self.__class__.__name__}")
        self.extra_params = kwargs
        
        # Statistics tracking
        self.stats = {
            "total_generated": 0,
            "total_failed": 0,
            "total_tokens_used": 0,
            "total_time": 0.0,
        }
    
    @abstractmethod
    def get_default_template(self) -> TaskTemplate:
        """
        Get the default task template for this generator.
        
        Returns:
            TaskTemplate object with default prompt and parameters
        """
        pass
    
    @abstractmethod
    async def generate_single(
        self,
        chunk: TextChunk,
        template: Optional[TaskTemplate] = None,
        **kwargs
    ) -> TaskResult:
        """
        Generate a single task result from a text chunk.
        
        Args:
            chunk: Text chunk to process
            template: Optional custom template to use
            **kwargs: Additional generation parameters
            
        Returns:
            TaskResult object with generated output
        """
        pass
    
    async def generate_batch(
        self,
        chunks: List[TextChunk],
        template: Optional[TaskTemplate] = None,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[TaskResult]:
        """
        Generate task results for multiple chunks.
        
        Args:
            chunks: List of text chunks to process
            template: Optional custom template to use
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional generation parameters
            
        Returns:
            List of TaskResult objects
        """
        with LogContext("generate_batch", component=self.__class__.__name__):
            self.logger.info(
                f"Generating batch of {len(chunks)} tasks",
                batch_size=len(chunks),
                task_type=self.task_type.value
            )
            
            import asyncio
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def generate_with_semaphore(chunk):
                async with semaphore:
                    try:
                        return await self.generate_single(chunk, template, **kwargs)
                    except Exception as e:
                        self.logger.error(
                            f"Failed to generate task for chunk {chunk.id}: {e}"
                        )
                        self.stats["total_failed"] += 1
                        return None
            
            # Execute all tasks
            tasks = [generate_with_semaphore(chunk) for chunk in chunks]
            results = await asyncio.gather(*tasks)
            
            # Filter out None results
            valid_results = [r for r in results if r is not None]
            
            self.logger.info(
                f"Batch generation complete",
                total_chunks=len(chunks),
                successful=len(valid_results),
                failed=len(chunks) - len(valid_results)
            )
            
            return valid_results
    
    def create_training_example(
        self,
        task_result: TaskResult,
        chunk: TextChunk,
        **kwargs
    ) -> TrainingExample:
        """
        Convert a task result into a training example.
        
        Args:
            task_result: The task result to convert
            chunk: The source text chunk
            **kwargs: Additional metadata
            
        Returns:
            TrainingExample object
        """
        return TrainingExample(
            id=uuid4(),
            input_text=self._format_input(chunk),
            output_text=task_result.output,
            task_type=self.task_type,
            source_document_id=chunk.document_id,
            source_chunk_id=chunk.id,
            quality_scores=task_result.quality_scores,
            metadata={
                "confidence": task_result.confidence,
                "model_used": task_result.model_used,
                "tokens_used": task_result.tokens_used,
                **kwargs
            }
        )
    
    def _format_input(self, chunk: TextChunk) -> str:
        """
        Format the chunk content for input.
        
        Args:
            chunk: Text chunk to format
            
        Returns:
            Formatted input text
        """
        # Default implementation - can be overridden
        return chunk.content
    
    def _build_prompt(
        self,
        chunk: TextChunk,
        template: TaskTemplate,
        **kwargs
    ) -> str:
        """
        Build the prompt from template and chunk.
        
        Args:
            chunk: Text chunk to process
            template: Task template with prompt template
            **kwargs: Additional template variables
            
        Returns:
            Formatted prompt string
        """
        # Replace template variables
        prompt = template.prompt_template
        
        # Common variables
        variables = {
            "text": chunk.content,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            **kwargs
        }
        
        # Replace all variables in template
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        
        return prompt
    
    def _calculate_confidence(
        self,
        response: AIResponse,
        chunk: TextChunk
    ) -> float:
        """
        Calculate confidence score for the generated output.
        
        Args:
            response: AI response object
            chunk: Source text chunk
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence from finish reason
        confidence = 1.0 if response.finish_reason == "stop" else 0.5
        
        # Adjust based on output length
        output_length = len(response.content.split())
        if output_length < 5:
            confidence *= 0.5
        elif output_length > 500:
            confidence *= 0.8
        
        return min(max(confidence, 0.0), 1.0)
    
    def _assess_quality(
        self,
        output: str,
        chunk: TextChunk
    ) -> Dict[str, float]:
        """
        Assess the quality of generated output.
        
        Args:
            output: Generated output text
            chunk: Source text chunk
            
        Returns:
            Dictionary of quality metric scores
        """
        scores = {}
        
        # Length check
        output_words = len(output.split())
        if output_words > 0:
            scores["length_score"] = min(output_words / 100, 1.0)
        else:
            scores["length_score"] = 0.0
        
        # Non-empty check
        scores["completeness"] = 1.0 if output.strip() else 0.0
        
        # Relevance (simple heuristic - can be enhanced)
        chunk_words = set(chunk.content.lower().split())
        output_words_set = set(output.lower().split())
        overlap = len(chunk_words & output_words_set)
        scores["relevance"] = min(overlap / len(chunk_words) if chunk_words else 0, 1.0)
        
        return scores
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get generation statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "task_type": self.task_type.value,
            "total_generated": self.stats["total_generated"],
            "total_failed": self.stats["total_failed"],
            "success_rate": (
                self.stats["total_generated"] / 
                (self.stats["total_generated"] + self.stats["total_failed"])
                if (self.stats["total_generated"] + self.stats["total_failed"]) > 0
                else 0.0
            ),
            "total_tokens_used": self.stats["total_tokens_used"],
            "total_time": self.stats["total_time"],
            "avg_time_per_task": (
                self.stats["total_time"] / self.stats["total_generated"]
                if self.stats["total_generated"] > 0
                else 0.0
            )
        }
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.stats = {
            "total_generated": 0,
            "total_failed": 0,
            "total_tokens_used": 0,
            "total_time": 0.0,
        }
    
    def __repr__(self) -> str:
        """String representation of the generator."""
        return f"{self.__class__.__name__}(task_type={self.task_type.value})"
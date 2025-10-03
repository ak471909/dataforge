"""
Task manager for orchestrating task generation.

This module manages the creation and execution of tasks across
different generators, providing a unified interface.
"""

import asyncio
from typing import Dict, List, Optional, Union
from uuid import UUID

from training_data_bot.core import (
    TextChunk,
    TaskTemplate,
    TaskResult,
    TrainingExample,
    TaskType,
    ProcessingJob,
    ProcessingStatus,
    get_logger,
    LogContext,
    TaskProcessingError,
)
from training_data_bot.ai import AIClient
from training_data_bot.tasks.base import BaseTaskGenerator
from training_data_bot.tasks.generators import (
    QAGenerator,
    ClassificationGenerator,
    SummarizationGenerator,
)


class TaskManager:
    """
    Task manager for orchestrating task generation.
    
    Manages multiple task generators and coordinates the creation
    of training examples from text chunks.
    """
    
    # Registry of available generators
    GENERATOR_REGISTRY = {
        TaskType.QA_GENERATION: QAGenerator,
        TaskType.CLASSIFICATION: ClassificationGenerator,
        TaskType.SUMMARIZATION: SummarizationGenerator,
    }
    
    def __init__(
        self,
        ai_client: Optional[AIClient] = None,
        **kwargs
    ):
        """
        Initialize the task manager.
        
        Args:
            ai_client: AI client for generation (creates default if None)
            **kwargs: Additional configuration parameters
        """
        self.logger = get_logger("tasks.TaskManager")
        
        # Initialize AI client
        if ai_client is None:
            # Create default client - should be provided in production
            self.logger.warning("No AI client provided, task generation will fail")
            self.ai_client = None
        else:
            self.ai_client = ai_client
        
        # Initialize generators dictionary
        self.generators: Dict[TaskType, BaseTaskGenerator] = {}
        
        # Task templates
        self.templates: Dict[UUID, TaskTemplate] = {}
        
        # Active jobs tracking
        self.jobs: Dict[UUID, ProcessingJob] = {}
        
        # Configuration
        self.max_concurrent = kwargs.get("max_concurrent", 5)
        
        self.logger.info(
            "TaskManager initialized",
            max_concurrent=self.max_concurrent
        )
    
    def register_generator(
        self,
        task_type: TaskType,
        generator: Optional[BaseTaskGenerator] = None,
        **kwargs
    ):
        """
        Register a task generator for a specific task type.
        
        Args:
            task_type: Type of task
            generator: Pre-initialized generator (creates default if None)
            **kwargs: Parameters for generator initialization
        """
        if generator is None:
            # Create default generator for this task type
            if task_type not in self.GENERATOR_REGISTRY:
                raise TaskProcessingError(
                    f"No default generator available for task type: {task_type}",
                    task_type=task_type.value
                )
            
            generator_class = self.GENERATOR_REGISTRY[task_type]
            generator = generator_class(self.ai_client, **kwargs)
        
        self.generators[task_type] = generator
        self.logger.info(f"Registered generator for {task_type.value}")
    
    def get_generator(self, task_type: TaskType) -> BaseTaskGenerator:
        """
        Get the generator for a specific task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Task generator instance
        """
        if task_type not in self.generators:
            # Auto-register default generator
            self.register_generator(task_type)
        
        return self.generators[task_type]
    
    async def generate_task(
        self,
        chunk: TextChunk,
        task_type: TaskType,
        template: Optional[TaskTemplate] = None,
        **kwargs
    ) -> TaskResult:
        """
        Generate a single task result.
        
        Args:
            chunk: Text chunk to process
            task_type: Type of task to generate
            template: Optional custom template
            **kwargs: Additional generation parameters
            
        Returns:
            TaskResult object
        """
        with LogContext("generate_task", task_type=task_type.value):
            generator = self.get_generator(task_type)
            
            result = await generator.generate_single(
                chunk=chunk,
                template=template,
                **kwargs
            )
            
            return result
    
    async def generate_tasks(
        self,
        chunks: List[TextChunk],
        task_types: List[TaskType],
        template: Optional[TaskTemplate] = None,
        **kwargs
    ) -> List[TaskResult]:
        """
        Generate multiple task results for chunks and task types.
        
        Args:
            chunks: List of text chunks to process
            task_types: List of task types to generate
            template: Optional custom template
            **kwargs: Additional generation parameters
            
        Returns:
            List of TaskResult objects
        """
        with LogContext("generate_tasks", 
                       chunk_count=len(chunks),
                       task_type_count=len(task_types)):
            
            self.logger.info(
                f"Generating tasks",
                chunks=len(chunks),
                task_types=[t.value for t in task_types]
            )
            
            # Create all task combinations with task_type tracking
            tasks = []
            task_type_map = {}  # Map result to task_type
            for chunk in chunks:
                for task_type in task_types:
                    task_coro = self.generate_task(
                        chunk=chunk,
                        task_type=task_type,
                        template=template,
                        **kwargs
                    )
                    tasks.append((task_coro, task_type))  # Store task_type with task

            
            # Execute with limited concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def generate_with_semaphore(task_tuple):
                task_coro, task_type = task_tuple
                async with semaphore:
                    try:
                        result = await task_coro
                        return (result, task_type)  # Return both
                    except Exception as e:
                        self.logger.error(f"Task generation failed: {e}")
                        return None

            results = await asyncio.gather(
                *[generate_with_semaphore(t) for t in tasks]
            )

            # Filter out None results
            valid_results = []
            for item in results:
                if item is not None and item[0] is not None:
                    valid_results.append(item)


            
            self.logger.info(
                f"Task generation complete",
                total_tasks=len(tasks),
                successful=len(valid_results),
                failed=len(tasks) - len(valid_results)
            )
            
            return valid_results
    
    async def create_training_examples(
        self,
        chunks: List[TextChunk],
        task_types: List[TaskType],
        **kwargs
    ) -> List[TrainingExample]:
        """
        Create training examples from chunks.
        
        Args:
            chunks: List of text chunks
            task_types: List of task types to generate
            **kwargs: Additional parameters
            
        Returns:
            List of TrainingExample objects
        """
        with LogContext("create_training_examples",
                       chunk_count=len(chunks)):
            
            # Generate all task results
            task_results = await self.generate_tasks(
                chunks=chunks,
                task_types=task_types,
                **kwargs
            )
            
            # Convert to training examples
            examples = []
            chunk_map = {chunk.id: chunk for chunk in chunks}
            
            for result, task_type in task_results:  # ← Now we have task_type
                chunk = chunk_map.get(result.input_chunk_id)
                if chunk:
                    generator = self.get_generator(task_type)  # ← Use the actual task_type
                    example = generator.create_training_example(
                        task_result=result,
                        chunk=chunk
                    )
                    examples.append(example)
            
            self.logger.info(
                f"Created {len(examples)} training examples"
            )
            
            return examples
    
    def _get_task_type_from_result(self, result: TaskResult) -> TaskType:
        """
        Determine task type from result.
        
        Args:
            result: Task result
            
        Returns:
            TaskType enum value
        """
        # Check which generator produced this result
        for task_type, generator in self.generators.items():
            if generator.task_type == task_type:
                # Simple heuristic - could be improved
                return task_type
        
        # Default fallback
        return TaskType.QA_GENERATION
    
    async def process_job(
        self,
        job: ProcessingJob,
        chunks: List[TextChunk],
        task_types: List[TaskType],
        **kwargs
    ) -> List[TrainingExample]:
        """
        Process a job with progress tracking.
        
        Args:
            job: Processing job to track
            chunks: Text chunks to process
            task_types: Task types to generate
            **kwargs: Additional parameters
            
        Returns:
            List of TrainingExample objects
        """
        with LogContext("process_job", job_id=str(job.id)):
            try:
                # Update job status
                job.status = ProcessingStatus.RUNNING
                job.total_items = len(chunks) * len(task_types)
                job.processed_items = 0
                
                import datetime
                job.started_at = datetime.datetime.utcnow()
                
                # Store job
                self.jobs[job.id] = job
                
                # Process in batches for progress tracking
                batch_size = max(1, len(chunks) // 10)  # 10% batches
                examples = []
                
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    
                    batch_examples = await self.create_training_examples(
                        chunks=batch_chunks,
                        task_types=task_types,
                        **kwargs
                    )
                    
                    examples.extend(batch_examples)
                    
                    # Update progress
                    job.processed_items += len(batch_chunks) * len(task_types)
                    job.update_progress()
                
                # Job complete
                job.status = ProcessingStatus.COMPLETED
                job.completed_at = datetime.datetime.utcnow()
                
                self.logger.info(
                    f"Job {job.id} completed",
                    examples_created=len(examples)
                )
                
                return examples
                
            except Exception as e:
                job.status = ProcessingStatus.FAILED
                job.error_message = str(e)
                self.logger.error(f"Job {job.id} failed: {e}")
                raise
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics from all generators.
        
        Returns:
            Dictionary with statistics per task type
        """
        stats = {}
        
        for task_type, generator in self.generators.items():
            stats[task_type.value] = generator.get_statistics()
        
        return stats
    
    def reset_statistics(self):
        """Reset statistics for all generators."""
        for generator in self.generators.values():
            generator.reset_statistics()
    
    def __repr__(self) -> str:
        """String representation of the manager."""
        return f"TaskManager(generators={len(self.generators)})"
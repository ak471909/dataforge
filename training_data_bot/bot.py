"""
Main Training Data Bot class.

This module provides the high-level interface for the entire
training data curation system.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from uuid import UUID, uuid4

from training_data_bot.core import (
    Document,
    TextChunk,
    TrainingExample,
    Dataset,
    TaskType,
    DocumentType,
    ExportFormat,
    ProcessingJob,
    ProcessingStatus,
    QualityReport,
    get_logger,
    LogContext,
    TrainingDataBotError,
    ConfigurationError,
)

from training_data_bot.sources import UnifiedLoader
from training_data_bot.preprocessing import TextPreprocessor
from training_data_bot.ai import AIClient
from training_data_bot.tasks import TaskManager
from training_data_bot.evaluation import QualityEvaluator
from training_data_bot.storage import DatasetExporter


class TrainingDataBot:
    """
    Main Training Data Bot class.
    
    This class provides a high-level interface for:
    - Loading documents from various sources
    - Processing text with task templates
    - Quality assessment and filtering
    - Dataset creation and export
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Training Data Bot.
        
        Args:
            config: Optional configuration overrides
        """
        self.logger = get_logger("training_data_bot")
        self.config = config or {}
        
        # Initialize components
        self._init_components()
        
        # State tracking
        self.documents: Dict[UUID, Document] = {}
        self.datasets: Dict[UUID, Dataset] = {}
        self.jobs: Dict[UUID, ProcessingJob] = {}
        
        self.logger.info("Training Data Bot initialized successfully")
    
    def _init_components(self):
        """Initialize all bot components."""
        try:
            # Document loading
            self.loader = UnifiedLoader()
            
            # Text preprocessing
            chunk_size = self.config.get("chunk_size", 1000)
            chunk_overlap = self.config.get("chunk_overlap", 200)
            self.preprocessor = TextPreprocessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # AI client (will be set by user or from config)
            ai_provider = self.config.get("ai_provider", "openai")
            api_key = self.config.get("api_key")
            
            if api_key:
                self.ai_client = AIClient(provider=ai_provider, api_key=api_key)
            else:
                self.ai_client = None
                self.logger.warning("No AI client configured - set api_key to use task generation")
            
            # Task management
            self.task_manager = TaskManager(ai_client=self.ai_client)
            
            # Quality evaluation
            quality_threshold = self.config.get("quality_threshold", 0.7)
            self.evaluator = QualityEvaluator(quality_threshold=quality_threshold)
            
            # Dataset export
            self.exporter = DatasetExporter()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            raise ConfigurationError(
                "Failed to initialize bot components",
                details={"error": str(e)},
                cause=e
            )
    
    def set_ai_client(self, provider: str, api_key: str, **kwargs):
        """
        Set or update the AI client.
        
        Args:
            provider: AI provider name (openai, anthropic)
            api_key: API key for the provider
            **kwargs: Additional client parameters
        """
        self.ai_client = AIClient(provider=provider, api_key=api_key, **kwargs)
        self.task_manager.ai_client = self.ai_client
        self.logger.info(f"AI client set to {provider}")
    
    async def load_documents(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        doc_types: Optional[List[DocumentType]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Load documents from various sources.
        
        Args:
            sources: File path(s), directory, or URL(s)
            doc_types: Optional filter for document types
            **kwargs: Additional loading parameters
            
        Returns:
            List of loaded Document objects
        """
        with LogContext("load_documents"):
            self.logger.info(f"Loading documents from sources")
            
            # Normalize sources to list
            if isinstance(sources, (str, Path)):
                sources = [sources]
            
            # Load documents
            documents = []
            for source in sources:
                # Handle URLs vs file paths
                if str(source).startswith('http'):
                    # It's a URL
                    doc = await self.loader.load_single(source, **kwargs)
                    documents.append(doc)
                else:
                    # It's a file system path
                    source_path = Path(source)
                    
                    # Check if it exists and is a directory
                    if source_path.exists() and source_path.is_dir():
                        self.logger.info(f"Loading directory: {source_path}")
                        dir_docs = await self.loader.load_directory(source_path, **kwargs)
                        documents.extend(dir_docs)
                    elif source_path.exists():
                        # It's a file
                        doc = await self.loader.load_single(source, **kwargs)
                        documents.append(doc)
                    else:
                        # Path doesn't exist
                        self.logger.error(f"Source not found: {source_path}")
                        raise TrainingDataBotError(f"Source not found: {source_path}")
            
            # Filter by document type if specified
            if doc_types:
                documents = [d for d in documents if d.doc_type in doc_types]
            
            # Store documents
            for doc in documents:
                self.documents[doc.id] = doc
            
            self.logger.info(f"Loaded {len(documents)} documents")
            return documents
            
    async def process_documents(
        self,
        documents: Optional[List[Document]] = None,
        task_types: Optional[List[TaskType]] = None,
        quality_filter: bool = True,
        **kwargs
    ) -> Dataset:
        """
        Process documents into training examples.
        
        Args:
            documents: Documents to process (uses all loaded if None)
            task_types: Types of tasks to generate (default: QA + Summarization)
            quality_filter: Whether to filter by quality threshold
            **kwargs: Additional processing parameters
            
        Returns:
            Dataset with training examples
        """
        with LogContext("process_documents"):
            # Use all documents if none specified
            if documents is None:
                documents = list(self.documents.values())
            
            if not documents:
                raise TrainingDataBotError("No documents to process")
            
            # Default task types
            if task_types is None:
                task_types = [TaskType.QA_GENERATION, TaskType.SUMMARIZATION]
            
            self.logger.info(
                f"Processing {len(documents)} documents",
                task_types=[t.value for t in task_types]
            )
            
            # Step 1: Preprocess into chunks
            all_chunks = []
            for doc in documents:
                chunks = self.preprocessor.process_document(doc)
                all_chunks.extend(chunks)
            
            self.logger.info(f"Created {len(all_chunks)} text chunks")
            
            # Step 2: Generate training examples
            examples = await self.task_manager.create_training_examples(
                chunks=all_chunks,
                task_types=task_types,
                **kwargs
            )
            
            self.logger.info(f"Generated {len(examples)} training examples")
            
            # Step 3: Quality filtering (optional)
            if quality_filter:
                filtered_examples = []
                for example in examples:
                    report = self.evaluator.evaluate_example(example, detailed=False)
                    if report.passed:
                        filtered_examples.append(example)
                
                self.logger.info(
                    f"Quality filtering: {len(filtered_examples)}/{len(examples)} passed"
                )
                examples = filtered_examples
            
            # Step 4: Create dataset
            dataset = Dataset(
                id=uuid4(),
                name=f"Dataset_{len(self.datasets) + 1}",
                description=f"Generated from {len(documents)} documents",
                examples=examples,
                total_examples=len(examples)
            )
            
            # Store dataset
            self.datasets[dataset.id] = dataset
            
            self.logger.info(f"Created dataset with {len(examples)} examples")
            return dataset
    
    async def evaluate_dataset(
        self,
        dataset: Dataset,
        detailed_report: bool = True
    ) -> QualityReport:
        """
        Evaluate dataset quality.
        
        Args:
            dataset: Dataset to evaluate
            detailed_report: Whether to generate detailed report
            
        Returns:
            QualityReport with evaluation results
        """
        with LogContext("evaluate_dataset", dataset_id=str(dataset.id)):
            self.logger.info(f"Evaluating dataset {dataset.name}")
            
            report = self.evaluator.evaluate_dataset(
                dataset=dataset,
                detailed_report=detailed_report
            )
            
            self.logger.info(
                f"Dataset evaluation complete",
                overall_score=report.overall_score,
                passed=report.passed
            )
            
            return report
    
    async def export_dataset(
        self,
        dataset: Dataset,
        output_path: Union[str, Path],
        format: ExportFormat = ExportFormat.JSONL,
        split_data: bool = True,
        **kwargs
    ) -> Path:
        """
        Export dataset to file.
        
        Args:
            dataset: Dataset to export
            output_path: Output file path
            format: Export format (JSONL, JSON, CSV, Parquet)
            split_data: Whether to split into train/val/test
            **kwargs: Additional export options
            
        Returns:
            Path to exported file(s)
        """
        with LogContext("export_dataset", dataset_id=str(dataset.id)):
            self.logger.info(
                f"Exporting dataset to {output_path}",
                format=format.value
            )
            
            result_path = await self.exporter.export_dataset(
                dataset=dataset,
                output_path=output_path,
                format=format,
                split_data=split_data,
                **kwargs
            )
            
            self.logger.info(f"Dataset exported successfully to {result_path}")
            return result_path
    
    async def quick_process(
        self,
        source: Union[str, Path],
        output_path: Union[str, Path],
        task_types: Optional[List[TaskType]] = None,
        export_format: ExportFormat = ExportFormat.JSONL
    ) -> Dataset:
        """
        Quick end-to-end processing: load -> process -> export.
        
        Args:
            source: Document source (file, directory, or URL)
            output_path: Output file path
            task_types: Task types to generate
            export_format: Export format
            
        Returns:
            Created dataset
        """
        with LogContext("quick_process"):
            self.logger.info("Starting quick process workflow")
            
            # Load documents
            documents = await self.load_documents([source])
            
            # Process into dataset
            dataset = await self.process_documents(
                documents=documents,
                task_types=task_types
            )
            
            # Export dataset
            await self.export_dataset(
                dataset=dataset,
                output_path=output_path,
                format=export_format
            )
            
            self.logger.info("Quick process complete")
            return dataset
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the bot's state.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "documents": {
                "total": len(self.documents),
                "by_type": self._count_by_type(self.documents.values(), "doc_type"),
                "total_size": sum(
                    len(doc.content) for doc in self.documents.values()
                )
            },
            "datasets": {
                "total": len(self.datasets),
                "total_examples": sum(
                    len(ds.examples) for ds in self.datasets.values()
                ),
                "by_task_type": self._count_examples_by_task_type()
            },
            "jobs": {
                "total": len(self.jobs),
                "by_status": self._count_by_type(self.jobs.values(), "status"),
                "active": len([
                    j for j in self.jobs.values()
                    if j.status == ProcessingStatus.RUNNING
                ])
            },
            "task_manager": self.task_manager.get_statistics() if self.task_manager else {}
        }
    
    def _count_by_type(self, items, attr_name: str) -> Dict[str, int]:
        """Count items by a specific attribute."""
        counts = {}
        for item in items:
            value = getattr(item, attr_name)
            key = value.value if hasattr(value, 'value') else str(value)
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    def _count_examples_by_task_type(self) -> Dict[str, int]:
        """Count examples by task type across all datasets."""
        counts = {}
        for dataset in self.datasets.values():
            for example in dataset.examples:
                task_type = example.task_type.value
                counts[task_type] = counts.get(task_type, 0) + 1
        return counts
    
    async def cleanup(self):
        """Cleanup resources and close connections."""
        try:
            if self.ai_client:
                await self.ai_client.close()
            
            self.logger.info("Bot cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()
    
    def __repr__(self) -> str:
        """String representation of the bot."""
        return (
            f"TrainingDataBot("
            f"documents={len(self.documents)}, "
            f"datasets={len(self.datasets)})"
        )
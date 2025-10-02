"""
Dataset exporter for multiple formats.

This module handles exporting training datasets to various formats
including JSONL, JSON, CSV, and Parquet.
"""

import json
import csv
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from datetime import datetime

from training_data_bot.core import (
    TrainingExample,
    Dataset,
    ExportFormat,
    get_logger,
    LogContext,
    ExportError,
)


class DatasetExporter:
    """
    Dataset exporter supporting multiple formats.
    
    Exports training examples and datasets to JSONL, JSON, CSV,
    and Parquet formats suitable for ML training pipelines.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the dataset exporter.
        
        Args:
            **kwargs: Additional configuration parameters
        """
        self.logger = get_logger("storage.DatasetExporter")
        self.config = kwargs
        
        self.logger.info("DatasetExporter initialized")
    
    async def export_dataset(
        self,
        dataset: Dataset,
        output_path: Union[str, Path],
        format: ExportFormat = ExportFormat.JSONL,
        split_data: bool = False,
        **kwargs
    ) -> Path:
        """
        Export a dataset to file.
        
        Args:
            dataset: Dataset to export
            output_path: Output file path
            format: Export format (JSONL, JSON, CSV, Parquet)
            split_data: Whether to split into train/val/test files
            **kwargs: Additional export options
            
        Returns:
            Path to exported file(s)
        """
        with LogContext("export_dataset", dataset_id=str(dataset.id)):
            output_path = Path(output_path)
            
            self.logger.info(
                f"Exporting dataset",
                format=format.value,
                examples=len(dataset.examples),
                split_data=split_data
            )
            
            try:
                if split_data:
                    return await self._export_split_dataset(
                        dataset, output_path, format, **kwargs
                    )
                else:
                    return await self._export_single_file(
                        dataset.examples, output_path, format, **kwargs
                    )
            except Exception as e:
                raise ExportError(
                    f"Failed to export dataset: {e}",
                    export_format=format.value,
                    output_path=str(output_path),
                    cause=e
                )
    
    async def export_examples(
        self,
        examples: List[TrainingExample],
        output_path: Union[str, Path],
        format: ExportFormat = ExportFormat.JSONL,
        **kwargs
    ) -> Path:
        """
        Export a list of training examples.
        
        Args:
            examples: List of training examples
            output_path: Output file path
            format: Export format
            **kwargs: Additional export options
            
        Returns:
            Path to exported file
        """
        with LogContext("export_examples"):
            self.logger.info(
                f"Exporting {len(examples)} examples",
                format=format.value
            )
            
            return await self._export_single_file(
                examples, Path(output_path), format, **kwargs
            )
    
    async def _export_single_file(
        self,
        examples: List[TrainingExample],
        output_path: Path,
        format: ExportFormat,
        **kwargs
    ) -> Path:
        """Export examples to a single file."""
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format == ExportFormat.JSONL:
            return await self._export_jsonl(examples, output_path, **kwargs)
        elif format == ExportFormat.JSON:
            return await self._export_json(examples, output_path, **kwargs)
        elif format == ExportFormat.CSV:
            return await self._export_csv(examples, output_path, **kwargs)
        elif format == ExportFormat.PARQUET:
            return await self._export_parquet(examples, output_path, **kwargs)
        else:
            raise ExportError(f"Unsupported export format: {format}")
    
    async def _export_split_dataset(
        self,
        dataset: Dataset,
        output_path: Path,
        format: ExportFormat,
        **kwargs
    ) -> Path:
        """Export dataset split into train/val/test files."""
        # Calculate split sizes
        total = len(dataset.examples)
        train_size = int(total * dataset.train_split)
        val_size = int(total * dataset.validation_split)
        
        # Split examples
        train_examples = dataset.examples[:train_size]
        val_examples = dataset.examples[train_size:train_size + val_size]
        test_examples = dataset.examples[train_size + val_size:]
        
        # Generate file paths
        base_path = output_path.parent / output_path.stem
        extension = self._get_extension(format)
        
        train_path = Path(f"{base_path}_train{extension}")
        val_path = Path(f"{base_path}_val{extension}")
        test_path = Path(f"{base_path}_test{extension}")
        
        # Export each split
        await self._export_single_file(train_examples, train_path, format, **kwargs)
        await self._export_single_file(val_examples, val_path, format, **kwargs)
        await self._export_single_file(test_examples, test_path, format, **kwargs)
        
        self.logger.info(
            f"Dataset split exported",
            train_size=len(train_examples),
            val_size=len(val_examples),
            test_size=len(test_examples)
        )
        
        return base_path.parent
    
    async def _export_jsonl(
        self,
        examples: List[TrainingExample],
        output_path: Path,
        **kwargs
    ) -> Path:
        """Export to JSONL format (one JSON object per line)."""
        include_metadata = kwargs.get("include_metadata", True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                data = self._example_to_dict(example, include_metadata)
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        self.logger.debug(f"Exported {len(examples)} examples to JSONL: {output_path}")
        return output_path
    
    async def _export_json(
        self,
        examples: List[TrainingExample],
        output_path: Path,
        **kwargs
    ) -> Path:
        """Export to JSON format (single array)."""
        include_metadata = kwargs.get("include_metadata", True)
        pretty_print = kwargs.get("pretty_print", True)
        
        data = [self._example_to_dict(ex, include_metadata) for ex in examples]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        self.logger.debug(f"Exported {len(examples)} examples to JSON: {output_path}")
        return output_path
    
    async def _export_csv(
        self,
        examples: List[TrainingExample],
        output_path: Path,
        **kwargs
    ) -> Path:
        """Export to CSV format."""
        include_metadata = kwargs.get("include_metadata", False)
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            # Determine fieldnames
            if include_metadata:
                fieldnames = [
                    'id', 'input_text', 'output_text', 'task_type',
                    'source_document_id', 'quality_scores', 'metadata'
                ]
            else:
                fieldnames = ['input_text', 'output_text', 'task_type']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for example in examples:
                row = {
                    'input_text': example.input_text,
                    'output_text': example.output_text,
                    'task_type': example.task_type.value,
                }
                
                if include_metadata:
                    row.update({
                        'id': str(example.id),
                        'source_document_id': str(example.source_document_id),
                        'quality_scores': json.dumps(example.quality_scores),
                        'metadata': json.dumps(example.metadata),
                    })
                
                writer.writerow(row)
        
        self.logger.debug(f"Exported {len(examples)} examples to CSV: {output_path}")
        return output_path
    
    async def _export_parquet(
        self,
        examples: List[TrainingExample],
        output_path: Path,
        **kwargs
    ) -> Path:
        """Export to Parquet format."""
        try:
            import pandas as pd
        except ImportError:
            raise ExportError(
                "pandas is required for Parquet export. Install with: pip install pandas pyarrow"
            )
        
        include_metadata = kwargs.get("include_metadata", True)
        
        # Convert examples to DataFrame
        data = [self._example_to_dict(ex, include_metadata) for ex in examples]
        df = pd.DataFrame(data)
        
        # Write to Parquet
        df.to_parquet(output_path, index=False)
        
        self.logger.debug(f"Exported {len(examples)} examples to Parquet: {output_path}")
        return output_path
    
    def _example_to_dict(
        self,
        example: TrainingExample,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Convert training example to dictionary."""
        data = {
            'input': example.input_text,
            'output': example.output_text,
            'task_type': example.task_type.value,
        }
        
        if include_metadata:
            data.update({
                'id': str(example.id),
                'source_document_id': str(example.source_document_id),
                'source_chunk_id': str(example.source_chunk_id) if example.source_chunk_id else None,
                'quality_scores': example.quality_scores,
                'difficulty': example.difficulty,
                'tags': example.tags,
                'metadata': example.metadata,
                'created_at': example.created_at.isoformat() if example.created_at else None,
            })
        
        return data
    
    def _get_extension(self, format: ExportFormat) -> str:
        """Get file extension for format."""
        extensions = {
            ExportFormat.JSONL: '.jsonl',
            ExportFormat.JSON: '.json',
            ExportFormat.CSV: '.csv',
            ExportFormat.PARQUET: '.parquet',
        }
        return extensions.get(format, '.txt')
    
    def get_export_info(self, output_path: Path) -> Dict[str, Any]:
        """
        Get information about an exported file.
        
        Args:
            output_path: Path to exported file
            
        Returns:
            Dictionary with file information
        """
        if not output_path.exists():
            return {"exists": False}
        
        return {
            "exists": True,
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "size_mb": output_path.stat().st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(output_path.stat().st_mtime).isoformat(),
        }
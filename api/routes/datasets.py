"""
Dataset management endpoints.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from uuid import UUID

from training_data_bot.core import ExportFormat
from api.dependencies import get_bot

router = APIRouter()


class ExportRequest(BaseModel):
    """Request model for dataset export."""
    format: str = "jsonl"
    split_data: bool = True


@router.get("/list")
async def list_datasets():
    """List all created datasets."""
    bot = get_bot()
    
    datasets = []
    for dataset_id, dataset in bot.datasets.items():
        datasets.append({
            "id": str(dataset.id),
            "name": dataset.name,
            "description": dataset.description,
            "total_examples": dataset.total_examples,
            "created_at": dataset.created_at.isoformat()
        })
    
    return {
        "total": len(datasets),
        "datasets": datasets
    }


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get details of a specific dataset."""
    bot = get_bot()
    
    try:
        ds_uuid = UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset ID")
    
    if ds_uuid not in bot.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = bot.datasets[ds_uuid]
    
    return {
        "id": str(dataset.id),
        "name": dataset.name,
        "description": dataset.description,
        "total_examples": dataset.total_examples,
        "train_split": dataset.train_split,
        "validation_split": dataset.validation_split,
        "test_split": dataset.test_split,
        "task_distribution": {k.value: v for k, v in dataset.task_distribution.items()},
        "created_at": dataset.created_at.isoformat()
    }


@router.get("/{dataset_id}/examples")
async def get_dataset_examples(
    dataset_id: str,
    limit: int = 10,
    offset: int = 0
):
    """Get examples from a dataset with pagination."""
    bot = get_bot()
    
    try:
        ds_uuid = UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset ID")
    
    if ds_uuid not in bot.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = bot.datasets[ds_uuid]
    examples = dataset.examples[offset:offset + limit]
    
    return {
        "dataset_id": str(dataset.id),
        "total_examples": len(dataset.examples),
        "offset": offset,
        "limit": limit,
        "examples": [
            {
                "id": str(ex.id),
                "input_text": ex.input_text[:200] + "..." if len(ex.input_text) > 200 else ex.input_text,
                "output_text": ex.output_text[:200] + "..." if len(ex.output_text) > 200 else ex.output_text,
                "task_type": ex.task_type.value,
                "quality_scores": ex.quality_scores
            }
            for ex in examples
        ]
    }


@router.post("/{dataset_id}/export")
async def export_dataset(dataset_id: str, request: ExportRequest):
    """Export a dataset to file."""
    bot = get_bot()
    
    try:
        ds_uuid = UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset ID")
    
    if ds_uuid not in bot.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = bot.datasets[ds_uuid]
    
    # Validate format
    valid_formats = ["jsonl", "json", "csv", "parquet"]
    if request.format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format: {request.format}. Valid: {valid_formats}"
        )
    
    # Create export directory
    export_dir = Path("output/api_exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Export path
    output_path = export_dir / f"{dataset.name}.{request.format}"
    
    try:
        # Export dataset
        result_path = await bot.export_dataset(
            dataset=dataset,
            output_path=output_path,
            format=ExportFormat(request.format),
            split_data=request.split_data
        )
        
        return {
            "status": "success",
            "dataset_id": str(dataset.id),
            "export_path": str(result_path),
            "format": request.format,
            "split_data": request.split_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/download")
async def download_dataset(dataset_id: str, format: str = "jsonl"):
    """Download a dataset file."""
    bot = get_bot()
    
    try:
        ds_uuid = UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset ID")
    
    if ds_uuid not in bot.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = bot.datasets[ds_uuid]
    
    # Find exported file
    export_dir = Path("output/api_exports")
    file_path = export_dir / f"{dataset.name}_train.{format}"
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="File not exported yet. Export dataset first."
        )
    
    return FileResponse(
        path=file_path,
        filename=f"{dataset.name}.{format}",
        media_type="application/octet-stream"
    )


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    bot = get_bot()
    
    try:
        ds_uuid = UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset ID")
    
    if ds_uuid not in bot.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    del bot.datasets[ds_uuid]
    
    return {"status": "success", "message": "Dataset deleted"}
"""
Document processing endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

from training_data_bot.core import TaskType
from api.dependencies import get_bot

router = APIRouter()


class ProcessRequest(BaseModel):
    """Request model for document processing."""
    document_paths: List[str]
    task_types: List[str] = ["qa_generation", "summarization"]
    quality_filter: bool = True
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


class ProcessUploadedRequest(BaseModel):
    """Request model for processing uploaded documents."""
    task_types: List[str] = ["qa_generation", "summarization"]
    quality_filter: bool = True


@router.post("/process")
async def process_documents(request: ProcessRequest):
    """
    Process documents from file paths.
    
    Args:
        document_paths: List of file paths to process
        task_types: Types of tasks to generate
        quality_filter: Whether to filter by quality threshold
        chunk_size: Optional chunk size override
        chunk_overlap: Optional chunk overlap override
    """
    bot = get_bot()
    
    # Validate task types
    valid_tasks = ["qa_generation", "summarization", "classification"]
    for task in request.task_types:
        if task not in valid_tasks:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task type: {task}. Valid: {valid_tasks}"
            )
    
    # Convert to TaskType enums
    task_types = [TaskType(task) for task in request.task_types]
    
    try:
        # Load documents
        documents = await bot.load_documents(request.document_paths)
        
        # Process documents
        dataset = await bot.process_documents(
            documents=documents,
            task_types=task_types,
            quality_filter=request.quality_filter
        )
        
        # Evaluate dataset
        report = await bot.evaluate_dataset(dataset)
        
        return {
            "status": "success",
            "dataset_id": str(dataset.id),
            "documents_processed": len(documents),
            "examples_generated": len(dataset.examples),
            "quality_score": report.overall_score,
            "quality_passed": report.passed,
            "task_types": request.task_types
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-uploaded")
async def process_uploaded_documents(request: ProcessUploadedRequest):
    """
    Process all uploaded documents.
    
    Args:
        task_types: Types of tasks to generate
        quality_filter: Whether to filter by quality threshold
    """
    bot = get_bot()
    
    if not bot.documents:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded. Upload documents first."
        )
    
    # Validate task types
    valid_tasks = ["qa_generation", "summarization", "classification"]
    for task in request.task_types:
        if task not in valid_tasks:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task type: {task}. Valid: {valid_tasks}"
            )
    
    # Convert to TaskType enums
    task_types = [TaskType(task) for task in request.task_types]
    
    try:
        # Process all uploaded documents
        documents = list(bot.documents.values())
        
        dataset = await bot.process_documents(
            documents=documents,
            task_types=task_types,
            quality_filter=request.quality_filter
        )
        
        # Evaluate dataset
        report = await bot.evaluate_dataset(dataset)
        
        return {
            "status": "success",
            "dataset_id": str(dataset.id),
            "documents_processed": len(documents),
            "examples_generated": len(dataset.examples),
            "quality_score": report.overall_score,
            "quality_passed": report.passed,
            "task_types": request.task_types
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_processing_status():
    """Get current processing status and statistics."""
    bot = get_bot()
    stats = bot.get_statistics()
    
    return {
        "status": "ready",
        "documents_loaded": stats["documents"]["total"],
        "datasets_created": stats["datasets"]["total"],
        "total_examples": stats["datasets"]["total_examples"],
        "statistics": stats
    }
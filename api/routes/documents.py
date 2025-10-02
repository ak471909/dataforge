"""
Document management endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path
import shutil
from uuid import uuid4

from training_data_bot.core import DocumentType
from api.dependencies import get_bot
from training_data_bot.core import get_logger

logger = get_logger("api.documents")

router = APIRouter()

# Temporary upload directory
UPLOAD_DIR = Path("temp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    description: Optional[str] = Form(None)
):
    """
    Upload one or more documents for processing.
    
    Supports: PDF, DOCX, TXT, MD, HTML, CSV, JSON
    """
    bot = get_bot()
    uploaded_files = []
    
    try:
        logger.info(f"Received upload request with {len(files)} files")
        
        file_paths = []
        
        for file in files:
            # Validate file extension
            file_ext = Path(file.filename).suffix.lower().lstrip('.')
            
            if file_ext not in ['pdf', 'docx', 'txt', 'md', 'html', 'csv', 'json']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}"
                )
            
            # Generate unique filename
            unique_filename = f"{uuid4()}_{file.filename}"
            file_path = UPLOAD_DIR / unique_filename
            
            logger.info(f"Saving file to: {file_path}")
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_paths.append(str(file_path))
            
            uploaded_files.append({
                "filename": file.filename,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "type": file_ext
            })
        
        # Load documents into bot
        logger.info(f"Loading {len(file_paths)} documents into bot")
        documents = await bot.load_documents(file_paths)
        logger.info(f"Successfully loaded {len(documents)} documents")
        
        return {
            "status": "success",
            "files_uploaded": len(uploaded_files),
            "documents_loaded": len(documents),
            "files": uploaded_files
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/list")
async def list_documents():
    """List all uploaded documents."""
    bot = get_bot()
    
    documents = []
    for doc_id, doc in bot.documents.items():
        documents.append({
            "id": str(doc.id),
            "title": doc.title,
            "type": doc.doc_type.value,
            "word_count": doc.word_count,
            "char_count": doc.char_count,
            "source": doc.source,
            "created_at": doc.created_at.isoformat()
        })
    
    return {
        "total": len(documents),
        "documents": documents
    }


@router.get("/{document_id}")
async def get_document(document_id: str):
    """Get details of a specific document."""
    bot = get_bot()
    
    from uuid import UUID
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    if doc_uuid not in bot.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = bot.documents[doc_uuid]
    
    return {
        "id": str(doc.id),
        "title": doc.title,
        "type": doc.doc_type.value,
        "word_count": doc.word_count,
        "char_count": doc.char_count,
        "source": doc.source,
        "content_preview": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
        "created_at": doc.created_at.isoformat(),
        "metadata": doc.metadata
    }


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document."""
    bot = get_bot()
    
    from uuid import UUID
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    if doc_uuid not in bot.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    del bot.documents[doc_uuid]
    
    return {"status": "success", "message": "Document deleted"}
"""
Document management routes.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.api.models.schemas import UploadResponse, DocumentsListResponse
from src.api.services.rag_service import rag_service
from src.api.core.config import api_config

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    # Validate file type
    file_extension = f".{file.filename.split('.')[-1].lower()}" if '.' in file.filename else ""
    
    if file_extension not in api_config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file_extension}' not supported. Allowed: {', '.join(api_config.ALLOWED_EXTENSIONS)}"
        )
    
    return await rag_service.upload_document(file, api_config.UPLOAD_DIR)

@router.get("/", response_model=DocumentsListResponse)
async def list_documents():
    """List all processed documents."""
    documents = rag_service.list_documents()
    return DocumentsListResponse(documents=documents)
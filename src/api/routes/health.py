"""
Health check routes.
"""
from fastapi import APIRouter
import datetime
from src.api.models.schemas import HealthResponse
from src.api.services.rag_service import rag_service

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return rag_service.get_health_info()

@router.get("/upload-status")
async def upload_status():
    """Check if uploads are processing."""
    try:
        # Get some basic stats to show the system is alive
        stats = rag_service.rag_pipeline.get_stats()
        return {
            "status": "processing" if stats.get("documents_processed", 0) > 0 else "ready",
            "documents_processed": stats.get("documents_processed", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
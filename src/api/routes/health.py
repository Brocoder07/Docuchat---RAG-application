"""
Health check routes.
"""
from fastapi import APIRouter
from src.api.models.schemas import HealthResponse
from src.api.services.rag_service import rag_service

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return rag_service.get_health_info()
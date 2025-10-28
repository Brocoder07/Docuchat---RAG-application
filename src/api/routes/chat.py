"""
Enhanced chat and query routes with better source tracking.
"""
from fastapi import APIRouter
from src.api.models.schemas import QueryRequest, QueryResponse
from src.api.services.rag_service import rag_service

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents with enhanced source information."""
    answer, relevant_sources, confidence, source_info = rag_service.query_documents(
        question=request.question,
        top_k=request.top_k
    )
    
    return QueryResponse(
        answer=answer,
        relevant_sources=relevant_sources,
        confidence=confidence,
        source_info=source_info
    )
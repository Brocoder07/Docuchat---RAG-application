"""
Chat and query routes.
"""
from fastapi import APIRouter
from src.api.models.schemas import QueryRequest, QueryResponse
from src.api.services.rag_service import rag_service

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents with a question."""
    answer, relevant_sources, confidence = rag_service.query_documents(
        question=request.question,
        top_k=request.top_k
    )
    
    return QueryResponse(
        answer=answer,
        relevant_sources=relevant_sources,
        confidence=confidence
    )
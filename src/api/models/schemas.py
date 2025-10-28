"""
Enhanced Pydantic models for request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    documents_loaded: int

class QueryRequest(BaseModel):
    """Query request schema."""
    question: str = Field(..., description="The question to ask")
    top_k: Optional[int] = Field(3, description="Number of relevant chunks to retrieve")

class SourceInfo(BaseModel):
    """Source information schema."""
    total_sources: int
    documents: List[str]
    primary_source: str
    chunk_details: List[Dict[str, Any]]

class QueryResponse(BaseModel):
    """Enhanced query response schema."""
    answer: str
    relevant_sources: int
    confidence: str
    source_info: SourceInfo  # Add detailed source information

class UploadResponse(BaseModel):
    """Upload response schema."""
    message: str
    filename: str
    file_id: str

class DocumentInfo(BaseModel):
    """Document information schema."""
    filename: str
    chunks: int
    source: str
    document_id: str

class DocumentsListResponse(BaseModel):
    """Documents list response."""
    documents: List[DocumentInfo]

class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str
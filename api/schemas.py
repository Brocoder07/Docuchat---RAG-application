"""
API request/response schemas with comprehensive validation.
Senior Engineer Principle: Strong typing prevents bugs and improves documentation.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import re

class HealthResponse(BaseModel):
    """Health check response with system status."""
    status: str = Field(..., description="Overall system status")
    documents_processed: int = Field(..., description="Number of processed documents")
    total_chunks: int = Field(..., description="Total chunks in vector store")
    llm_ready: bool = Field(..., description="LLM service status")
    llm_model: str = Field(..., description="Active LLM model")
    vector_store_initialized: bool = Field(..., description="Vector store status")
    timestamp: str = Field(..., description="Response timestamp")  # 🚨 Changed to str
    
    @validator('timestamp', pre=True, always=True)
    def convert_timestamp(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        return v
    
    @validator('status')
    def validate_status(cls, v):
        allowed_statuses = ['healthy', 'degraded', 'unhealthy', 'initializing']
        if v not in allowed_statuses:
            raise ValueError(f'Status must be one of: {allowed_statuses}')
        return v

class QueryRequest(BaseModel):
    """Query request with validation."""
    question: str = Field(..., min_length=1, max_length=1000, description="User question")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    
    @validator('question')
    def validate_question(cls, v):
        # Remove excessive whitespace
        v = re.sub(r'\s+', ' ', v).strip()
        if len(v) < 2:
            raise ValueError('Question must be at least 2 characters long')
        return v

class SourceChunk(BaseModel):
    """Individual source chunk information."""
    document: str = Field(..., description="Source document name")
    content_preview: str = Field(..., description="Chunk content preview")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    chunk_id: str = Field(..., description="Unique chunk identifier")

class SourceInfo(BaseModel):
    """Comprehensive source information."""
    total_sources: int = Field(..., ge=0, description="Number of source documents")
    documents: List[str] = Field(..., description="List of source documents")
    primary_source: str = Field(..., description="Primary source document")
    chunk_details: List[SourceChunk] = Field(..., description="Detailed chunk information")
    retrieved_count: int = Field(..., ge=0, description="Total chunks retrieved")

class QueryResponse(BaseModel):
    """Query response with answer and sources."""
    answer: str = Field(..., description="Generated answer")
    relevant_sources: int = Field(..., ge=0, description="Number of relevant sources")
    confidence: str = Field(..., description="Answer confidence level")
    source_info: SourceInfo = Field(..., description="Detailed source information")
    model_used: str = Field(..., description="LLM model used for generation")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

class UploadResponse(BaseModel):
    """Document upload response."""
    message: str = Field(..., description="Upload status message")
    filename: str = Field(..., description="Original filename")
    file_id: str = Field(..., description="Unique file identifier")
    chunks_count: int = Field(..., ge=0, description="Number of chunks created")
    document_id: str = Field(..., description="Unique document identifier")

class DocumentInfo(BaseModel):
    """Document metadata information."""
    filename: str = Field(..., description="Document filename")
    chunks: int = Field(..., ge=0, description="Number of chunks")
    source: str = Field(..., description="File path or source")
    document_id: str = Field(..., description="Unique document identifier")
    processed_at: str = Field(..., description="Processing timestamp")

class DocumentsListResponse(BaseModel):
    """List of processed documents."""
    documents: List[DocumentInfo] = Field(..., description="List of documents")
    total_count: int = Field(..., ge=0, description="Total documents count")

class ErrorResponse(BaseModel):
    """Standardized error response."""
    detail: str = Field(..., description="Error description")
    error_type: Optional[str] = Field(None, description="Error category")
    timestamp: str = Field(..., description="Error timestamp")  # 🚨 Changed to str
    
    @validator('timestamp', pre=True, always=True)
    def convert_timestamp(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        return v

class SystemStatusResponse(BaseModel):
    """Comprehensive system status."""
    pipeline: Dict[str, Any] = Field(..., description="Pipeline status")
    vector_store: Dict[str, Any] = Field(..., description="Vector store status")
    llm_service: Dict[str, Any] = Field(..., description="LLM service status")
    evaluation: Dict[str, Any] = Field(..., description="Evaluation metrics")
    performance_alerts: List[str] = Field(..., description="Active performance alerts")

class DeleteResponse(BaseModel):
    """Document deletion response."""
    message: str = Field(..., description="Deletion status message")
    document_id: str = Field(..., description="Deleted document identifier")
    chunks_removed: int = Field(..., ge=0, description="Number of chunks removed")
"""
Consolidated API routes with comprehensive error handling.
Senior Engineer Principle: Single file for all routes improves maintainability.
"""
import logging
import uuid
import os
import time
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from typing import List, Optional

from core.rag_pipeline import rag_pipeline
from core.evaluator import evaluator
from api.schemas import (
    HealthResponse, QueryRequest, QueryResponse, UploadResponse,
    DocumentsListResponse, DocumentInfo, SystemStatusResponse, 
    ErrorResponse, DeleteResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get(
    "/health", 
    response_model=HealthResponse,
    summary="System Health Check",
    description="Get overall system health status and component statuses"
)
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        status = rag_pipeline.get_status()
        
        # Determine overall system status
        overall_status = "healthy"
        if not status["initialized"]:
            overall_status = "unhealthy"
        elif status["vector_store"].get("error"):
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            documents_processed=status["documents_processed"],
            total_chunks=status["vector_store"].get("total_chunks", 0),
            llm_ready=status["llm_service"].get("initialized", False),
            llm_model=status["llm_service"].get("current_model", "unknown"),
            vector_store_initialized=status["vector_store"].get("initialized", False),
            timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail="Health check failed",
                error_type="system_error",
                timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
            ).dict()
        )

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query Documents",
    description="Ask questions about your uploaded documents"
)
async def query_documents(request: QueryRequest):
    """Query documents with performance tracking."""
    start_time = time.time()
    
    try:
        # Validate pipeline state
        if not rag_pipeline.initialized:
            raise HTTPException(
                status_code=503,
                detail=ErrorResponse(
                    detail="RAG pipeline not initialized",
                    error_type="service_unavailable",
                    timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
                ).dict()
            )
        
        # Process query
        result = rag_pipeline.query(request.question, top_k=request.top_k)
        processing_time = time.time() - start_time
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    detail=result.get("error", "Query processing failed"),
                    error_type="query_error",
                    timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
                ).dict()
            )
        
        # 🚨 CRITICAL FIX: Ensure source_info is always present
        source_info = result.get("source_info")
        if not source_info:
            source_info = {
                "total_sources": 0,
                "documents": [],
                "primary_source": "None",
                "chunk_details": [],
                "retrieved_count": result.get("chunks_retrieved", 0)
            }
        
        # Track evaluation metrics
        evaluator.evaluate_query(
            question=request.question,
            answer=result["answer"],
            relevant_chunks=result["sources"],
            response_time=processing_time
        )
        
        return QueryResponse(
            answer=result["answer"],
            relevant_sources=result["chunks_retrieved"],
            confidence=result["confidence"],
            source_info=source_info,
            model_used=result.get("model_used", "unknown"),
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail="Internal server error during query processing",
                error_type="internal_error",
                timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
            ).dict()
        )

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload Document",
    description="Upload and process a document for querying"
)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document with comprehensive validation."""
    try:
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        allowed_extensions = ['.pdf', '.txt', '.docx', '.md']
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    detail=f"File type '{file_extension}' not supported. Allowed: {', '.join(allowed_extensions)}",
                    error_type="validation_error",
                    timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
                ).dict()
            )
        
        # Validate file size (50MB max)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(
                status_code=413,
                detail=ErrorResponse(
                    detail="File too large. Maximum size is 50MB",
                    error_type="validation_error",
                    timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
                ).dict()
            )
        
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    detail="Uploaded file is empty",
                    error_type="validation_error",
                    timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
                ).dict()
            )
        
        file_id = str(uuid.uuid4())
        temp_filename = f"{file_id}_{file.filename}"
        temp_path = os.path.join("data/uploads", temp_filename)
        
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        time.sleep(0.1)  # Ensure file is written
        # Process document
        result = rag_pipeline.process_document(temp_path, file.filename)
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    detail=result.get("error", "Document processing failed"),
                    error_type="processing_error",
                    timestamp=datetime.now().isoformat()
                ).dict()
            )
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            file_id=file_id,
            chunks_count=result["chunks_count"],
            document_id=result["document_id"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail="Internal server error during document upload",
                error_type="internal_error",
                timestamp=datetime.now().isoformat()
            ).dict()
        )
    finally:
        # 🚨 CRITICAL: Clean up temp file in finally block
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Cleaned up temp file: {temp_path}")
            except OSError as e:
                logger.warning(f"Could not remove temp file {temp_path}: {e}")

@router.get(
    "/documents",
    response_model=DocumentsListResponse,
    summary="List Documents",
    description="Get list of all processed documents"
)
async def list_documents():
    """List all processed documents."""
    try:
        documents = rag_pipeline.list_documents()
        
        document_list = []
        for doc in documents:
            document_list.append(DocumentInfo(
                filename=doc["filename"],
                chunks=doc["chunks_count"],
                source=doc["file_path"],
                document_id=doc["document_id"],
                processed_at=doc["processed_at"]
            ))
        
        return DocumentsListResponse(
            documents=document_list,
            total_count=len(document_list)
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail="Failed to retrieve document list",
                error_type="internal_error",
                timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
            ).dict()
        )

@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    summary="Delete Document",
    description="Remove a document and all its chunks from the system"
)
async def delete_document(document_id: str):
    """Delete a specific document."""
    try:
        success = rag_pipeline.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    detail=f"Document {document_id} not found",
                    error_type="not_found",
                    timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
                ).dict()
            )
        
        return DeleteResponse(
            message="Document deleted successfully",
            document_id=document_id,
            chunks_removed=1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail="Failed to delete document",
                error_type="internal_error",
                timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
            ).dict()
        )

@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="System Status",
    description="Get detailed system status and performance metrics"
)
async def system_status():
    """Get comprehensive system status."""
    try:
        pipeline_status = rag_pipeline.get_status()
        evaluation_metrics = evaluator.get_aggregate_metrics(hours=24)
        performance_alerts = evaluator.get_performance_alerts()
        
        # Format alerts as readable messages
        alert_messages = []
        for alert in performance_alerts:
            alert_messages.append(f"{alert['type']}: {alert['message']}")
        
        return SystemStatusResponse(
            pipeline=pipeline_status,
            vector_store=pipeline_status["vector_store"],
            llm_service=pipeline_status["llm_service"],
            evaluation={
                "recent_metrics": evaluation_metrics,
                "performance_alerts": performance_alerts
            },
            performance_alerts=alert_messages
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail="Failed to retrieve system status",
                error_type="internal_error",
                timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
            ).dict()
        )

@router.get(
    "/evaluation/metrics",
    summary="Evaluation Metrics",
    description="Get RAG evaluation metrics for monitoring and optimization"
)
async def get_evaluation_metrics(hours: int = Query(24, ge=1, le=168)):
    """Get evaluation metrics for specified time period."""
    try:
        metrics = evaluator.get_aggregate_metrics(hours=hours)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get evaluation metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail="Failed to retrieve evaluation metrics",
                error_type="internal_error",
                timestamp=datetime.now().isoformat()  # 🚨 Use ISO string
            ).dict()
        )

@router.post("/debug/reset-vector-store")
async def reset_vector_store():
    """Reset vector store (development only)."""
    try:
        success = rag_pipeline.vector_store.reset_collection()
        if success:
            rag_pipeline.processed_documents = []
            return {"message": "Vector store reset successfully"}
        else:
            raise HTTPException(status_code=500, detail="Reset failed")
    except Exception as e:
        logger.error(f"Vector store reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
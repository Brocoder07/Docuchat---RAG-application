"""
Consolidated API routes with comprehensive error handling.
FIXED: Removed broken metrics from /status response and made upload async-friendly.
ADDED: Duplicate file check (per-user) using MD5 hashing on upload.
"""
import logging
import uuid
import os
import time
import hashlib  # 🚨 Import hashlib for hashing
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query, Depends, Form
from typing import List, Optional

from core.rag_pipeline import rag_pipeline
from core.evaluator import evaluator
from core.firebase import get_current_user
from api.schemas import (
    HealthResponse, QueryRequest, QueryResponse, UploadResponse,
    DocumentsListResponse, DocumentInfo, SystemStatusResponse, 
    ErrorResponse, DeleteResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

# -----------------------------------------------------------------
# Helper background wrapper
# -----------------------------------------------------------------
def _background_process_document(
    temp_path: str, 
    filename: str, 
    user_uid: str, 
    document_id: str,
    file_hash: str  # 🚨 Pass the hash
):
    """
    Background worker wrapper that calls the synchronous processing function.
    Logs outcomes. This runs in the background and must handle its own exceptions.
    """
    try:
        logger.info(f"Background: Starting processing for {filename} (ID: {document_id}) user {user_uid}")
        
        # 🚨 Pass the hash to the processing function
        result = rag_pipeline.process_document(
            temp_path, 
            filename, 
            user_id=user_uid,
            file_hash=file_hash,
            document_id=document_id # Pass the doc_id from the route
        )
        
        if result.get("success"):
            logger.info(f"Background: Document {filename} processed successfully (ID: {document_id})")
        else:
            logger.error(f"Background: Document {filename} processing failed (ID: {document_id}): {result.get('error')}")
    except Exception as e:
        logger.exception(f"Background: Unexpected error while processing {filename} (ID: {document_id}): {e}")
    finally:
        # Try to clean temp file - don't raise
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"Background: Removed temp file {temp_path}")
        except Exception as e:
            logger.warning(f"Background: Failed to cleanup temp file {temp_path}: {e}")

# ... (Health endpoint remains the same) ...
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
        
        overall_status = "healthy"
        if not status["initialized"]:
            overall_status = "unhealthy"
        elif status["vector_store"].get("error"):
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            documents_processed=status["vector_store"].get("unique_documents", 0), 
            total_chunks=status["vector_store"].get("total_chunks", 0),
            llm_ready=status["llm_service"].get("initialized", False),
            llm_model=status["llm_service"].get("current_model", "unknown"),
            vector_store_initialized=status["vector_store"].get("initialized", False),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail="Health check failed",
                error_type="system_error",
                timestamp=datetime.now().isoformat()
            ).dict()
        )

# ... (Query endpoint remains the same) ...
@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query Documents (Secured)",
    description="Ask questions about your uploaded documents"
)
async def query_documents(request: QueryRequest, user_uid: str = Depends(get_current_user)):
    """Query documents with performance tracking."""
    start_time = time.time()
    
    try:
        if not rag_pipeline.initialized:
            raise HTTPException(
                status_code=503,
                detail=ErrorResponse(
                    detail="RAG pipeline not initialized",
                    error_type="service_unavailable",
                    timestamp=datetime.now().isoformat()
                ).dict()
            )
        
        result = rag_pipeline.query(
            question=request.question, 
            top_k=request.top_k, 
            filename=request.filename,
            chat_history=request.chat_history,
            user_id=user_uid
        )
        
        processing_time = time.time() - start_time
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    detail=result.get("error", "Query processing failed"),
                    error_type="query_error",
                    timestamp=datetime.now().isoformat()
                ).dict()
            )
        
        source_info = result.get("source_info")
        if not source_info:
            source_info = {
                "total_sources": 0,
                "documents": [],
                "primary_source": "None",
                "chunk_details": [],
                "retrieved_count": result.get("chunks_retrieved", 0)
            }
        
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
                timestamp=datetime.now().isoformat()
            ).dict()
        )

# -----------------------------------------------------------------
# 🚨 MODIFIED: Upload endpoint (now with duplicate checking)
# -----------------------------------------------------------------
@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload Document (Secured)",
    description="Upload and process a document for querying"
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    user_uid: str = Depends(get_current_user)
):
    """
    Upload and process document with validation and duplicate checking.
    """
    try:
        # --- 1. Validation (as before) ---
        file_extension = os.path.splitext(file.filename)[1].lower()
        allowed_extensions = ['.pdf', '.txt', '.docx', '.md']
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail="File type not supported.")
        
        # --- 2. Hashing and Duplicate Check (NEW) ---
        logger.debug(f"Reading file {file.filename} for hashing...")
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        file_hash = hashlib.md5(content).hexdigest()
        
        if rag_pipeline.check_document_exists(file_hash, user_uid):
            logger.warning(f"User {user_uid} attempted to upload duplicate file: {file.filename} (Hash: {file_hash})")
            raise HTTPException(
                status_code=409,  # 409 Conflict
                detail=f"This exact file ({file.filename}) has already been uploaded."
            )
        
        # --- 3. Save to Temp File (as before) ---
        document_id = str(uuid.uuid4())
        temp_filename = f"{document_id}_{file.filename}"
        temp_path = os.path.join("data/uploads", temp_filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(content)  # Write the content we already read
        
        # --- 4. Schedule Background Task (with hash) ---
        background_tasks.add_task(
            _background_process_document, 
            temp_path, 
            file.filename, 
            user_uid, 
            document_id,
            file_hash  # 🚨 Pass the hash
        )
        
        return UploadResponse(
            message="Document uploaded successfully and scheduled for processing",
            filename=file.filename,
            file_id=str(uuid.uuid4()), # This is just a response ID
            chunks_count=0,
            document_id=document_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail="Internal server error during document upload",
                error_type="internal_error",
                timestamp=datetime.now().isoformat()
            ).dict()
        )

# ... (rest of the file remains the same) ...
# -----------------------------------------------------------------
# Documents listing
# -----------------------------------------------------------------
@router.get(
    "/documents",
    response_model=DocumentsListResponse,
    summary="List Documents (Secured)",
    description="Get list of all processed documents for the current user"
)
async def list_documents(user_uid: str = Depends(get_current_user)):
    """List all processed documents for the current user."""
    try:
        documents = rag_pipeline.list_documents(user_id=user_uid)
        
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
                timestamp=datetime.now().isoformat()
            ).dict()
        )

# -----------------------------------------------------------------
# Delete endpoint
# -----------------------------------------------------------------
@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    summary="Delete Document (Secured)",
    description="Remove a document and all its chunks from the system (user-owned only)"
)
async def delete_document(document_id: str, user_uid: str = Depends(get_current_user)):
    """Delete a specific document, verifying user ownership."""
    try:
        success = rag_pipeline.delete_document(document_id, user_id=user_uid)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    detail=f"Document {document_id} not found or user does not have permission",
                    error_type="not_found_or_forbidden",
                    timestamp=datetime.now().isoformat()
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
                timestamp=datetime.now().isoformat()
            ).dict()
        )

# -----------------------------------------------------------------
# Status & metrics remain unchanged (returns background-processed docs as they become available)
# -----------------------------------------------------------------
@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="System Status (Secured)",
    description="Get detailed system status and performance metrics"
)
async def system_status(user_uid: str = Depends(get_current_user)):
    """Get comprehensive system status."""
    try:
        pipeline_status = rag_pipeline.get_status(user_id=user_uid)
        
        # These are now reliable
        evaluation_metrics = evaluator.get_aggregate_metrics(hours=24)
        performance_alerts = evaluator.get_performance_alerts()
        
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
                timestamp=datetime.now().isoformat()
            ).dict()
        )

@router.get(
    "/evaluation/metrics",
    summary="Evaluation Metrics (Secured)",
    description="Get RAG evaluation metrics for monitoring and optimization"
)
async def get_evaluation_metrics(hours: int = Query(24, ge=1, le=168), user_uid: str = Depends(get_current_user)):
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
                timestamp=datetime.now().isoformat()
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
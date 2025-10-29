"""
Enhanced RAG service with universal hallucination prevention and production monitoring.
"""
import os
import uuid
import logging
from typing import List, Optional, Tuple, Dict, Any
from fastapi import UploadFile, HTTPException

from src.rag_pipeline import RAGPipeline
from src.api.models.schemas import UploadResponse, DocumentInfo
from src.api.core.config import api_config

logger = logging.getLogger(__name__)

class HallucinationMonitor:
    """Monitor and track hallucination patterns for production."""
    
    def __init__(self):
        self.hallucination_count = 0
        self.total_queries = 0
        self.low_confidence_queries = 0
    
    def track_query(self, question: str, verification_results: Dict):
        """Track query results for monitoring."""
        self.total_queries += 1
        
        if verification_results["confidence"] < 0.7:
            self.low_confidence_queries += 1
            logger.warning(f"Low confidence answer: '{question[:100]}...' (confidence: {verification_results['confidence']:.2f})")
        
        if verification_results["confidence"] < 0.4:
            self.hallucination_count += 1
            logger.error(f"POTENTIAL HALLUCINATION: '{question[:100]}...' (confidence: {verification_results['confidence']:.2f})")
        
        # Log metrics periodically
        if self.total_queries % 10 == 0:
            self._log_metrics()
    
    def _log_metrics(self):
        """Log monitoring metrics."""
        if self.total_queries > 0:
            hallucination_rate = self.hallucination_count / self.total_queries
            low_confidence_rate = self.low_confidence_queries / self.total_queries
            logger.info(
                f"Query Metrics - Total: {self.total_queries}, "
                f"Hallucination rate: {hallucination_rate:.1%}, "
                f"Low confidence rate: {low_confidence_rate:.1%}"
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics."""
        return {
            "total_queries": self.total_queries,
            "hallucination_count": self.hallucination_count,
            "low_confidence_queries": self.low_confidence_queries,
            "hallucination_rate": self.hallucination_count / max(self.total_queries, 1),
            "low_confidence_rate": self.low_confidence_queries / max(self.total_queries, 1)
        }

class RAGService:
    def __init__(self):
        """Initialize RAGService with production monitoring."""
        try:
            self.rag_pipeline = RAGPipeline()
            self.initialized = False
            self.initialization_error = None
            self.monitor = HallucinationMonitor()
        except Exception as e:
            self.rag_pipeline = None
            self.initialized = False
            self.initialization_error = str(e)
            self.monitor = HallucinationMonitor()
            logger.exception("Failed to create RAGPipeline during service init.")

    def initialize(self):
        """Initialize the underlying RAG pipeline safely."""
        try:
            if self.rag_pipeline:
                self.rag_pipeline.initialize()
                self.initialized = True
                self.initialization_error = None
                logger.info("RAGService initialized successfully with production monitoring.")
            else:
                raise RuntimeError("RAGPipeline is not available.")
        except Exception as e:
            self.initialized = False
            self.initialization_error = str(e)
            logger.exception("Error initializing RAGService.")

    @property
    def is_initialized(self):
        """Check if the service is ready."""
        return bool(self.initialized and self.rag_pipeline)

    def get_health_info(self) -> dict:
        """Return overall health/status information with monitoring metrics."""
        try:
            stats = self.rag_pipeline.get_stats() if self.rag_pipeline else {}
            monitoring_metrics = self.monitor.get_metrics()
            
            health_info = {
                "status": "healthy" if self.is_initialized else "initializing",
                "documents_processed": stats.get("documents_processed", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "llm_ready": stats.get("llm_ready", False),
                "llm_model": stats.get("llm_model", "unknown"),
                "vector_store_initialized": stats.get("total_chunks", 0) > 0,
                "monitoring": monitoring_metrics
            }

            if self.initialization_error:
                health_info["initialization_error"] = self.initialization_error
                health_info["status"] = "error"

            return health_info
        except Exception as e:
            logger.exception("Error getting health info.")
            return {
                "status": "error",
                "details": str(e),
                "documents_processed": 0,
                "total_chunks": 0,
                "llm_ready": False,
                "llm_model": "unknown"
            }

    async def upload_document(self, file: UploadFile, upload_dir: str) -> UploadResponse:
        """Upload and process a document with enhanced error handling."""
        if not self.is_initialized:
            self.initialize()
            if not self.is_initialized:
                raise HTTPException(
                    status_code=500,
                    detail=f"RAGService initialization failed: {self.initialization_error or 'Unknown error'}"
                )

        file_extension = os.path.splitext(file.filename)[1].lower()
        if not file_extension or file_extension not in api_config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(api_config.ALLOWED_EXTENSIONS)}"
            )

        try:
            contents = await file.read()
            file_size = len(contents)
            if file_size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            if file_size > api_config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max: {api_config.MAX_FILE_SIZE / (1024*1024):.1f}MB"
                )
            await file.seek(0)
        except Exception as e:
            logger.error(f"Error reading file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error reading uploaded file: {str(e)}")

        try:
            existing_docs = self.rag_pipeline.get_document_list()
            for doc in existing_docs:
                if doc.get("title") == file.filename:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File '{file.filename}' already uploaded."
                    )
        except Exception:
            logger.warning("Duplicate check failed, continuing anyway.")

        file_id = str(uuid.uuid4())[:8]
        saved_filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(upload_dir, saved_filename)

        try:
            with open(file_path, "wb") as f:
                f.write(contents)

            logger.info(f"Saved uploaded file to {file_path}")
            result = self.rag_pipeline.process_document(file_path, filename=file.filename)

            logger.info(f"Document processed successfully: {file.filename}")
            return UploadResponse(
                message="Document uploaded and processed successfully",
                filename=file.filename,
                file_id=result["document_id"]
            )
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error processing document {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    def query_documents(self, question: str, top_k: int = 3) -> Tuple[str, int, str, Dict[str, Any]]:
        """Query documents with universal hallucination prevention."""
        if not self.is_initialized:
            self.initialize()
            if not self.is_initialized:
                raise HTTPException(
                    status_code=500,
                    detail=f"RAGService initialization failed: {self.initialization_error or 'Unknown error'}"
                )

        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        stats = self.rag_pipeline.get_stats()
        if stats.get("total_chunks", 0) == 0:
            raise HTTPException(status_code=400, detail="No documents available. Upload some first.")

        try:
            answer, relevant_chunks, source_info = self.rag_pipeline.query(question)
            
            # Track query for monitoring
            verification_results = source_info.get("verification", {})
            self.monitor.track_query(question, verification_results)
            
            # Ensure source_info has all required fields
            if not source_info or 'total_sources' not in source_info:
                source_info = {
                    "total_sources": 0,
                    "documents": [],
                    "primary_source": "None", 
                    "chunk_details": [],
                    "retrieved_count": len(relevant_chunks),
                    "verification": verification_results
                }
            
            logger.info(f"Query processed successfully, found {len(relevant_chunks)} relevant chunks.")
            confidence = self._calculate_confidence(relevant_chunks, answer, verification_results)
            return answer, len(relevant_chunks), confidence, source_info
            
        except Exception as e:
            logger.exception(f"Error processing query '{question}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    def _calculate_confidence(self, relevant_chunks: List[Tuple[str, float, Dict]], answer: str, verification_results: Dict) -> str:
        """Estimate confidence of the generated answer with proper scoring."""
        if not relevant_chunks:
            return "very low"
    
        # Use verification confidence as primary
        verification_confidence = verification_results.get("confidence", 0.5)
    
        # Fix: Use proper scores (they should be positive now)
        scores = [s for _, s, _ in relevant_chunks if isinstance(s, (int, float)) and s >= 0]
    
        if not scores:
            retrieval_avg = 0.3  # Default low confidence
        else:
            retrieval_avg = sum(scores) / len(scores)
    
        # Combined confidence - prioritize verification
        combined_confidence = (verification_confidence * 0.7) + (retrieval_avg * 0.3)
    
        # Adjust confidence based on answer quality
        if "cannot find" in answer.lower() or "does not provide" in answer.lower():
            combined_confidence *= 0.8  # Penalize for incomplete answers
    
        if combined_confidence > 0.7:
            return "very high"
        elif combined_confidence > 0.55:
            return "high"
        elif combined_confidence > 0.4:
            return "medium"
        elif combined_confidence > 0.25:
            return "low"
        else:
            return "very low"

    def list_documents(self) -> List[DocumentInfo]:
        """List all processed documents."""
        try:
            docs = self.rag_pipeline.get_document_list()
            return [
                DocumentInfo(
                    filename=d.get("title", "unknown"),
                    chunks=d.get("num_chunks", 0),
                    source=d.get("file_path", "unknown"),
                    document_id=d.get("document_id", "unknown")
                )
                for d in docs
            ]
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store."""
        try:
            if hasattr(self.rag_pipeline, "delete_document"):
                return self.rag_pipeline.delete_document(document_id)
            logger.warning("delete_document not implemented in RAGPipeline.")
            return False
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False

    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get production monitoring metrics."""
        return self.monitor.get_metrics()

# Singleton instance
rag_service = RAGService()
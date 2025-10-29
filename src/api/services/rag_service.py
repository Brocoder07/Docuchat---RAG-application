"""
Enhanced RAG service with safe initialization, proper health checks, and source tracking.
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

class RAGService:
    def __init__(self):
        """Initialize RAGService with lazy RAG pipeline creation."""
        try:
            self.rag_pipeline = RAGPipeline()
            self.initialized = False
            self.initialization_error = None
        except Exception as e:
            self.rag_pipeline = None
            self.initialized = False
            self.initialization_error = str(e)
            logger.exception("Failed to create RAGPipeline during service init.")

    def initialize(self):
        """Initialize the underlying RAG pipeline safely."""
        try:
            if self.rag_pipeline:
                self.rag_pipeline.initialize()
                self.initialized = True
                self.initialization_error = None
                logger.info("RAGService initialized successfully.")
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
        """Return overall health/status information."""
        try:
            stats = self.rag_pipeline.get_stats() if self.rag_pipeline else {}
            total_chunks = stats.get("total_chunks") or 0
            docs_processed = stats.get("documents_processed") or 0
            llm_ready = stats.get("llm_ready", False)
            llm_model = stats.get("llm_model", "unknown")

            status = "healthy" if self.is_initialized else "initializing"

            health_info = {
                "status": status,
                "documents_processed": docs_processed,
                "total_chunks": total_chunks,
                "llm_ready": llm_ready,
                "llm_model": llm_model,
                "vector_store_initialized": total_chunks > 0,
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
                "llm_model": "unknown",
                "documents_processed": 0,
                "total_chunks": 0
            }

    async def upload_document(self, file: UploadFile, upload_dir: str) -> UploadResponse:
        """Upload and process a document with duplicate prevention."""
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
        """Query documents with enhanced source information."""
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
            logger.info(f"Query processed successfully, found {len(relevant_chunks)} relevant chunks.")
            confidence = self._calculate_confidence(relevant_chunks, answer)
            return answer, len(relevant_chunks), confidence, source_info
        except Exception as e:
            logger.exception(f"Error processing query '{question}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    def _calculate_confidence(self, relevant_chunks: List[Tuple[str, float, dict]], answer: str) -> str:
        """Estimate confidence of the generated answer."""
        if not relevant_chunks:
            return "very low"
        scores = [s for _, s, _ in relevant_chunks if isinstance(s, (int, float))]
        avg = sum(scores) / len(scores) if scores else 0
        if avg > 0.7:
            return "very high"
        elif avg > 0.55:
            return "high"
        elif avg > 0.4:
            return "medium"
        elif avg > 0.25:
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


# Singleton instance
rag_service = RAGService()

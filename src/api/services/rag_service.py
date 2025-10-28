"""
Enhanced RAG service with better source tracking.
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
    """Enhanced service layer for RAG operations with source tracking."""
    
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.is_initialized = False
        self.initialization_error = None
    
    def initialize(self):
        """Initialize the RAG pipeline."""
        try:
            logger.info("Initializing RAG service...")
            self.rag_pipeline.initialize()
            self.is_initialized = True
            self.initialization_error = None
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            self.initialization_error = str(e)
            raise
    
    async def upload_document(self, file: UploadFile, upload_dir: str) -> UploadResponse:
        """Upload and process a document with duplicate prevention."""
        if not self.is_initialized:
            try:
                self.initialize()
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Service initialization failed: {str(e)}"
                )

        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        if not file_extension:
            raise HTTPException(
                status_code=400,
                detail="File must have an extension"
            )

        if file_extension not in api_config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{file_extension}' not supported. Allowed: {', '.join(api_config.ALLOWED_EXTENSIONS)}"
            )

        # Read file content for duplicate checking
        try:
            contents = await file.read()
            file_size = len(contents)
        
            # Check file size
            if file_size > api_config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {api_config.MAX_FILE_SIZE / (1024*1024):.1f}MB"
                )
        
            # Check if file is empty
            if file_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty"
                )
        
            # Reset file pointer for processing
            await file.seek(0)
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error reading file {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error reading uploaded file: {str(e)}"
            )

        # Check for duplicate content (basic content-based deduplication)
        try:
            existing_documents = self.rag_pipeline.get_document_list()
            for existing_doc in existing_documents:
                if existing_doc.filename == file.filename:
                    logger.warning(f"Duplicate filename detected: {file.filename}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"A file with name '{file.filename}' has already been uploaded. Please upload a different file or rename this file."
                    )
        except Exception as e:
            logger.warning(f"Could not check for duplicates: {str(e)}")

        # Create unique filename
        file_id = str(uuid.uuid4())[:8]
        saved_filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(upload_dir, saved_filename)

        try:
            # Save uploaded file
            with open(file_path, "wb") as f:
                f.write(contents)
    
            logger.info(f"Saved uploaded file to: {file_path}")
    
            # Process document through enhanced RAG pipeline
            success, document_id, processed_filename = self.rag_pipeline.process_document(file_path)
    
            if success:
                logger.info(f"Successfully processed document: {file.filename} (ID: {document_id})")
                return UploadResponse(
                    message="Document uploaded and processed successfully",
                    filename=file.filename,  # Return original filename, not processed one
                    file_id=document_id
                )
            else:
                # Clean up failed upload
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise HTTPException(
                    status_code=500, 
                    detail="Failed to process document. The document may be corrupted, in an unsupported format, or contain no extractable text."
                )
        
        except HTTPException:
            raise
        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up failed upload: {file_path}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up file {file_path}: {str(cleanup_error)}")
    
            logger.error(f"Error processing document {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error while processing document: {str(e)}"
            )
    
    def query_documents(self, question: str, top_k: int = 3) -> Tuple[str, int, str, Dict[str, Any]]:
        """
        Query documents with enhanced source information.
        
        Args:
            question: The question to ask
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Tuple of (answer, relevant_sources, confidence, source_info)
        """
        if not self.is_initialized:
            try:
                self.initialize()
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Service initialization failed: {str(e)}"
                )
        
        # Validate question
        if not question or not question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Clean and validate question
        question = question.strip()
        if len(question) < 2:
            raise HTTPException(
                status_code=400,
                detail="Question must be at least 2 characters long"
            )
        
        # Check if vector store is initialized
        stats = self.rag_pipeline.get_stats()
        if stats["total_chunks"] == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents available. Please upload documents first."
            )
        
        try:
            answer, relevant_chunks, source_info = self.rag_pipeline.query(
                question=question,
                top_k=top_k
            )
        
            logger.info(f"Query processed successfully. Found {len(relevant_chunks)} relevant chunks")

            # Ensure source_info has the required fields
            if not source_info:
                source_info = {
                    "total_sources": 0,
                    "documents": [],
                    "primary_source": "Unknown",
                    "chunk_details": []
                }
        
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(relevant_chunks, answer)
        
            return answer, source_info.get('total_sources', 0), confidence, source_info
        
        except Exception as e:
            logger.error(f"Error processing query '{question}': {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing your question: {str(e)}"
            )
        
    def _calculate_confidence(self, relevant_chunks: List[Tuple[str, float, dict]], answer: str) -> str:
        """Calculate confidence based on multiple factors."""
        if not relevant_chunks:
            return "very low"
    
        # Factor 1: Best similarity score (most important)
        best_score = relevant_chunks[0][1]
    
        # Factor 2: Average similarity score of top chunks
        avg_score = sum(chunk[1] for chunk in relevant_chunks) / len(relevant_chunks)
    
        # Factor 3: Number of relevant chunks found
        chunk_count = len(relevant_chunks)
    
        # Factor 4: Answer quality (length and specificity)
        answer_quality = self._assess_answer_quality(answer)
    
        # Calculate weighted confidence score
        confidence_score = (
            best_score * 0.5 +        # Best score weight: 50%
            avg_score * 0.3 +         # Average score weight: 30%
            (min(chunk_count / 3, 1.0)) * 0.1 +  # Chunk count weight: 10%
            answer_quality * 0.1      # Answer quality weight: 10%
        )
    
        # Convert to confidence level
        if confidence_score > 0.7:
            return "very high"
        elif confidence_score > 0.55:
            return "high"
        elif confidence_score > 0.4:
            return "medium"
        elif confidence_score > 0.25:
            return "low"
        else:
            return "very low"

    def _assess_answer_quality(self, answer: str) -> float:
        """Assess the quality of the generated answer."""
        if not answer or len(answer.strip()) < 20:
            return 0.0
    
        # Check for generic/unhelpful responses
        generic_phrases = [
            "i don't know", "i cannot find", "the context doesn't contain",
            "based on the document", "according to the documents"
        ]
    
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in generic_phrases):
            return 0.3
    
        # Score based on answer length and specificity
        if len(answer) > 100 and " " in answer:  # Reasonable length with spaces
            return 0.8
        elif len(answer) > 50:
            return 0.6
        else:
            return 0.4
    
    def get_health_info(self) -> dict:
        """Get system health information."""
        try:
            stats = self.rag_pipeline.get_stats()
            docs_loaded = stats["total_chunks"]
        
            status = "healthy" if self.is_initialized else "unhealthy"
        
            # Show actual model being used
            if hasattr(self.rag_pipeline.llm_integration, 'model_name'):
                model = self.rag_pipeline.llm_integration.model_name
            else:
                model = "ollama-llama3.2:1b"
        
            health_info = {
                "status": status,
                "model": model,
                "documents_loaded": docs_loaded,
                "vector_store_initialized": docs_loaded > 0,
                "collection_name": stats.get("collection_name", "unknown"),
            }
        
            if self.initialization_error:
                health_info["initialization_error"] = self.initialization_error
                health_info["status"] = "error"
        
            return health_info
        
        except Exception as e:
            logger.error(f"Error getting health info: {str(e)}")
            return {
                "status": "error",
                "model": "unknown",
                "documents_loaded": 0,
                "error": str(e)
            }
    
    def list_documents(self) -> List[DocumentInfo]:
        """List all processed documents."""
        try:
            documents = self.rag_pipeline.get_document_list()
            document_list = []
            
            for doc in documents:
                document_list.append(DocumentInfo(
                    filename=doc.get('filename', 'unknown'),
                    chunks=doc.get('chunk_count', 0),
                    source=doc.get('source', 'unknown'),
                    document_id=doc.get('document_id', 'unknown')
                ))
            
            logger.info(f"Listed {len(document_list)} processed documents")
            return document_list
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store."""
        try:
            success = self.rag_pipeline.delete_document(document_id)
            if success:
                logger.info(f"Successfully deleted document: {document_id}")
            return success
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False

# Global service instance
rag_service = RAGService()
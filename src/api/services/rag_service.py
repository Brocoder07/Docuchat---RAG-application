"""
RAG service handling business logic.
"""
import os
import uuid
import logging
from typing import List, Optional, Tuple
from fastapi import UploadFile, HTTPException

from src.rag_pipeline import RAGPipeline
from src.api.models.schemas import UploadResponse, DocumentInfo
from src.api.core.config import api_config

logger = logging.getLogger(__name__)

class RAGService:
    """Service layer for RAG operations."""
    
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.is_initialized = False
        self.initialization_error = None
        self.processed_file_hashes = set()  # Track processed files to prevent duplicates
    
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
        """
        Upload and process a document.
        
        Args:
            file: Uploaded file
            upload_dir: Directory to save files
            
        Returns:
            Upload response
        """
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
        
        # Read and validate file content first
        try:
            contents = await file.read()
            
            # Check file size
            if len(contents) > api_config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {api_config.MAX_FILE_SIZE / (1024*1024):.1f}MB"
                )
            
            # Check if file is empty
            if len(contents) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty"
                )
            
            # Check for duplicate files based on content hash
            file_hash = hash(contents)
            if file_hash in self.processed_file_hashes:
                logger.info(f"Duplicate file detected: {file.filename}")
                raise HTTPException(
                    status_code=400,
                    detail="This file has already been processed. Please upload a different file."
                )
            
            # Reset file pointer for potential re-reading
            await file.seek(0)
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error reading file {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error reading uploaded file: {str(e)}"
            )
        
        # Create unique filename
        file_id = str(uuid.uuid4())[:8]
        saved_filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(upload_dir, saved_filename)
        
        try:
            # Save uploaded file
            with open(file_path, "wb") as f:
                f.write(contents)
            
            logger.info(f"Saved uploaded file to: {file_path}")
            
            # Process document through RAG pipeline
            success = self.rag_pipeline.process_document(file_path)
            
            if success:
                # Track processed file to prevent duplicates
                self.processed_file_hashes.add(file_hash)
                
                logger.info(f"Successfully processed document: {file.filename}")
                return UploadResponse(
                    message="Document uploaded and processed successfully",
                    filename=file.filename,
                    file_id=file_id
                )
            else:
                # Clean up failed upload
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise HTTPException(
                    status_code=500, 
                    detail="Failed to process document. The document may be corrupted or in an unsupported format."
                )
                
        except HTTPException:
            # Re-raise HTTP exceptions
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
    
    def query_documents(self, question: str, top_k: int = 3) -> Tuple[str, int, str]:
        """
        Query documents with a question.
        
        Args:
            question: The question to ask
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Tuple of (answer, relevant_sources, confidence)
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
        
        if self.rag_pipeline.embedding_manager.index is None:
            raise HTTPException(
                status_code=400,
                detail="No documents available. Please upload documents first."
            )
        
        try:
            logger.info(f"Processing query: '{question}' with top_k={top_k}")
            
            # Validate top_k parameter
            if top_k < 1 or top_k > 20:
                raise HTTPException(
                    status_code=400,
                    detail="top_k must be between 1 and 20"
                )
            
            answer, relevant_chunks = self.rag_pipeline.query(
                question=question,
                top_k=top_k
            )
            
            logger.info(f"Query processed successfully. Found {len(relevant_chunks)} relevant chunks")
            
            # Calculate confidence based on similarity scores
            confidence = "low"
            if relevant_chunks:
                best_score = relevant_chunks[0][1] if relevant_chunks else 0
                if best_score > 0.7:
                    confidence = "high"
                elif best_score > 0.4:
                    confidence = "medium"
                
                logger.info(f"Best similarity score: {best_score:.4f}, Confidence: {confidence}")
            else:
                logger.info("No relevant chunks found for query")
            
            return answer, len(relevant_chunks), confidence
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error processing query '{question}': {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing your question: {str(e)}"
            )
    
    def get_health_info(self) -> dict:
        """Get system health information."""
        try:
            docs_loaded = 0
            if (self.rag_pipeline.embedding_manager.index is not None and 
                hasattr(self.rag_pipeline.embedding_manager, 'chunks')):
                docs_loaded = len(self.rag_pipeline.embedding_manager.chunks)
            
            status = "healthy" if self.is_initialized else "unhealthy"
            model = "smart_rule_based"
            
            health_info = {
                "status": status,
                "model": model,
                "documents_loaded": docs_loaded,
                "vector_store_initialized": self.rag_pipeline.embedding_manager.index is not None,
                "processed_files_count": len(self.processed_file_hashes)
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
            if (not self.rag_pipeline.embedding_manager.metadata or 
                not hasattr(self.rag_pipeline.embedding_manager, 'metadata')):
                return []
            
            # Extract unique documents from metadata
            documents = {}
            for meta in self.rag_pipeline.embedding_manager.metadata:
                source = meta.get("source", "unknown")
                if source not in documents:
                    documents[source] = {
                        "filename": os.path.basename(source),
                        "chunks": 0,
                        "source": source
                    }
                documents[source]["chunks"] += 1
            
            document_list = [DocumentInfo(**doc) for doc in documents.values()]
            logger.info(f"Listed {len(document_list)} processed documents")
            return document_list
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def clear_processed_files(self):
        """Clear the processed files cache (useful for testing)."""
        self.processed_file_hashes.clear()
        logger.info("Cleared processed files cache")

# Global service instance
rag_service = RAGService()
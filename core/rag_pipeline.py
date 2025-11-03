"""
Main RAG pipeline orchestrator - FIXED with all required methods.
"""
import logging
import uuid
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from core.config import config
from core.llm_service import llm_service
from core.vector_store import vector_store
from core.document_processor import document_processor

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Production RAG pipeline with monitoring and error handling."""
    
    def __init__(self):
        self.initialized = False
        self.processed_documents = []
        self.query_history = []
        
        # 🚨 FIX: Initialize components immediately
        from core.vector_store import vector_store
        from core.llm_service import llm_service
        
        self.vector_store = vector_store
        self.llm_service = llm_service
    
    def initialize(self) -> bool:
        """Initialize all components with proper error handling."""
        try:
            logger.info("🚀 Initializing RAG Pipeline...")
            
            # Initialize vector store first (foundation)
            if not self.vector_store.initialize():
                logger.error("❌ Failed to initialize vector store")
                return False
            
            # Initialize LLM service (core intelligence)
            if not self.llm_service.initialize():
                logger.error("❌ Failed to initialize LLM service")
                return False
            
            self.initialized = True
            logger.info("✅ RAG Pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG Pipeline initialization failed: {str(e)}")
            self.initialized = False
            return False
    
    def _build_empty_source_info(self) -> Dict[str, Any]:
        """Build empty source information."""
        return {
            "total_sources": 0,
            "documents": [],
            "primary_source": "None",
            "chunk_details": [],
            "retrieved_count": 0
        }
    
    def _build_context(self, relevant_chunks: List[Tuple]) -> str:
        """Build clean context from relevant chunks."""
        context_parts = []
        
        for i, (chunk_text, score, metadata) in enumerate(relevant_chunks):
            source = metadata.get('source', 'Unknown')
            context_parts.append(f"[Source: {source} | Relevance: {score:.2f}]")
            context_parts.append(chunk_text)
            context_parts.append("")  # Empty line between chunks
        
        return "\n".join(context_parts)
    
    def _prepare_source_info(self, relevant_chunks: List[Tuple]) -> Dict[str, Any]:
        """Prepare structured source information."""
        if not relevant_chunks:
            return self._build_empty_source_info()
        
        # Extract unique documents
        documents = set()
        chunk_details = []
        
        for i, (chunk_text, score, metadata) in enumerate(relevant_chunks):
            doc_name = metadata.get('source', 'Unknown Document')
            documents.add(doc_name)
            
            chunk_details.append({
                'document': doc_name,
                'content_preview': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                'confidence': float(score),
                'chunk_id': metadata.get('chunk_id', f'chunk_{i+1}')
            })
        
        # Determine primary source (document with highest total confidence)
        source_scores = {}
        for chunk in relevant_chunks:
            doc_name = chunk[2].get('source', 'Unknown Document')
            score = chunk[1]
            source_scores[doc_name] = source_scores.get(doc_name, 0) + score
        
        primary_source = max(source_scores.items(), key=lambda x: x[1])[0] if source_scores else "Unknown"
        
        return {
            "total_sources": len(documents),
            "documents": list(documents),
            "primary_source": primary_source,
            "chunk_details": chunk_details,
            "retrieved_count": len(relevant_chunks)
        }

    # REST OF YOUR EXISTING METHODS (process_document, query, etc.) REMAIN THE SAME
    # Just make sure they use self.vector_store and self.llm_service instead of direct imports
    
    def process_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Process document end-to-end with progress tracking.
        Senior Engineer Principle: Clear input/output contracts.
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Pipeline not initialized",
                "document_id": None
            }
        
        try:
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            logger.info(f"📦 Processing document: {filename} (ID: {document_id})")
            
            # Step 1: Extract and chunk text
            processing_result = document_processor.process_file(file_path, filename)
            if not processing_result["success"]:
                return {
                    "success": False,
                    "error": processing_result["error"],
                    "document_id": document_id
                }
            
            chunks = processing_result["chunks"]
            if not chunks:
                return {
                    "success": False,
                    "error": "No chunks generated from document",
                    "document_id": document_id
                }
            
            # Step 2: Prepare metadata for vector store
            metadata_list = []
            for i, chunk in enumerate(chunks):
                metadata_list.append({
                    "source": filename,
                    "filename": filename,
                    "document_id": document_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processed_at": datetime.now().isoformat()
                })
            
            # Step 3: Store in vector database
            if not vector_store.add_documents(chunks, metadata_list, document_id):
                return {
                    "success": False,
                    "error": "Failed to store document in vector database",
                    "document_id": document_id
                }
            
            # Step 4: Track processed document
            document_info = {
                "document_id": document_id,
                "filename": filename,
                "file_path": file_path,
                "chunks_count": len(chunks),
                "processed_at": datetime.now().isoformat()
            }
            self.processed_documents.append(document_info)
            
            logger.info(f"✅ Document processed successfully: {filename} ({len(chunks)} chunks)")
            
            return {
                "success": True,
                "document_id": document_id,
                "chunks_count": len(chunks),
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"Processing error: {str(e)}",
                "document_id": None
            }
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Query documents with proper context management.
        FIXED: Always return source_info even when no chunks found
        """
        if not self.initialized:
            return {
                "success": False,
                "answer": "Pipeline not initialized. Please check system status.",
                "sources": [],
                "source_info": self._build_empty_source_info(),  # 🚨 ADD THIS
                "error": "Pipeline not initialized"
            }
    
        try:
            logger.info(f"🔍 Processing query: {question}")
        
            # Step 1: Retrieve relevant chunks
            relevant_chunks = vector_store.search(question, top_k=top_k)
        
            if not relevant_chunks:
                logger.info(f"❌ No relevant content found for: {question}")
                return {
                    "success": True,
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": [],
                    "source_info": self._build_empty_source_info(),  # 🚨 ADD THIS
                    "confidence": "very low",
                    "chunks_retrieved": 0,
                    "model_used": "none"
                }
        
            # Step 2: Build context from relevant chunks
            context = self._build_context(relevant_chunks)
        
            # Step 3: Generate answer using LLM
            llm_result = llm_service.generate_answer(question, context)
        
            if not llm_result["success"]:
                return {
                    "success": False,
                    "answer": "I encountered an error while generating the answer.",
                    "sources": [],
                    "source_info": self._build_empty_source_info(),  # 🚨 ADD THIS
                    "error": llm_result.get("error", "LLM error")
                }
        
            # Step 4: Prepare source information
            source_info = self._prepare_source_info(relevant_chunks)
        
            # Step 5: Calculate confidence
            confidence = self._calculate_confidence(relevant_chunks, llm_result["answer"])
        
            # Step 6: Track query for analytics
            self._track_query(question, llm_result["answer"], len(relevant_chunks))
        
            logger.info(f"✅ Query processed successfully: {len(relevant_chunks)} chunks used")
        
            return {
                "success": True,
                "answer": llm_result["answer"],
                "sources": relevant_chunks,
                "source_info": source_info,
                "confidence": confidence,
                "chunks_retrieved": len(relevant_chunks),
                "model_used": llm_result.get("model", "unknown")
            }
        
        except Exception as e:
            logger.error(f"❌ Query processing failed: {str(e)}")
            return {
                "success": False,
                "answer": "I encountered an error while processing your query.",
                "sources": [],
                "source_info": self._build_empty_source_info(),  # 🚨 ADD THIS
                "error": str(e)
            }
    
    def _build_context(self, relevant_chunks: List[Tuple]) -> str:
        """Build clean context from relevant chunks."""
        context_parts = []
        
        for i, (chunk_text, score, metadata) in enumerate(relevant_chunks):
            source = metadata.get('source', 'Unknown')
            context_parts.append(f"[Source: {source} | Relevance: {score:.2f}]")
            context_parts.append(chunk_text)
            context_parts.append("")  # Empty line between chunks
        
        return "\n".join(context_parts)
    
    def _prepare_source_info(self, relevant_chunks: List[Tuple]) -> Dict[str, Any]:
        """Prepare structured source information."""
        if not relevant_chunks:
            return {
                "total_sources": 0,
                "documents": [],
                "primary_source": "None",
                "chunk_details": []
            }
        
        # Extract unique documents
        documents = set()
        chunk_details = []
        
        for i, (chunk_text, score, metadata) in enumerate(relevant_chunks):
            doc_name = metadata.get('source', 'Unknown Document')
            documents.add(doc_name)
            
            chunk_details.append({
                'document': doc_name,
                'content_preview': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                'confidence': float(score),
                'chunk_id': metadata.get('chunk_id', f'chunk_{i+1}')
            })
        
        # Determine primary source (document with highest total confidence)
        source_scores = {}
        for chunk in relevant_chunks:
            doc_name = chunk[2].get('source', 'Unknown Document')
            score = chunk[1]
            source_scores[doc_name] = source_scores.get(doc_name, 0) + score
        
        primary_source = max(source_scores.items(), key=lambda x: x[1])[0] if source_scores else "Unknown"
        
        return {
            "total_sources": len(documents),
            "documents": list(documents),
            "primary_source": primary_source,
            "chunk_details": chunk_details,
            "retrieved_count": len(relevant_chunks)
        }
    
    def _calculate_confidence(self, relevant_chunks: List[Tuple], answer: str) -> str:
        """Calculate answer confidence based on multiple factors."""
        if not relevant_chunks:
            return "very low"
        
        # Factor 1: Average similarity score of retrieved chunks
        avg_similarity = sum(score for _, score, _ in relevant_chunks) / len(relevant_chunks)
        
        # Factor 2: Answer length (very short answers might be low confidence)
        answer_length_factor = min(1.0, len(answer) / 100)
        
        # Factor 3: Presence of uncertainty phrases
        uncertainty_phrases = ['cannot find', 'not provided', 'unable to', 'no information']
        uncertainty_penalty = 0.0
        for phrase in uncertainty_phrases:
            if phrase in answer.lower():
                uncertainty_penalty = 0.3
                break
        
        # Combined confidence score
        combined_confidence = (avg_similarity * 0.7) + (answer_length_factor * 0.3) - uncertainty_penalty
        
        # Map to confidence levels
        if combined_confidence > 0.8:
            return "very high"
        elif combined_confidence > 0.65:
            return "high"
        elif combined_confidence > 0.5:
            return "medium"
        elif combined_confidence > 0.3:
            return "low"
        else:
            return "very low"
    
    def _track_query(self, question: str, answer: str, chunks_used: int):
        """Track query for analytics and monitoring with size limits."""
        query_record = {
            "question": question,
            "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,
            "chunks_used": chunks_used,
            "timestamp": datetime.now().isoformat()
        }
        self.query_history.append(query_record)
        
        # 🚨 FIX: Keep only last 50 queries to prevent memory issues
        if len(self.query_history) > 50:
            self.query_history = self.query_history[-50:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        vector_stats = vector_store.get_collection_stats()
        llm_status = llm_service.get_status()
        
        return {
            "initialized": self.initialized,
            "documents_processed": len(self.processed_documents),
            "total_queries": len(self.query_history),
            "vector_store": vector_stats,
            "llm_service": llm_status,
            "timestamp": datetime.now().isoformat()
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents."""
        return self.processed_documents.copy()
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the pipeline."""
        try:
            # Remove from vector store
            success = vector_store.delete_document(document_id)
            
            if success:
                # Remove from processed documents
                self.processed_documents = [
                    doc for doc in self.processed_documents 
                    if doc.get('document_id') != document_id
                ]
                logger.info(f"✅ Document {document_id} deleted successfully")
            else:
                logger.warning(f"Document {document_id} not found in vector store")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False

    
# Global pipeline instance
rag_pipeline = RAGPipeline()
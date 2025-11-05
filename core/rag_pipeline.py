"""
Main RAG pipeline orchestrator - Upgraded with Security (Firebase UID as str).
FIXED: get_status user_id is now Optional.
"""
import logging
import uuid
import os
import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from core.config import config
from core.llm_service import llm_service
from core.vector_store import vector_store
from core.document_processor import document_processor

logger = logging.getLogger(__name__)

class RAGPipeline:
    
    def __init__(self):
        # ... (this method is unchanged) ...
        self.initialized = False
        self.processed_documents = [] 
        self.query_history = []
        
        from core.vector_store import vector_store
        from core.llm_service import llm_service
        
        self.vector_store = vector_store
        self.llm_service = llm_service
    
    def initialize(self) -> bool:
        # ... (this method is unchanged) ...
        try:
            logger.info("🚀 Initializing RAG Pipeline...")
            
            if not self.vector_store.initialize():
                logger.error("❌ Failed to initialize vector store")
                return False
            
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
    
    def _clean_query_for_routing(self, question: str) -> str:
        # ... (this method is unchanged) ...
        constraint_patterns = [
            r"in \d+-\d+ words",
            r"in \d+ words",
            r"list \d+ items",
            r"list top \d+",
            r"in \d+ sentences",
        ]
        
        cleaned_question = question
        for pattern in constraint_patterns:
            cleaned_question = re.sub(pattern, "", cleaned_question, flags=re.IGNORECASE).strip()
        
        cleaned_question = re.sub(r"\b(in|list)$", "", cleaned_question, flags=re.IGNORECASE).strip()
        
        if cleaned_question != question:
            logger.info(f"Original query for routing: '{question}' -> Cleaned: '{cleaned_question}'")
        
        return cleaned_question
    
    def _build_empty_source_info(self) -> Dict[str, Any]:
        # ... (this method is unchanged) ...
        return {
            "total_sources": 0,
            "documents": [],
            "primary_source": "None",
            "chunk_details": [],
            "retrieved_count": 0
        }
    
    def _build_context(self, relevant_chunks: List[Tuple]) -> str:
        # ... (this method is unchanged) ...
        context_parts = []
        
        for i, (chunk_text, score, metadata) in enumerate(relevant_chunks):
            source = metadata.get('source', 'Unknown')
            context_parts.append(f"[Source: {source} | Relevance: {score:.2f}]")
            context_parts.append(chunk_text)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _prepare_source_info(self, relevant_chunks: List[Tuple]) -> Dict[str, Any]:
        # ... (this method is unchanged) ...
        if not relevant_chunks:
            return {
                "total_sources": 0,
                "documents": [],
                "primary_source": "None",
                "chunk_details": []
            }
        
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

    
    def process_document(self, file_path: str, filename: str, user_id: str) -> Dict[str, Any]:
        # ... (this method is unchanged) ...
        if not self.initialized:
            return {"success": False, "error": "Pipeline not initialized", "document_id": None}
        
        try:
            document_id = str(uuid.uuid4())
            logger.info(f"📦 Processing document: {filename} (ID: {document_id}) for user {user_id}")
            
            processing_result = document_processor.process_file(file_path, filename)
            if not processing_result["success"]:
                return {**processing_result, "document_id": document_id}
            
            chunks = processing_result["chunks"]
            if not chunks:
                return {"success": False, "error": "No chunks generated from document", "document_id": document_id}
            
            metadata_list = []
            for i, chunk in enumerate(chunks):
                metadata_list.append({
                    "source": filename,
                    "filename": filename,
                    "document_id": document_id,
                    "user_id": user_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processed_at": datetime.now().isoformat()
                })
            
            if not self.vector_store.add_documents(chunks, metadata_list, document_id):
                return {"success": False, "error": "Failed to store document in vector database", "document_id": document_id}
            
            logger.info(f"✅ Document processed successfully: {filename} ({len(chunks)} chunks) for user {user_id}")
            
            return {
                "success": True,
                "document_id": document_id,
                "chunks_count": len(chunks),
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            return {"success": False, "error": f"Processing error: {str(e)}", "document_id": None}
    
    def query(self, question: str, user_id: str, top_k: Optional[int] = None, filename: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        # ... (this method is unchanged) ...
        if not self.initialized:
            return {"success": False, "answer": "Pipeline not initialized...", "sources": [], "source_info": self._build_empty_source_info(), "error": "Pipeline not initialized"}
    
        try:
            logger.info(f"🔍 Processing query for user {user_id}: {question}")
            
            cleaned_question = self._clean_query_for_routing(question)
            query_type = llm_service.route_query(cleaned_question)
            
            if query_type == "general":
                logger.info("🔄 Query is 'general'. Rewriting with HyDE...")
                hyde_result = llm_service.generate_hypothetical_query(cleaned_question)
                search_query = hyde_result["query"]
                
                if not hyde_result["success"]:
                    logger.warning(f"⚠️ HyDE failed, falling back... Error: {hyde_result['error']}")
                    search_query = cleaned_question
            else:
                logger.info("✅ Query is 'specific'. Using original query for search.")
                search_query = question
            
            relevant_chunks = self.vector_store.search(
                search_query, 
                user_id=user_id,
                top_k=top_k, 
                filename=filename
            )
        
            if not relevant_chunks:
                logger.info(f"❌ No relevant content found for user {user_id}: {question}")
                return {"success": True, "answer": "I couldn't find any relevant information...", "sources": [], "source_info": self._build_empty_source_info(), "confidence": "very low", "chunks_retrieved": 0, "model_used": "none"}
        
            context = self._build_context(relevant_chunks)
            llm_result = llm_service.generate_answer(question, context, chat_history)
        
            if not llm_result["success"]:
                return {"success": False, "answer": "I encountered an error...", "sources": [], "source_info": self._build_empty_source_info(), "error": llm_result.get("error", "LLM error")}
        
            source_info = self._prepare_source_info(relevant_chunks)
            confidence = self._calculate_confidence(relevant_chunks, llm_result["answer"])
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
            return {"success": False, "answer": "I encountered an error...", "sources": [], "source_info": self._build_empty_source_info(), "error": str(e)}

    
    def _calculate_confidence(self, relevant_chunks: List[Tuple], answer: str) -> str:
        # ... (this method is unchanged) ...
        if not relevant_chunks:
            return "very low"
        
        avg_similarity = sum(score for _, score, _ in relevant_chunks) / len(relevant_chunks)
        answer_length_factor = min(1.0, len(answer) / 100)
        
        uncertainty_phrases = ['cannot find', 'not provided', 'unable to', 'no information']
        uncertainty_penalty = 0.0
        for phrase in uncertainty_phrases:
            if phrase in answer.lower():
                uncertainty_penalty = 0.3
                break
        
        combined_confidence = (avg_similarity * 0.7) + (answer_length_factor * 0.3) - uncertainty_penalty
        
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
        # ... (this method is unchanged) ...
        query_record = {
            "question": question,
            "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,
            "chunks_used": chunks_used,
            "timestamp": datetime.now().isoformat()
        }
        self.query_history.append(query_record)
        
        if len(self.query_history) > 50:
            self.query_history = self.query_history[-50:]
    
    # -----------------------------------------------------------------
    # 🚨 MODIFIED: `get_status` now accepts `Optional[str]`
    # -----------------------------------------------------------------
    def get_status(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive pipeline status (global or user-specific)."""
        
        llm_status = self.llm_service.get_status()
        
        # Get stats (user-specific or global)
        vector_stats = self.vector_store.get_collection_stats(user_id=user_id)
        
        docs_processed = 0
        if user_id:
            user_docs = self.list_documents(user_id=user_id)
            docs_processed = len(user_docs)
        elif vector_stats:
             # For public health check, report global doc count
            docs_processed = vector_stats.get("unique_documents", 0)

        return {
            "initialized": self.initialized,
            "documents_processed": docs_processed,
            "total_queries": len(self.query_history), # (This is global)
            "vector_store": vector_stats,
            "llm_service": llm_status,
            "timestamp": datetime.now().isoformat()
        }
    
    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        # ... (this method is unchanged) ...
        return self.vector_store.list_documents_by_user(user_id=user_id)
    
    def delete_document(self, document_id: str, user_id: str) -> bool:
        # ... (this method is unchanged) ...
        try:
            doc_metadata = self.vector_store.get_document_metadata(document_id, user_id)
            if not doc_metadata:
                logger.warning(f"User {user_id} tried to delete non-existent or un-owned doc {document_id}")
                return False
            
            success = self.vector_store.delete_document(document_id, user_id)
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False

# Global pipeline instance
rag_pipeline = RAGPipeline()
"""
Main RAG pipeline orchestrator - Upgraded with Security (Firebase UID as str).
FIXED: get_status user_id is now Optional and context builder robustified.
ADDED: Granular performance tracking for retrieval, routing, and generation latency.
ADDED: Special logic path for "summary" queries to retrieve full context.
ADDED: file_hash to process_document to store as metadata for duplicate checking.
"""
import logging
import uuid
import os
import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import time

from core.config import config
from core.llm_service import llm_service
from core.vector_store import vector_store
from core.document_processor import document_processor
from core.evaluator import evaluator

logger = logging.getLogger(__name__)

class RAGPipeline:
    
    def __init__(self):
        # ... (no changes)
        self.initialized = False
        self.processed_documents = [] 
        self.query_history = []
        
        self.vector_store = vector_store
        self.llm_service = llm_service
    
    def initialize(self) -> bool:
        # ... (no changes)
        try:
            logger.info("🚀 Initializing RAG Pipeline...")
            
            if not self.vector_store.initialize():
                logger.error("❌ Failed to initialize vector store")
                return False
            
            if not self.llm_service.initialize():
                logger.error("❌ Failed to initialize LLM service")
                return False
            
            if not evaluator.grounding_model:
                 logger.warning("❌ RAG evaluator grounding model not initialized. Evaluation may be limited.")

            self.initialized = True
            logger.info("✅ RAG Pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG Pipeline initialization failed: {str(e)}")
            self.initialized = False
            return False
    
    # ... (no changes to helper methods _clean_query_for_routing, _build_empty_source_info, _build_context, _prepare_source_info) ...
    def _clean_query_for_routing(self, question: str) -> str:
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
        return {
            "total_sources": 0,
            "documents": [],
            "primary_source": "None",
            "chunk_details": [],
            "retrieved_count": 0
        }
    
    def _build_context(self, relevant_chunks: List[Tuple]) -> str:
        context_parts = []
        
        for i, (chunk_text, score, metadata) in enumerate(relevant_chunks):
            chunk_text_safe = chunk_text if isinstance(chunk_text, str) else (str(chunk_text) if chunk_text is not None else "")
            source = metadata.get('source', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
            if score < 99.0: 
                context_parts.append(f"[Source: {source} | Relevance: {score:.2f}]")
            else:
                context_parts.append(f"[Source: {source}]")
            context_parts.append(chunk_text_safe)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _prepare_source_info(self, relevant_chunks: List[Tuple]) -> Dict[str, Any]:
        if not relevant_chunks:
            return {
                "total_sources": 0,
                "documents": [],
                "primary_source": "None",
                "chunk_details": [],
                "retrieved_count": 0
            }
        
        documents = set()
        chunk_details = []
        
        for i, (chunk_text, score, metadata) in enumerate(relevant_chunks):
            doc_name = metadata.get('source', 'Unknown Document') if isinstance(metadata, dict) else 'Unknown Document'
            documents.add(doc_name)
            
            chunk_str = chunk_text if isinstance(chunk_text, str) else (str(chunk_text) if chunk_text is not None else "")
            chunk_details.append({
                'document': doc_name,
                'content_preview': chunk_str[:200] + "..." if len(chunk_str) > 200 else chunk_str,
                'confidence': float(score),
                'chunk_id': metadata.get('chunk_id', f'chunk_{i+1}') if isinstance(metadata, dict) else f'chunk_{i+1}'
            })
        
        source_scores = {}
        for chunk in relevant_chunks:
            meta = chunk[2] if isinstance(chunk[2], dict) else {}
            doc_name = meta.get('source', 'Unknown Document')
            score = chunk[1]
            source_scores[doc_name] = source_scores.get(doc_name, 0.0) + float(score)
        
        primary_source = max(source_scores.items(), key=lambda x: x[1])[0] if source_scores else "Unknown"
        
        return {
            "total_sources": len(documents),
            "documents": list(documents),
            "primary_source": primary_source,
            "chunk_details": chunk_details,
            "retrieved_count": len(relevant_chunks)
        }

    # -----------------------------------------------------------------
    # 🚨 START: MODIFIED process_document
    # -----------------------------------------------------------------
    def process_document(
        self, 
        file_path: str, 
        filename: str, 
        user_id: str,
        file_hash: str,      # 🚨 Added
        document_id: str   # 🚨 Added
    ) -> Dict[str, Any]:
    # -----------------------------------------------------------------
    # 🚨 END: MODIFIED process_document
    # -----------------------------------------------------------------
        if not self.initialized:
            return {"success": False, "error": "Pipeline not initialized", "document_id": None}
        
        try:
            # document_id is now passed in from the route
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
                    "file_hash": file_hash,  # 🚨 Store the hash in metadata
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
            logger.error(f"❌ Document processing failed: {str(e)}", exc_info=True)
            return {"success": False, "error": f"Processing error: {str(e)}", "document_id": None}
    
    # -----------------------------------------------------------------
    # 🚨 START: NEW METHOD for duplicate check
    # -----------------------------------------------------------------
    def check_document_exists(self, file_hash: str, user_id: str) -> bool:
        """
        Checks if a document with the same hash already exists for this user.
        """
        if not self.initialized:
            logger.warning("check_document_exists called before pipeline initialized")
            return False
        
        try:
            return self.vector_store.check_hash_exists(file_hash, user_id)
        except Exception as e:
            logger.error(f"Error checking for document hash: {e}")
            # Fail-safe: allow upload to proceed if check fails
            return False
    # -----------------------------------------------------------------
    # 🚨 END: NEW METHOD
    # -----------------------------------------------------------------

    def query(self, question: str, user_id: str, top_k: Optional[int] = None, filename: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        # ... (no changes in query method)
        if not self.initialized:
            return {"success": False, "answer": "Pipeline not initialized...", "sources": [], "source_info": self._build_empty_source_info(), "error": "Pipeline not initialized"}
        
        start_pipeline = time.perf_counter()
        timings = {"routing": 0.0, "hyde": 0.0, "retrieval": 0.0, "generation": 0.0, "total": 0.0}
        
        context = ""
        answer = "I couldn't find any relevant information in the document."
        llm_result = {}
        relevant_chunks = []
        
        q_lower = question.lower()
        is_summary_request = ("summary" in q_lower or "summarize" in q_lower or "what is this" in q_lower or "general idea" in q_lower)

        try:
            logger.info(f"🔍 Processing query for user {user_id}: {question}")

            if is_summary_request and filename:
                logger.info(f"🔄 Summary path triggered for file: {filename}")
                start_retrieval = time.perf_counter()
                
                relevant_chunks = self.vector_store.get_all_chunks_for_file(
                    filename=filename,
                    user_id=user_id,
                    limit=50 
                )
                timings["retrieval"] = time.perf_counter() - start_retrieval
                logger.info(f"🔍 Summary retrieval found {len(relevant_chunks)} chunks. (Retrieval: {timings['retrieval']:.2f}s)")
            
            else:
                cleaned_question = self._clean_query_for_routing(question)
                
                start_route = time.perf_counter()
                query_type = llm_service.route_query(cleaned_question)
                timings["routing"] = time.perf_counter() - start_route
                
                if query_type == "general":
                    logger.info(f"🔄 Query is 'general'. Rewriting with HyDE... (Routing: {timings['routing']:.2f}s)")
                    start_hyde = time.perf_counter()
                    hyde_result = llm_service.generate_hypothetical_query(cleaned_question)
                    timings["hyde"] = time.perf_counter() - start_hyde
                    
                    search_query = hyde_result.get("query", cleaned_question)
                    if not hyde_result.get("success", False):
                        logger.warning(f"⚠️ HyDE failed, falling back... Error: {hyde_result.get('error')} (HyDE Time: {timings['hyde']:.2f}s)")
                    else:
                        logger.info(f"✅ HyDE rewrite successful. (HyDE Time: {timings['hyde']:.2f}s)")
                else:
                    logger.info(f"✅ Query is 'specific'. Using original query. (Routing: {timings['routing']:.2f}s)")
                    search_query = question
                
                start_retrieval = time.perf_counter()
                relevant_chunks_unfiltered = self.vector_store.search(
                    search_query, 
                    user_id=user_id,
                    top_k=top_k, 
                    filename=filename
                )
                timings["retrieval"] = time.perf_counter() - start_retrieval
                logger.info(f"🔍 Retrieval found {len(relevant_chunks_unfiltered)} potential chunks. (Retrieval: {timings['retrieval']:.2f}s)")
                
                threshold = config.rag.SIMILARITY_THRESHOLD
                relevant_chunks = [
                    chunk for chunk in relevant_chunks_unfiltered if chunk[1] >= threshold
                ]
                logger.info(f"Retrieved {len(relevant_chunks_unfiltered)} chunks, "
                            f"kept {len(relevant_chunks)} after threshold ({threshold})")

            if not relevant_chunks:
                logger.info(f"❌ No relevant content found *above threshold* for user {user_id}: {question}")
            else:
                context = self._build_context(relevant_chunks)
                
                start_generation = time.perf_counter()
                
                llm_result = llm_service.generate_answer(
                    question, 
                    context, 
                    chat_history,
                    is_summary=is_summary_request
                )
                
                timings["generation"] = time.perf_counter() - start_generation
            
                if not llm_result.get("success", False):
                    answer = "I encountered an error..."
                    logger.error(f"❌ LLM generation failed. (Generation: {timings['generation']:.2f}s)")
                else:
                    answer = llm_result.get("answer", "")
                    logger.info(f"✅ LLM generation successful. (Generation: {timings['generation']:.2f}s)")
        
            source_info = self._prepare_source_info(relevant_chunks)
            confidence = self._calculate_confidence(relevant_chunks, answer)
            self._track_query(question, answer, len(relevant_chunks))
        
            return {
                "success": True,
                "answer": answer,
                "sources": relevant_chunks,
                "source_info": source_info,
                "confidence": confidence,
                "chunks_retrieved": len(relevant_chunks),
                "model_used": llm_result.get("model", "unknown")
            }
        
        except Exception as e:
            logger.error(f"❌ Query processing failed: {str(e)}", exc_info=True)
            return {"success": False, "answer": "I encountered an error...", "sources": [], "source_info": self._build_empty_source_info(), "error": str(e)}
        
        finally:
            timings["total"] = time.perf_counter() - start_pipeline
            logger.info(f"🏁 Query processing finished. (Total Time: {timings['total']:.2f}s)")
            
            evaluator.evaluate_query(
                question=question,
                answer=answer,
                context=context,
                relevant_chunks=relevant_chunks,
                response_time=timings["total"],
                timings=timings
            )
    
    # ... (no changes to _calculate_confidence, _track_query, get_status, list_documents, delete_document) ...
    def _calculate_confidence(self, relevant_chunks: List[Tuple], answer: str) -> str:
        if not relevant_chunks:
            return "very low"
        
        avg_similarity = sum(score for _, score, _ in relevant_chunks) / len(relevant_chunks)
        answer_length_factor = min(1.0, len(answer) / 100)
        
        uncertainty_phrases = ['cannot find', 'not provided', 'unable to', 'no information', "i couldn't find", "i could not find"]
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
        query_record = {
            "question": question,
            "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,
            "chunks_used": chunks_used,
            "timestamp": datetime.now().isoformat()
        }
        self.query_history.append(query_record)
        
        if len(self.query_history) > 50:
            self.query_history = self.query_history[-50:]
    
    def get_status(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive pipeline status (global or user-specific)."""
        
        llm_status = self.llm_service.get_status()
        
        vector_stats = self.vector_store.get_collection_stats(user_id=user_id)
        
        docs_processed = 0
        if user_id:
            user_docs = self.list_documents(user_id=user_id)
            docs_processed = len(user_docs)
        elif vector_stats:
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
        return self.vector_store.list_documents_by_user(user_id=user_id)
    
    def delete_document(self, document_id: str, user_id: str) -> bool:
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
"""
Enhanced RAG pipeline with universal anti-hallucination architecture.
Works across all document types - resumes, reports, articles, research papers, etc.
"""
import os
import logging
import uuid
import re  # ADDED MISSING IMPORT
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.llm_integration import LLMIntegration
from src.config import config
from src.utils.error_handlers import handle_rag_errors, log_execution_time
from src.evaluation.rag_metrics import evaluator

# Import the new universal components
try:
    from src.context.universal_boundary_detector import UniversalBoundaryDetector
    from src.validation.universal_hallucination_detector import UniversalHallucinationDetector
    from src.context.adaptive_context_builder import AdaptiveContextBuilder
except ImportError as e:
    # Fallback to inline definitions if modules aren't available yet
    logger = logging.getLogger(__name__)
    logger.warning(f"Universal components not imported - using fallback implementations: {str(e)}")
    
    # Simple fallback implementations
    class UniversalBoundaryDetector:
        def detect_content_units(self, text): 
            return [{'content': text}]
        def extract_entity_boundaries(self, text): 
            return {}

    class UniversalHallucinationDetector:
        def detect_hallucinations(self, answer, context_chunks, question=None):
            return {
                'is_hallucinating': False, 
                'overall_hallucination_score': 0.0,
                'speculation': {'score': 0.0},
                'context_mismatch': {'score': 0.0},
                'entity_invention': {'score': 0.0},
                'factual_contradiction': {'score': 0.0},
                'unsupported_claims': {'score': 0.0}
            }

    class AdaptiveContextBuilder:
        def build_universal_context(self, chunks, question):
            context_parts = []
            for i, chunk_data in enumerate(chunks[:5]):
                if isinstance(chunk_data, tuple) and len(chunk_data) >= 2:
                    chunk, score = chunk_data[0], chunk_data[1]
                    context_parts.append(f"--- CONTENT UNIT {i+1} (relevance: {score:.2f}) ---")
                    context_parts.append(chunk)
                    context_parts.append("")
            
            context = "\n".join(context_parts)
            context += """
            
CRITICAL INSTRUCTIONS:
1. Use ONLY the information provided in the context above
2. Do not add, infer, or assume any information not explicitly stated
3. If information is missing, say 'The context does not provide information about X'
4. Keep different content units separate - do not mix information across boundaries
5. Be precise and factual - avoid speculative language
6. If unsure, acknowledge the limitation rather than guessing"""
            
            return context

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline with universal anti-hallucination architecture."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.llm_integration = LLMIntegration()
        
        # Universal anti-hallucination components
        self.boundary_detector = UniversalBoundaryDetector()
        self.hallucination_detector = UniversalHallucinationDetector()
        self.context_builder = AdaptiveContextBuilder()
        
        self.is_initialized = False
        self.performance_stats = {
            "total_queries": 0,
            "avg_retrieval_time": 0.0,
            "avg_llm_time": 0.0,
            "successful_queries": 0,
            "hallucination_rejections": 0
        }
    
    @handle_rag_errors
    def initialize(self):
        """Initialize the pipeline components."""
        try:
            self.embedding_manager.initialize_model()
            self.llm_integration.initialize(use_openai=False)
            self.is_initialized = True
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise
    
    @handle_rag_errors
    @log_execution_time
    def process_document(self, file_path: str) -> Tuple[bool, str, str]:
        """
        Process a document and add to the vector store.
        
        Returns:
            Tuple of (success, document_id, filename)
        """
        if not self.is_initialized:
            self.initialize()

        try:
            # Load and chunk document
            text = self.document_processor.load_document(file_path)
            if not text:
                logger.error(f"Failed to load document or no text extracted: {file_path}")
                return False, "", "Failed to load document"
        
            if len(text.strip()) < 10:  # Minimum text length
                logger.error(f"Document contains too little text: {file_path}")
                return False, "", "Document contains too little text"
        
            chunks = self.document_processor.chunk_text(text)
            if not chunks:
                logger.error("No chunks created from document")
                return False, "", "No text chunks could be created"
        
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            
            # Generate unique document ID and get filename
            document_id = str(uuid.uuid4())
            filename = os.path.basename(file_path)
        
            # Create enhanced metadata for each chunk
            metadata = [{
                "source": file_path, 
                "chunk_id": i, 
                "filename": filename,
                "document_id": document_id,
                "document_name": filename,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            } for i, chunk in enumerate(chunks)]
        
            # Add to ChromaDB
            self.embedding_manager.add_to_index(chunks, metadata, document_id)
        
            logger.info(f"Added {len(chunks)} chunks from document {document_id}")
            return True, document_id, filename
        
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return False, "", f"Error processing document: {str(e)}"

    @handle_rag_errors
    @log_execution_time
    def query(self, question: str, top_k: int = 3, filter_by_document: Optional[str] = None) -> Tuple[Optional[str], List[Tuple[str, float, dict]], Dict]:
        """
        Universal RAG query that works for any document type with anti-hallucination.
        
        Args:
            question: The question to ask
            top_k: Number of relevant chunks to retrieve
            filter_by_document: Optional document ID to filter results
            
        Returns:
            Tuple of (answer, relevant_chunks, source_info)
        """
        if not self.is_initialized:
            self.initialize()
        
        # Update performance stats
        self.performance_stats["total_queries"] += 1
        
        # Check if we have any documents
        stats = self.embedding_manager.get_stats()
        if stats["total_chunks"] == 0:
            return "No documents have been processed yet. Please upload documents first.", [], {}
        
        try:
            # Build metadata filter if specified
            filter_metadata = None
            if filter_by_document:
                filter_metadata = {"document_id": filter_by_document}
            
            # Retrieve relevant chunks
            relevant_chunks = self.embedding_manager.search(
                question, 
                top_k=top_k, 
                filter_metadata=filter_metadata
            )
            
            # Remove duplicates
            relevant_chunks = self.remove_duplicate_chunks(relevant_chunks)

            if not relevant_chunks:
                return "No relevant information found in the documents.", [], {}
            
            # Build universal context with anti-hallucination structure
            context = self.context_builder.build_universal_context(relevant_chunks, question)
            
            # Generate answer with universal hallucination prevention
            answer = self._generate_universal_answer(question, context, relevant_chunks)
            
            # Extract source information
            source_info = self._extract_source_info(relevant_chunks)
            
            # Update successful queries
            self.performance_stats["successful_queries"] += 1
            
            return answer, relevant_chunks, source_info
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing your question: {str(e)}", [], {}
    
    def _generate_universal_answer(self, question: str, context: str, chunks: List[Tuple[str, float, dict]]) -> str:
        """
        Generate answer with universal hallucination prevention.
        Works across all document types.
        """
        max_attempts = 2
        
        for attempt in range(max_attempts):
            # Generate answer using LLM
            if self.llm_integration.use_ollama:
                answer = self._call_ollama_with_context(question, context)
            else:
                answer = self.llm_integration.generate_answer(question, chunks)
            
            # Universal hallucination validation
            hallucination_check = self.hallucination_detector.detect_hallucinations(
                answer, chunks, question
            )
            
            if not hallucination_check.get('is_hallucinating', False):
                logger.info(f"Answer validated (hallucination score: {hallucination_check.get('overall_hallucination_score', 0):.2f})")
                return answer
            else:
                logger.warning(f"Hallucination detected (score: {hallucination_check.get('overall_hallucination_score', 0):.2f})")
                self.performance_stats["hallucination_rejections"] += 1
                
                if attempt == max_attempts - 1:
                    # Final fallback to conservative extraction
                    return self._conservative_extraction(question, chunks, hallucination_check)
        
        # Ultimate fallback
        return "I cannot provide a sufficiently accurate answer based on the available information."
    
    def _call_ollama_with_context(self, question: str, context: str) -> str:
        """Call Ollama with the structured context."""
        try:
            # Build a better prompt
            prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context.

    CONTEXT:
    {context}

    QUESTION: {question}

    IMPORTANT RULES:
    - Use ONLY information from the context above
    - Do not use external knowledge
    - If the context doesn't contain the answer, say so
    - Be concise and factual

    ANSWER:"""
        
            logger.info(f"📤 Sending prompt to Ollama (length: {len(prompt)})")
        
            # Call Ollama - FIXED: Only pass the prompt, not extra arguments
            if hasattr(self.llm_integration, '_call_ollama_api'):
                logger.info("🔄 Using _call_ollama_api method")
                # FIX: Only pass the prompt, not question and empty list
                result = self.llm_integration._call_ollama_api(prompt)
                logger.info(f"📥 Received response: {result[:100] if result else 'None'}")
                return result
            elif hasattr(self.llm_integration, 'generate_answer'):
                logger.info("🔄 Using generate_answer method")
                # Create dummy chunks since we're using context directly
                dummy_chunks = [(context, 1.0, {})]
                result = self.llm_integration.generate_answer(question, dummy_chunks)
                logger.info(f"📥 Received response: {result[:100] if result else 'None'}")
                return result
            else:
                logger.error("🚨 No available LLM method found")
                return ""
            
        except Exception as e:
            logger.error(f"❌ Error calling Ollama with context: {str(e)}")
            import traceback
            logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            return ""
    
    def _conservative_extraction(self, question: str, chunks: List[Tuple[str, float, dict]],
                               hallucination_check: Dict) -> str:
        """Conservative answer extraction when hallucination is detected."""
        
        if not chunks:
            return "I cannot provide an answer based on the available information."
        
        # Use the most relevant chunk for safe extraction
        best_chunk_text = chunks[0][0] if chunks else ""
        best_score = chunks[0][1] if chunks else 0.0
        
        # Simple question type detection
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ['what', 'tell me about', 'describe', 'explain']):
            # Return a safe preview of the most relevant content
            preview_length = min(400, len(best_chunk_text))
            preview = best_chunk_text[:preview_length]
            
            if len(best_chunk_text) > preview_length:
                preview += "..."
            
            confidence_note = ""
            if best_score < 0.4:
                confidence_note = "\n\nNote: The relevance of this information to your question is limited."
            
            return f"Based on the document, here is relevant information:\n\n{preview}{confidence_note}"
        
        elif any(keyword in question_lower for keyword in ['list', 'what are', 'name']):
            # Try to extract list items safely
            list_items = self._extract_safe_list_items(best_chunk_text)
            if list_items:
                return f"Based on the document:\n\n" + "\n".join([f"• {item}" for item in list_items[:5]])
        
        # Generic fallback
        return "I cannot provide a sufficiently accurate answer based on the available information. The document contains relevant content, but I cannot confidently extract a precise answer without potential inaccuracies."
    
    def _extract_safe_list_items(self, text: str) -> List[str]:
        """Safely extract list items from text without interpretation."""
        items = []
        
        # Look for bullet points
        bullet_items = re.findall(r'[•\-*]\s*([^\n]+)', text)
        items.extend([item.strip() for item in bullet_items if len(item.strip()) > 10])
        
        # Look for numbered lists
        numbered_items = re.findall(r'\d+[\.)]\s*([^\n]+)', text)
        items.extend([item.strip() for item in numbered_items if len(item.strip()) > 10])
        
        return items[:8]  # Limit to 8 items
    
    @handle_rag_errors
    @log_execution_time
    def query_with_evaluation(self, question: str, top_k: int = 3, 
                            evaluate: bool = True) -> Tuple[str, List, Dict, Dict]:
        """
        Enhanced query with evaluation metrics for performance tracking.
        
        Returns:
            Tuple of (answer, relevant_chunks, source_info, evaluation_metrics)
        """
        answer, relevant_chunks, source_info = self.query(question, top_k)
        
        evaluation_metrics = {}
        if evaluate and config.evaluation.enable_evaluation:
            evaluation_metrics = evaluator.evaluate_query(question, answer, relevant_chunks)
            logger.info(f"Query evaluation - Precision: {evaluation_metrics['retrieval_precision']:.3f}, "
                       f"Hallucination: {evaluation_metrics['hallucination_rate']:.3f}")
        
        return answer, relevant_chunks, source_info, evaluation_metrics
    
    def query_with_metadata_filters(self, question: str, top_k: int = 3,
                                  document_ids: List[str] = None,
                                  filename_filter: str = None) -> Tuple[str, List, Dict]:
        """
        Enhanced query with multiple metadata filters.
        """
        filter_metadata = {}
        
        # Build complex filters
        if document_ids:
            filter_metadata["document_id"] = {"$in": document_ids}
        
        if filename_filter:
            filter_metadata["filename"] = {"$contains": filename_filter}
        
        # Use embedding manager's search with filters
        relevant_chunks = self.embedding_manager.search(
            question, 
            top_k=top_k, 
            filter_metadata=filter_metadata if filter_metadata else None
        )
        
        if not relevant_chunks:
            return "No relevant information found with the specified filters.", [], {}
        
        # Build universal context
        context = self.context_builder.build_universal_context(relevant_chunks, question)
        
        # Generate answer with anti-hallucination
        answer = self._generate_universal_answer(question, context, relevant_chunks)
        
        return answer, relevant_chunks, {}
    
    def _extract_source_info(self, chunks: List[Tuple[str, float, dict]]) -> Dict:
        """
        Extract detailed source information from retrieved chunks.
        """
        if not chunks:
            return {}
    
        source_info = {
            "total_sources": 0,
            "documents": [],
            "chunk_details": [],
            "primary_source": "Unknown"
        }
    
        # Track unique documents by document_id to avoid duplicates
        unique_docs = {}
        seen_chunk_content = set()
    
        for i, (chunk, score, metadata) in enumerate(chunks):
            # Skip if we've already seen very similar chunk content
            chunk_preview = chunk[:80] + chunk[-20:] if len(chunk) > 100 else chunk
            content_signature = f"{len(chunk)}:{hash(chunk_preview)}"
        
            if content_signature in seen_chunk_content:
                continue
            seen_chunk_content.add(content_signature)
        
            # Extract proper filename
            raw_filename = metadata.get('filename', metadata.get('document_name', 'Unknown Document'))
            source_file = metadata.get('source', 'Unknown Source')
        
            # Clean filename - remove UUID prefix
            if '_' in raw_filename and len(raw_filename.split('_')[0]) == 8:
                clean_filename = '_'.join(raw_filename.split('_')[1:])
            else:
                clean_filename = raw_filename
        
            doc_id = metadata.get('document_id', 'unknown')
        
            # Add to unique documents
            if doc_id not in unique_docs:
                unique_docs[doc_id] = {
                    'document_name': clean_filename,
                    'filename': clean_filename,
                    'source_file': source_file,
                    'document_id': doc_id
                }
        
            # Add chunk details
            source_info["chunk_details"].append({
                "chunk_id": i + 1,
                "document": clean_filename,
                "filename": clean_filename,
                "confidence": f"{score:.3f}",
                "source_file": source_file,
                "document_id": doc_id,
                "content_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
    
        source_info["total_sources"] = len(unique_docs)
        source_info["documents"] = list(set([doc['document_name'] for doc in unique_docs.values()]))
        source_info["primary_source"] = source_info["documents"][0] if source_info["documents"] else "Unknown"
    
        # Sort chunks by confidence score (highest first)
        source_info["chunk_details"].sort(key=lambda x: float(x["confidence"]), reverse=True)
    
        # Limit chunk details to avoid overwhelming display
        source_info["chunk_details"] = source_info["chunk_details"][:5]
    
        return source_info
    
    def get_document_list(self) -> List[Dict]:
        """Get list of all processed documents."""
        if not self.is_initialized:
            self.initialize()
        
        return self.embedding_manager.list_documents()
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store."""
        try:
            self.embedding_manager.delete_document(document_id)
            logger.info(f"Successfully deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        if not self.is_initialized:
            self.initialize()
        
        base_stats = self.embedding_manager.get_stats()
        
        # Add performance stats
        base_stats.update({
            "performance": self.performance_stats,
            "evaluation": evaluator.get_aggregate_metrics()
        })
        
        return base_stats
    
    def get_evaluation_summary(self) -> Dict:
        """Get summary of evaluation metrics across all queries."""
        return evaluator.get_aggregate_metrics()
    
    def experiment_with_chunk_sizes(self, file_path: str, chunk_sizes: List[int] = None) -> Dict[int, Dict]:
        """
        Experiment with different chunk sizes for optimization.
        """
        if chunk_sizes is None:
            chunk_sizes = getattr(config.chunking, 'experimental_sizes', [500, 800, 1000])
        
        results = {}
        
        for chunk_size in chunk_sizes:
            logger.info(f"Testing chunk size: {chunk_size}")
            
            # Process with different chunk size
            text = self.document_processor.load_document(file_path)
            if not text:
                continue
                
            chunks = self.document_processor.chunk_text(text, chunk_size=chunk_size, 
                                                      chunk_overlap=min(100, chunk_size//4))
            
            results[chunk_size] = {
                "chunks_count": len(chunks),
                "avg_chunk_length": np.mean([len(chunk) for chunk in chunks]) if chunks else 0,
                "total_chars_processed": sum(len(chunk) for chunk in chunks),
                "chunk_size_variance": np.var([len(chunk) for chunk in chunks]) if chunks else 0
            }
            
            logger.info(f"Chunk size {chunk_size}: {len(chunks)} chunks, "
                       f"avg length: {results[chunk_size]['avg_chunk_length']:.0f} chars")
        
        return results
    
    def remove_duplicate_chunks(self, chunks: List[Tuple[str, float, dict]]) -> List[Tuple[str, float, dict]]:
        """Remove duplicate or very similar chunks from results."""
        unique_chunks = []
        seen_content = set()
    
        # Sort by score first to keep highest scoring duplicates
        chunks.sort(key=lambda x: x[1], reverse=True)
    
        for chunk, score, metadata in chunks:
            # Create a robust content signature
            chunk_text = chunk.strip().lower()
            if len(chunk_text) < 20:  # Skip very short chunks
                continue
            
            # Use beginning and end for signature
            signature_parts = []
            if len(chunk_text) > 50:
                signature_parts.extend([chunk_text[:30], chunk_text[-20:]])
            else:
                signature_parts.append(chunk_text)
        
            content_sig = "|".join(signature_parts)
        
            if content_sig not in seen_content:
                seen_content.add(content_sig)
                unique_chunks.append((chunk, score, metadata))
    
        return unique_chunks
    
    def clear_evaluation_history(self):
        """Clear evaluation history."""
        evaluator.clear_history()
        logger.info("Evaluation history cleared")

def main():
    """Test the universal RAG pipeline with anti-hallucination features."""
    pipeline = RAGPipeline()
    
    # Create a test document
    test_file = "data/raw_documents/test_universal.txt"
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""
        Universal Document Test
        
        This is a test document to demonstrate the universal RAG pipeline.
        
        Project Alpha:
        - Developed using Python and FastAPI
        - Features machine learning integration
        - Deployed on cloud infrastructure
        
        Project Beta:
        - Built with Java and Spring Boot  
        - Includes RESTful API design
        - Uses PostgreSQL database
        
        Technologies:
        - Programming: Python, Java, JavaScript
        - Frameworks: FastAPI, Spring Boot, React
        - Databases: PostgreSQL, MongoDB
        """)
    
    # Test the universal pipeline
    print("=== Testing Universal RAG Pipeline ===")
    
    # Process document
    success, doc_id, filename = pipeline.process_document(test_file)
    if success:
        print(f"✓ Document processed successfully: {filename} (ID: {doc_id})")
        
        # Test universal query
        query = "What technologies are mentioned?"
        answer, chunks, source_info = pipeline.query(query)
        
        print(f"\nQuery: '{query}'")
        print(f"Answer: {answer}")
        print(f"\n📊 Source Information:")
        print(f"   - Documents: {', '.join(source_info['documents'])}")
        print(f"   - Chunks used: {len(source_info['chunk_details'])}")
        
        # Test with evaluation
        answer, chunks, source_info, eval_metrics = pipeline.query_with_evaluation(
            "Tell me about Project Alpha"
        )
        
        print(f"\n📈 Evaluation Metrics:")
        for metric, value in eval_metrics.items():
            print(f"   - {metric}: {value:.3f}")
            
        # Cleanup
        pipeline.delete_document(doc_id)
        os.remove(test_file)
            
    else:
        print("✗ Failed to process document")

if __name__ == "__main__":
    main()
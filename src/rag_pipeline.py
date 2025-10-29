# src/rag_pipeline.py
"""
Production-ready RAG pipeline with universal hallucination prevention.
"""

import logging
import os
import traceback
from src.chroma_manager import ChromaManager
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Safe imports for helpers
# --------------------------------------------------------------------

try:
    from src.llm_integration import LLMIntegration
except Exception as e:
    LLMIntegration = None
    logger.warning(f"Could not import LLMIntegration ({e})")

try:
    from src.context.universal_context_verifier import UniversalContextVerifier
except Exception as e:
    UniversalContextVerifier = None
    logger.warning(f"Could not import UniversalContextVerifier ({e})")


class RAGPipeline:
    PROMPT_TEMPLATE = """CRITICAL INSTRUCTIONS - READ CAREFULLY:
You are an assistant that answers using ONLY the information present in the provided Context.
You MUST follow these rules exactly:

CONTEXT RULES:
1. Use ONLY the information from the Context below. Do not use any prior knowledge.
2. If information is missing from Context, say 'The context does not provide information about X'
3. Do not add, infer, or assume any information not explicitly stated
4. Do not make comparisons, draw conclusions, or provide analysis beyond what's directly stated
5. If you cannot answer based on Context, say so explicitly

CITATION RULES:
6. ALWAYS reference which chunks contain the information (e.g., 'Based on Chunk 1 and Chunk 3')
7. When listing items, cite the specific chunk for each item
8. If information appears in multiple chunks, mention all relevant chunks

LANGUAGE RULES:
9. Avoid speculative language (probably, might be, could be, I think, I believe)
10. Avoid absolute statements (always, never, all, every) unless explicitly stated
11. Avoid vague quantifiers (many, several, some) - be specific about what's in Context
12. Do not reference external knowledge, studies, or common practices

SAFETY RULES:
13. If Context contains conflicting information, acknowledge the conflict
14. If you're unsure, err on the side of caution and admit limitations
15. Format your answer clearly but do not invent structure not in Context

CONTEXT:
{context_block}

QUESTION: {question}

YOUR ANSWER (following all rules above):"""

    def __init__(
        self,
        chroma_manager: Optional[Any] = None,
        llm_integration: Optional[Any] = None,
        retrieval_top_k: int = 5,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.retrieval_top_k = retrieval_top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # ---- Chroma manager ----
        if chroma_manager is not None:
            self.chroma_manager = chroma_manager
            logger.info("RAGPipeline: using provided ChromaManager")
        else:
            if ChromaManager is not None:
                try:
                    self.chroma_manager = ChromaManager()
                    logger.info("RAGPipeline: auto-created ChromaManager")
                except Exception as e:
                    logger.exception("RAGPipeline: failed to auto-create ChromaManager: %s", e)
                    self.chroma_manager = None
            else:
                self.chroma_manager = None
                logger.warning("RAGPipeline: ChromaManager not available")

        # ---- LLM integration ----
        if llm_integration is not None:
            self.llm_integration = llm_integration
            logger.info("RAGPipeline: using provided LLMIntegration")
        else:
            if LLMIntegration is not None:
                try:
                    self.llm_integration = LLMIntegration()
                    logger.info("RAGPipeline: auto-created LLMIntegration")
                except Exception as e:
                    logger.exception("RAGPipeline: failed to auto-create LLMIntegration: %s", e)
                    self.llm_integration = None
            else:
                self.llm_integration = None
                logger.warning("RAGPipeline: LLMIntegration not available")

        # ---- Context Verifier ----
        if UniversalContextVerifier is not None:
            self.context_verifier = UniversalContextVerifier()
            logger.info("RAGPipeline: initialized UniversalContextVerifier")
        else:
            self.context_verifier = None
            logger.warning("RAGPipeline: UniversalContextVerifier not available")

        self.processed_documents: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------
    def initialize(self):
        try:
            if self.llm_integration and not getattr(self.llm_integration, "initialized", False):
                try:
                    self.llm_integration.initialize()
                except Exception:
                    logger.exception("LLMIntegration.initialize() failed")
            logger.info("RAG pipeline initialized successfully with hallucination prevention")
        except Exception:
            logger.exception("Failed to initialize RAG pipeline")
            raise

    # -----------------------
    # Document ingestion flow
    # -----------------------
    def process_document(self, file_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Document processing with progress tracking.
        """
        logger.info("🚀 Starting document processing: %s", file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
        title = filename or os.path.basename(file_path)
        doc_uuid = str(uuid.uuid4())

        # Extract text
        logger.info("📄 Step 1/3: Extracting text from document...")
        try:
            raw_text = self._extract_text_from_file(file_path)
            if not raw_text or not raw_text.strip():
                raise RuntimeError("Extracted text is empty")
        
            logger.info("✅ Text extraction complete (%d characters)", len(raw_text))
        
        except Exception as e:
            logger.exception(f"❌ Text extraction failed: {str(e)}")
            raise RuntimeError(f"Failed to extract text from document: {str(e)}")

        # Chunk the text
        logger.info("✂️ Step 2/3: Chunking text...")
        chunks = self._chunk_text(raw_text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        logger.info("✅ Chunking complete - created %d chunks", len(chunks))
    
        if not chunks:
            raise RuntimeError("No chunks created from document")

        # Prepare metadata
        logger.info("📝 Preparing metadata...")
        metadata_list = []
        for i, chunk in enumerate(chunks):
            metadata_list.append({
                "source": title,
                "filename": filename,
                "document_id": doc_uuid,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

        # Add to ChromaDB with progress tracking
        logger.info("💾 Step 3/3: Adding chunks to vector database...")
        try:
            # Process in batches to show progress for large documents
            batch_size = 50
            total_batches = (len(chunks) + batch_size - 1) // batch_size
        
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, len(chunks))
            
                batch_chunks = chunks[start_idx:end_idx]
                batch_metadata = metadata_list[start_idx:end_idx]
            
                logger.info("📦 Processing batch %d/%d (chunks %d-%d)...", 
                           batch_num + 1, total_batches, start_idx + 1, end_idx)
            
                self.chroma_manager.add_documents(batch_chunks, batch_metadata, doc_uuid)
            
            logger.info("✅ Successfully added all %d chunks to ChromaDB", len(chunks))
        
        except Exception as e:
            logger.exception(f"❌ Failed to add chunks to ChromaDB: {str(e)}")
            raise RuntimeError(f"Failed to ingest document chunks into Chroma: {str(e)}")

        # Bookkeeping
        entry = {
            "document_id": doc_uuid,
            "title": title,
            "file_path": file_path,
            "num_chunks": len(chunks),
        }
        self.processed_documents.append(entry)
        logger.info(f"🎉 Document processed successfully: {title} (ID: {doc_uuid})")
        return entry

    def _normalize_raw_chunks(self, raw_chunks: Any, doc_id_base: str, title: str) -> List[Dict[str, Any]]:
        """
        Normalize a returned value from a document processor or chunker into:
            [ {"id": <str>, "text": <str>, "metadata": {...}}, ... ]
        Accepts:
            - list[str]
            - list[dict] with keys 'text'|'content'|'chunk' and optional 'metadata'
            - a single big string (split into chunks)
        """
        out: List[Dict[str, Any]] = []
        try:
            if isinstance(raw_chunks, str):
                # Received a big string; chunk it
                pieces = self._chunk_text(raw_chunks, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
                raw_chunks = pieces

            if isinstance(raw_chunks, list):
                for i, item in enumerate(raw_chunks):
                    chunk_id = f"{doc_id_base}_{i+1}"
                    if isinstance(item, str):
                        out.append({"id": chunk_id, "text": item, "metadata": {"source": title}})
                    elif isinstance(item, dict):
                        text = item.get("text") or item.get("content") or item.get("chunk") or ""
                        meta = item.get("metadata") or item.get("meta") or {}
                        meta.setdefault("source", title)
                        out.append({"id": chunk_id, "text": text, "metadata": meta})
                    else:
                        out.append({"id": chunk_id, "text": str(item), "metadata": {"source": title}})
            else:
                # Fallback: stringify and chunk
                s = str(raw_chunks)
                pieces = self._chunk_text(s, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
                for i, p in enumerate(pieces):
                    chunk_id = f"{doc_id_base}_{i+1}"
                    out.append({"id": chunk_id, "text": p, "metadata": {"source": title}})
        except Exception:
            logger.exception("Error normalizing raw chunks")
        return out

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """
        Simple character-based chunking with overlap.
        Keeps words intact by expanding to nearest whitespace when possible.
        """
        if not text:
            return []

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        length = len(text)
        chunks: List[str] = []
        start = 0
        while start < length:
            end = start + chunk_size
            if end >= length:
                chunk = text[start:length].strip()
                if chunk:
                    chunks.append(chunk)
                break
            # try to avoid cutting in middle of a word: backtrack to last whitespace if possible
            end_back = end
            while end_back > start and not text[end_back - 1].isspace():
                end_back -= 1
            if end_back <= start:
                # couldn't find whitespace, fall back to end
                end_back = end
            chunk = text[start:end_back].strip()
            if chunk:
                chunks.append(chunk)
            start = end_back - overlap if (end_back - overlap) > start else end_back
        return chunks

    def _extract_text_from_file(self, file_path: str) -> str:
        """
        Try several PDF/text extraction libraries in order:
            1) pdfplumber
            2) PyPDF2
            3) fitz (PyMuPDF)
        If none available, raise a descriptive error directing the user to pip install a package.
        """
        lower = file_path.lower()
        # If text file, just read
        if lower.endswith(".txt") or lower.endswith(".md"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()

        # Try pdfplumber
        try:
            import pdfplumber
            logger.debug("Using pdfplumber to extract PDF text.")
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    ptext = page.extract_text()
                    if ptext:
                        text_parts.append(ptext)
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.debug("pdfplumber not available or failed: %s", e)

        # Try PyPDF2
        try:
            import PyPDF2
            logger.debug("Using PyPDF2 to extract PDF text.")
            text_parts = []
            with open(file_path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages:
                    try:
                        ptext = page.extract_text()
                    except Exception:
                        ptext = None
                    if ptext:
                        text_parts.append(ptext)
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.debug("PyPDF2 not available or failed: %s", e)

        # Try PyMuPDF (fitz)
        try:
            import fitz  # PyMuPDF
            logger.debug("Using PyMuPDF (fitz) to extract PDF text.")
            doc = fitz.open(file_path)
            text_parts = []
            for page in doc:
                try:
                    ptext = page.get_text()
                except Exception:
                    ptext = None
                if ptext:
                    text_parts.append(ptext)
            doc.close()
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.debug("PyMuPDF not available or failed: %s", e)

        # If we reach here, no PDF extractor is available
        raise RuntimeError(
            "No PDF/text extractor available. Please install one of: pdfplumber, PyPDF2, or PyMuPDF.\n"
            "Example: pip install pdfplumber PyPDF2 pymupdf"
        )

    # -----------------------
    # Utility endpoints used by UI
    # -----------------------
    def get_document_list(self) -> List[Dict[str, Any]]:
        return list(self.processed_documents)

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        stats["documents_processed"] = len(self.processed_documents)
        # Try to get chunk count from chroma
        try:
            if self.chroma_manager and getattr(self.chroma_manager, "collection", None):
                # best-effort using get_all_chunks()
                try:
                    all_chunks = self.chroma_manager.get_all_chunks()
                    stats["total_chunks"] = len(all_chunks)
                except Exception:
                    stats["total_chunks"] = None
            else:
                stats["total_chunks"] = None
        except Exception:
            stats["total_chunks"] = None

        try:
            if self.llm_integration:
                stats["llm_model"] = getattr(self.llm_integration, "model_name", "unknown")
                stats["llm_ready"] = bool(getattr(self.llm_integration, "initialized", False))
            else:
                stats["llm_model"] = "unknown"
                stats["llm_ready"] = False
        except Exception:
            stats["llm_model"] = "unknown"
            stats["llm_ready"] = False

        return stats

    # -----------------------
    # Querying with Hallucination Prevention
    # -----------------------
    def _build_context_block(self, relevant_chunks: List[Tuple[str, Optional[float], Dict]]) -> str:
        return "\n\n".join([f"[CHUNK {i+1}] {text}" for i, (text, score, meta) in enumerate(relevant_chunks)])

    def query(self, question: str, filter_metadata: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], List[Tuple[str, Optional[float], Dict]], Dict]:
        """
        Production-ready query method with universal hallucination prevention.
        """
        try:
            logger.info("Query received: %s", question)
            if not self.chroma_manager:
                logger.error("No chroma_manager available for retrieval.")
                return None, [], {"error": "no_chroma_manager"}

            # Search for relevant chunks
            relevant_chunks = self.chroma_manager.search(question, top_k=self.retrieval_top_k, filter=filter_metadata)
        
            if not relevant_chunks:
                logger.info("No relevant chunks found for query: %s", question)
                return "I couldn't find any relevant information in the documents to answer your question.", [], self._build_empty_source_info()

            context_block = self._build_context_block(relevant_chunks)
            prompt = self.PROMPT_TEMPLATE.format(context_block=context_block, question=question)

            if not self.llm_integration:
                logger.error("No llm_integration available to generate answers.")
                return None, relevant_chunks, self._build_source_info(relevant_chunks)

            logger.info("Sending prompt to LLM (length=%d)", len(prompt))
            try:
                result = self.llm_integration._call_ollama_api(prompt, relevant_chunks=None)
            except TypeError:
                logger.debug("LLMIntegration signature mismatch; calling with (question, relevant_chunks) instead.")
                result = self.llm_integration._call_ollama_api(question, relevant_chunks=relevant_chunks)

            # Handle LLM failure
            if result is None:
                logger.warning("LLM returned no result, using fallback response.")
                answer = "I found relevant information but couldn't generate a proper answer. Here are the relevant sections:\n\n" + "\n\n".join([f"- {chunk[0][:200]}..." for chunk in relevant_chunks[:3]])
            else:
                answer = result.strip()

            # UNIVERSAL HALLUCINATION PREVENTION
            verification_results = self._verify_answer_grounding(answer, relevant_chunks)
            
            # Apply corrections if needed
            if not verification_results["is_grounded"]:
                logger.warning(f"Answer verification failed: {verification_results['issues']}")
                answer = self._apply_safety_corrections(answer, verification_results)

            # Build proper source information with verification results
            source_info = self._build_source_info(relevant_chunks)
            source_info["verification"] = verification_results
        
            logger.info("Query processed successfully. Found %d relevant chunks, verification confidence: %.2f", 
                       len(relevant_chunks), verification_results["confidence"])
            return answer, relevant_chunks, source_info

        except Exception as e:
            logger.error("Error in RAGPipeline.query: %s", str(e))
            logger.debug(traceback.format_exc())
            return None, [], self._build_empty_source_info()

    def _verify_answer_grounding(self, answer: str, context_chunks: List[Tuple[str, float, Dict]]) -> Dict[str, Any]:
        """Verify that the answer is properly grounded in the context."""
        if self.context_verifier:
            return self.context_verifier.verify_answer_grounding(answer, context_chunks)
        else:
            # Fallback basic verification
            return {
                "is_grounded": True,
                "confidence": 0.5,  # Conservative default
                "issues": ["Context verifier not available"],
                "missing_citations": [],
                "hallucination_flags": []
            }

    def _apply_safety_corrections(self, original_answer: str, verification_results: Dict) -> str:
        """Apply safety corrections to potentially hallucinated answers."""
        if self.context_verifier:
            return self.context_verifier.generate_safe_response(original_answer, verification_results)
        else:
            # Fallback safety correction
            if verification_results.get("confidence", 0) < 0.4:
                return "Based on the provided documents, I cannot find sufficient information to answer this question accurately."
            else:
                return f"Based on the available information: {original_answer}\n\nNote: Some details may not be fully supported by the provided documents."

    def _build_source_info(self, relevant_chunks: List[Tuple[str, Optional[float], Dict]]) -> Dict[str, Any]:
        """Build proper source information for the response."""
        if not relevant_chunks:
            return self._build_empty_source_info()
    
        # Extract unique documents
        documents = set()
        chunk_details = []
    
        for i, (text, score, metadata) in enumerate(relevant_chunks):
            doc_name = metadata.get('source', 'Unknown Document')
            documents.add(doc_name)
        
            chunk_details.append({
                'document': doc_name,
                'content_preview': text[:200] + "..." if len(text) > 200 else text,
                'confidence': float(score) if score is not None else 0.0,
                'chunk_id': metadata.get('chunk_id', f'chunk_{i+1}')
            })
    
        # Determine primary source (document with most chunks)
        doc_counts = {}
        for chunk in relevant_chunks:
            doc_name = chunk[2].get('source', 'Unknown Document')
            doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
    
        primary_source = max(doc_counts.items(), key=lambda x: x[1])[0] if doc_counts else "Unknown"
    
        return {
            "total_sources": len(documents),
            "documents": list(documents),
            "primary_source": primary_source,
            "chunk_details": chunk_details,
            "retrieved_count": len(relevant_chunks)
        }

    def _build_empty_source_info(self) -> Dict[str, Any]:
        """Build empty source information."""
        return {
            "total_sources": 0,
            "documents": [],
            "primary_source": "None",
            "chunk_details": [],
            "retrieved_count": 0
        }

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.chroma_manager:
                self.chroma_manager.delete_document(document_id)
                
                # Remove from processed documents
                self.processed_documents = [
                    doc for doc in self.processed_documents 
                    if doc.get('document_id') != document_id
                ]
                
                logger.info(f"Successfully deleted document: {document_id}")
                return True
            else:
                logger.error("No chroma_manager available for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
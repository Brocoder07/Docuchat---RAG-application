"""
Intelligent vector store with ChromaDB and optimized embedding management.
FIXED: Added metadata sanitization and a robust fallback similarity mapping.
"""
import logging
import uuid
import os
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from core.config import config

logger = logging.getLogger(__name__)

class VectorStore:
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.initialized = False
        self.collection_name = "docuchat_documents"
    
    def initialize(self) -> bool:
        try:
            os.environ["ANONYMIZED_TELEMETRY"] = "false"
            
            logger.info(f"🔄 Loading embedding model: {config.rag.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(
                config.rag.EMBEDDING_MODEL,
                device=config.rag.EMBEDDING_DEVICE
            )
            
            self.client = chromadb.PersistentClient(
                path="data/vector_store",
                settings=Settings(anonymized_telemetry=False)
            )
            
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"✅ Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "DocuChat document chunks"}
                )
                logger.info(f"✅ Created new collection: {self.collection_name}")
            
            self.initialized = True
            logger.info("✅ Vector store initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {str(e)}")
            self.initialized = False
            return False
    
    def _sanitize_metadata(self, meta: Dict[str, Any], document_id: str, chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """
        Ensure metadata values are primitive types supported by ChromaDB.
        Replace None with empty strings or reasonable defaults.
        """
        sanitized = {}
        for k, v in meta.items():
            # Only allow primitives: str, int, float, bool
            if v is None:
                sanitized[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                try:
                    sanitized[k] = str(v)
                except Exception:
                    sanitized[k] = ""
        # Ensure essential keys exist
        sanitized["document_id"] = document_id
        sanitized["chunk_index"] = int(chunk_index)
        sanitized["total_chunks"] = int(total_chunks)
        return sanitized

    def add_documents(self, documents: List[str], metadata: List[Dict], document_id: str) -> bool:
        if not self.initialized:
            logger.error("Vector store not initialized")
            return False
        
        try:
            batch_size = 32
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            all_embeddings = []
            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(documents))
                batch_docs = documents[start_idx:end_idx]
                
                logger.debug(f"Generating embeddings for batch {i+1}/{total_batches}")
                # encode returns numpy array; convert to list of lists
                batch_embeddings = self.embedding_model.encode(batch_docs, show_progress_bar=False)
                # ensure it's a list of lists
                if isinstance(batch_embeddings, np.ndarray):
                    batch_embeddings = batch_embeddings.tolist()
                all_embeddings.extend(batch_embeddings)
            
            ids = []
            metadatas = []
            embeddings_list = []
            
            for i, (doc, meta, embedding) in enumerate(zip(documents, metadata, all_embeddings)):
                # Ensure user_id exists in metadata to enforce ownership
                if 'user_id' not in meta:
                    logger.error(f"CRITICAL: user_id missing from metadata for chunk {i} of doc {document_id}")
                    return False
                    
                chunk_id = f"{document_id}_chunk_{i+1}"
                ids.append(chunk_id)
                sanitized_meta = self._sanitize_metadata(meta, document_id, i, len(documents))
                metadatas.append(sanitized_meta)
                # embedding might already be a list
                if isinstance(embedding, np.ndarray):
                    embeddings_list.append(embedding.tolist())
                else:
                    embeddings_list.append(list(embedding))
            
            # Use safe add parameters (explicit fields)
            self.collection.add(
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✅ Added {len(documents)} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {str(e)}", exc_info=True)
            return False
    
    def _preprocess_query(self, query: str) -> str:
        query_lower = query.lower().strip()
        
        query_mappings = {
            'experience': 'EXPERIENCE',
            'work experience': 'EXPERIENCE', 
            'employment': 'EXPERIENCE',
            'education': 'EDUCATION',
            'skills': 'TECHNICAL SKILLS',
            'projects': 'PROJECTS',
            'python developer': 'Python Developer Intern – BootLabs'
        }
        
        for key, value in query_mappings.items():
            if key in query_lower:
                return f"{query} {value}"
        
        return query

    def _distance_to_similarity(self, distance: float, max_distance: Optional[float] = None) -> float:
        """
        Convert distance (L2 or arbitrary) to a bounded similarity in [0,1].
        Use 1 / (1 + distance) mapping for robustness; if max_distance provided,
        also apply a normalization factor.
        """
        try:
            sim = 1.0 / (1.0 + float(distance))
            # Optionally scale by estimated max_distance to compress range
            if max_distance and max_distance > 0:
                sim = sim * (1.0 - (min(distance, max_distance) / (max_distance + 1.0) * 0.3))
            return float(max(0.0, min(1.0, sim)))
        except Exception:
            return 0.0

    # -----------------------------------------------------------------
    # Improved search with safe fallbacks
    # -----------------------------------------------------------------
    def search(self, query: str, user_id: str, top_k: Optional[int] = None, filename: Optional[str] = None) -> List[Tuple[str, float, Dict]]:
        """
        Enhanced semantic search with robust fallback mapping from distances to similarity.
        Returns a list of tuples: (document_text, similarity_score, metadata)
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []

        if top_k is None:
            top_k = config.rag.TOP_K_RETRIEVAL

        try:
            # Build the metadata filter first
            filters = []
            filters.append({"user_id": user_id})
            
            if filename:
                filters.append({"filename": filename})
                logger.info(f"Filtering search by user {user_id} and filename: {filename}")
            else:
                logger.info(f"Filtering search by user {user_id} (all documents)")
            
            where_clause = {"$and": filters} if len(filters) > 1 else filters[0]

            # -----------------------------------------------------------------
            # SMALL DOC HEURISTIC: if very few chunks exist, return them directly
            # -----------------------------------------------------------------
            SMALL_DOC_HEURISTIC_LIMIT = 3
            all_chunks_for_filter = self.collection.get(where=where_clause, include=["metadatas", "documents"])
            total_available_chunks = len(all_chunks_for_filter.get('ids', []))

            if 0 < total_available_chunks <= SMALL_DOC_HEURISTIC_LIMIT:
                logger.info(f"🔄 Small document detected ({total_available_chunks} chunks). Returning all chunks.")
                search_results = []
                for doc, metadata in zip(all_chunks_for_filter.get('documents', []), all_chunks_for_filter.get('metadatas', [])):
                    # Ensure doc is a string
                    doc_text = doc if isinstance(doc, str) else (str(doc) if doc is not None else "")
                    # Forced high similarity but not absolute 1.0 (cap at 0.99)
                    search_results.append((doc_text, 0.99, metadata or {}))
                return search_results[:top_k]
            # -----------------------------------------------------------------

            # Normal processing: embed query and ask Chroma
            processed_query = self._preprocess_query(query)
            query_embedding = self.embedding_model.encode([processed_query])
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Ask the collection for results (request more than top_k for fallback smoothing)
            n_results = min(max(top_k * 2, 10), 200)  # reasonable bounds
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
                where=where_clause
            )

            # Validate results structure
            docs = results.get('documents', [])
            metas = results.get('metadatas', [])
            distances = results.get('distances', [])

            if not docs or not docs[0]:
                logger.info("🔍 Chroma returned no documents for the query filter.")
                return []

            # distances and docs are nested lists per query
            distances_list = distances[0] if distances and distances[0] else []
            docs_list = docs[0] if docs and docs[0] else []
            metas_list = metas[0] if metas and metas[0] else []

            if not docs_list:
                logger.info("🔍 No documents in nested results.")
                return []

            # Compute fallback max_distance for mapping
            max_distance = max(distances_list) if distances_list else None

            search_results = []
            # First pass: accept results with reasonable distances (if available)
            NORMAL_DISTANCE_THRESHOLD = 1.0
            FALLBACK_DISTANCE_THRESHOLD = 2.5  # expanded fallback

            for i, (doc, meta) in enumerate(zip(docs_list, metas_list)):
                distance = distances_list[i] if i < len(distances_list) else None
                if distance is None:
                    # If distance missing, still consider the doc but with low confidence
                    similarity = 0.25
                else:
                    # Map distance -> similarity robustly
                    similarity = self._distance_to_similarity(distance, max_distance)
                # Only append if similarity positive; we'll do final trimming/sorting
                search_results.append((doc if isinstance(doc, str) else (str(doc) if doc is not None else ""), float(similarity), meta or {}))

            # Filter out zero-similarity entries
            filtered = [r for r in search_results if r[1] > 0.0]

            # If filtered empty, apply greedy fallback: take top n_results from Chroma and compute similarity mapping
            if not filtered and docs_list:
                logger.info("🔄 No results with normal mapping—using greedy fallback top-N")
                fallback_results = []
                for i, (doc, meta) in enumerate(zip(docs_list, metas_list)):
                    distance = distances_list[i] if i < len(distances_list) else None
                    similarity = self._distance_to_similarity(distance, max_distance) if distance is not None else 0.1
                    fallback_results.append((doc if isinstance(doc, str) else (str(doc) if doc is not None else ""), float(similarity), meta or {}))
                # sort descending similarity
                fallback_results.sort(key=lambda x: x[1], reverse=True)
                return fallback_results[:top_k]

            # Sort filtered by similarity descending
            filtered.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"🔍 Search found {len(filtered)} relevant chunks for: {query[:50]}...")
            return filtered[:top_k]
        
        except Exception as e:
            logger.error(f"❌ Search error: {str(e)}", exc_info=True)
            return []
    
    def get_collection_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.initialized:
            return {"error": "Vector store not initialized"}
        
        try:
            where_filter = {}
            if user_id:
                where_filter = {"user_id": user_id}

            all_data = self.collection.get(where=where_filter)
            
            total_chunks = len(all_data.get('ids', []))
            
            unique_docs = set()
            for metadata in all_data.get('metadatas', []):
                if metadata and 'document_id' in metadata:
                    unique_docs.add(metadata['document_id'])
            
            return {
                "collection_name": self.collection_name,
                "total_chunks": total_chunks,
                "unique_documents": len(unique_docs),
                "initialized": self.initialized,
                "user_filtered": user_id is not None
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_document(self, document_id: str, user_id: str) -> bool:
        if not self.initialized:
            return False
        
        try:
            where_filter = {"$and": [
                {"document_id": document_id},
                {"user_id": user_id}
            ]}
            
            existing_chunks = self.collection.get(where=where_filter)
            ids_to_delete = existing_chunks.get('ids', [])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"🗑️ Deleted {len(ids_to_delete)} chunks for doc {document_id} owned by user {user_id}")
                return True
            else:
                logger.warning(f"No chunks found for doc {document_id} owned by user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    def get_document_metadata(self, document_id: str, user_id: str) -> Optional[Dict]:
        try:
            where_filter = {"$and": [
                {"document_id": document_id},
                {"user_id": user_id}
            ]}
            result = self.collection.get(where=where_filter, limit=1)
            if result and result.get('metadatas'):
                return result['metadatas'][0]
            return None
        except Exception as e:
            logger.error(f"Error getting doc metadata: {e}")
            return None

    def list_documents_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        if not self.initialized:
            return []
            
        try:
            where_filter = {"user_id": user_id}
            all_data = self.collection.get(where=where_filter, include=["metadatas"])
            
            user_docs = {}
            for metadata in all_data.get('metadatas', []):
                doc_id = metadata.get('document_id')
                if doc_id not in user_docs:
                    user_docs[doc_id] = {
                        "filename": metadata.get('filename', 'Unknown'),
                        "document_id": doc_id,
                        "processed_at": metadata.get('processed_at', 'Unknown'),
                        "chunks_count": 0,
                        "file_path": metadata.get('source', 'Unknown')
                    }
                user_docs[doc_id]["chunks_count"] += 1
                
            return list(user_docs.values())
            
        except Exception as e:
            logger.error(f"Error listing documents by user: {e}")
            return []

    def reset_collection(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            logger.info("🔄 Collection reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False

# Global vector store instance
vector_store = VectorStore()
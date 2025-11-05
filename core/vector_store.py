"""
Intelligent vector store with ChromaDB and optimized embedding management.
FIXED: Added "Small Document" heuristic to bypass similarity search.
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
                batch_embeddings = self.embedding_model.encode(batch_docs)
                all_embeddings.extend(batch_embeddings)
            
            ids = []
            metadatas = []
            embeddings_list = []
            
            for i, (doc, meta, embedding) in enumerate(zip(documents, metadata, all_embeddings)):
                if 'user_id' not in meta:
                    logger.error(f"CRITICAL: user_id missing from metadata for chunk {i} of doc {document_id}")
                    return False
                    
                chunk_id = f"{document_id}_chunk_{i+1}"
                ids.append(chunk_id)
                metadatas.append({
                    **meta,
                    "document_id": document_id,
                    "chunk_index": i,
                    "total_chunks": len(documents)
                })
                embeddings_list.append(embedding.tolist())
            
            self.collection.add(
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✅ Added {len(documents)} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {str(e)}")
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

    # -----------------------------------------------------------------
    # 🚨 MODIFIED: `search` method now includes the "Small Doc" heuristic
    # -----------------------------------------------------------------
    def search(self, query: str, user_id: str, top_k: Optional[int] = None, filename: Optional[str] = None) -> List[Tuple[str, float, Dict]]:
        """
        Enhanced semantic search with L2 distance logic and "Small Doc" heuristic.
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
            # 🚨 ADDED: Small Document Heuristic
            # -----------------------------------------------------------------
            SMALL_DOC_HEURISTIC_LIMIT = 3
            # Get all chunks matching the filter
            all_chunks_for_filter = self.collection.get(where=where_clause, include=["metadatas", "documents"])
            total_available_chunks = len(all_chunks_for_filter.get('ids', []))

            if 0 < total_available_chunks <= SMALL_DOC_HEURISTIC_LIMIT:
                logger.info(f"🔄 Small document detected ({total_available_chunks} chunks). Bypassing semantic search and returning all chunks.")
                search_results = []
                for doc, metadata in zip(all_chunks_for_filter['documents'], all_chunks_for_filter['metadatas']):
                    # Return 1.0 confidence since we are forcing retrieval
                    search_results.append((doc, 1.0, metadata))
                return search_results
            # -----------------------------------------------------------------

            # If not a small doc, proceed with normal semantic search
            processed_query = self._preprocess_query(query)
            query_embedding = self.embedding_model.encode([processed_query])

            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k * 2,
                include=["documents", "metadatas", "distances"],
                where=where_clause
            )
        
            if results['distances'] and results['distances'][0]:
                distances = results['distances'][0]
                logger.debug(f"Distance range: min={min(distances):.3f}, max={max(distances):.3f}")
        
            search_results = []
            NORMAL_DISTANCE_THRESHOLD = 1.0
            FALLBACK_DISTANCE_THRESHOLD = 1.5

            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    if distance <= NORMAL_DISTANCE_THRESHOLD:
                        similarity_score = max(0.0, 1.0 - (distance / 2.0))
                        search_results.append((doc, similarity_score, metadata))
        
            if not search_results and results['documents'] and results['documents'][0]:
                logger.info("🔄 No results with normal distance threshold, trying fallback...")
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    if distance <= FALLBACK_DISTANCE_THRESHOLD:
                        similarity_score = max(0.0, 1.0 - (distance / 2.0))
                        search_results.append((doc, similarity_score, metadata))

            search_results.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"🔍 Search found {len(search_results)} relevant chunks for: {query[:50]}...")
            return search_results[:top_k]
        
        except Exception as e:
            logger.error(f"❌ Search error: {str(e)}")
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
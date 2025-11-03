"""
Intelligent vector store with ChromaDB and optimized embedding management.
Senior Engineer Principle: Separate storage concerns from business logic.
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
    """
    Managed vector store with optimized operations and error handling.
    Senior Engineer Principle: Hide complexity behind simple interfaces.
    """
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.initialized = False
        self.collection_name = "docuchat_documents"
    
    def initialize(self) -> bool:
        """Initialize vector store with proper error handling."""
        try:
            # 🚨 DISABLE CHROMADB TELEMETRY (BEFORE ANY CHROMADB IMPORTS/INIT)
            os.environ["ANONYMIZED_TELEMETRY"] = "false"
            
            # Initialize embedding model
            logger.info(f"🔄 Loading embedding model: {config.rag.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(
                config.rag.EMBEDDING_MODEL,
                device=config.rag.EMBEDDING_DEVICE
            )
            
            # Initialize ChromaDB with telemetry disabled
            self.client = chromadb.PersistentClient(
                path="data/vector_store",
                settings=Settings(anonymized_telemetry=False)  # Explicit disable
            )
            
            # Get or create collection
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
        """
        Add documents to vector store with batch processing.
        Senior Engineer Principle: Process large documents efficiently.
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return False
        
        try:
            # Generate embeddings in batches for memory efficiency
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
            
            # Prepare data for ChromaDB
            ids = []
            metadatas = []
            embeddings_list = []
            
            for i, (doc, meta, embedding) in enumerate(zip(documents, metadata, all_embeddings)):
                chunk_id = f"{document_id}_chunk_{i+1}"
                ids.append(chunk_id)
                metadatas.append({
                    **meta,
                    "document_id": document_id,
                    "chunk_index": i,
                    "total_chunks": len(documents)
                })
                embeddings_list.append(embedding.tolist())
            
            # Add to collection
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
        """Enhanced query preprocessing for resumes."""
        query_lower = query.lower().strip()
        
        # Map common resume queries to section headers
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

    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float, Dict]]:
        """
        Enhanced semantic search with better query processing.
        FIXED: Corrected threshold logic and similarity calculation
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []
    
        if top_k is None:
            top_k = config.rag.TOP_K_RETRIEVAL
    
        try:
            processed_query = self._preprocess_query(query)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([processed_query])
        
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k * 2,  # Get more results for filtering
                include=["documents", "metadatas", "distances"]
            )
        
            # 🚨 DEBUG: Check distance range to understand similarity calculation
            if results['distances'] and results['distances'][0]:
                distances = results['distances'][0]
                logger.debug(f"Distance range: min={min(distances):.3f}, max={max(distances):.3f}")
        
            # Process results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # 🚨 FIXED: Better similarity calculation for cosine distance
                    # ChromaDB typically uses cosine distance (0-2 range)
                    # similarity = 1 means identical, 0 means orthogonal
                    similarity_score = 1.0 - distance
                
                    # 🚨 FIXED: Use reasonable threshold for normal search
                    if similarity_score >= 0.3:
                        search_results.append((doc, similarity_score, metadata))
        
            # 🚨 FIXED: Fallback with LOWER threshold (was backwards before)
            if not search_results and results['documents'] and results['documents'][0]:
                logger.info("🔄 No results with normal threshold, trying fallback...")
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity_score = 1.0 - distance
                    if similarity_score >= 0.15:  # 🚨 LOWER threshold for fallback
                        search_results.append((doc, similarity_score, metadata))
        
            logger.info(f"🔍 Search found {len(search_results)} relevant chunks for: {query[:50]}...")
            return search_results[:top_k]  # Return only top_k
        
        except Exception as e:
            logger.error(f"❌ Search error: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics for monitoring."""
        if not self.initialized:
            return {"error": "Vector store not initialized"}
        
        try:
            all_data = self.collection.get()
            total_chunks = len(all_data.get('ids', []))
            
            # Count unique documents
            unique_docs = set()
            for metadata in all_data.get('metadatas', []):
                if metadata and 'document_id' in metadata:
                    unique_docs.add(metadata['document_id'])
            
            return {
                "collection_name": self.collection_name,
                "total_chunks": total_chunks,
                "unique_documents": len(unique_docs),
                "initialized": self.initialized
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a specific document."""
        if not self.initialized:
            return False
        
        try:
            # Get all chunks for this document
            all_data = self.collection.get()
            ids_to_delete = []
            
            for i, metadata in enumerate(all_data.get('metadatas', [])):
                if metadata and metadata.get('document_id') == document_id:
                    ids_to_delete.append(all_data['ids'][i])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"🗑️ Deleted {len(ids_to_delete)} chunks for document {document_id}")
                return True
            else:
                logger.warning(f"No chunks found for document {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset entire collection (for debugging)."""
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
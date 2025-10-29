"""
ChromaDB vector store manager with schema compatibility fixes.
"""
import os
import logging
import shutil
from typing import List, Tuple, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import config

logger = logging.getLogger(__name__)

class ChromaManager:
    """Manages ChromaDB vector store with persistent storage and schema recovery."""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.collection_name = config.collection_name
        
    def initialize(self):
        """Initialize ChromaDB client and collection with schema recovery."""
        try:
            # Initialize ChromaDB with persistent storage
            self.client = chromadb.PersistentClient(path=config.vector_store_path)
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(config.embedding.model_name)
            logger.info(f"Loaded embedding model: {config.embedding.model_name}")
            
            # Try to get or create collection with schema recovery
            self._initialize_collection()
                
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            # Try to recover from schema issues
            if "no such column" in str(e).lower():
                logger.warning("Schema compatibility issue detected, attempting recovery...")
                self._recover_from_schema_issue()
            else:
                raise
    
    def _initialize_collection(self):
        """Initialize collection with error handling."""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "DocuChat document embeddings"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                # Other error, re-raise
                raise
    
    def _recover_from_schema_issue(self):
        """Recover from ChromaDB schema compatibility issues."""
        logger.warning("Attempting to recover from ChromaDB schema issue...")
        
        try:
            # Try to reset the client
            self.client = None
            
            # Remove and recreate the vector store directory
            if os.path.exists(config.vector_store_path):
                shutil.rmtree(config.vector_store_path)
                logger.info("Removed corrupted vector store")
            
            os.makedirs(config.vector_store_path, exist_ok=True)
            
            # Reinitialize
            self.client = chromadb.PersistentClient(path=config.vector_store_path)
            self._initialize_collection()
            
            logger.info("✅ Successfully recovered from schema issue")
            
        except Exception as recovery_error:
            logger.error(f"Failed to recover from schema issue: {str(recovery_error)}")
            raise
    
    def add_documents(self, chunks: List[str], metadata: List[Dict], document_id: str):
        """
        Add document chunks to ChromaDB.
        """
        if not self.collection:
            self.initialize()

        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
            # Debug: log first chunk content
            if chunks:
                logger.debug(f"First chunk preview: {chunks[0][:100]}...")
        
            embeddings = self.embedding_model.encode(chunks).tolist()
            logger.info(f"Generated embeddings with shape: {len(embeddings)}x{len(embeddings[0]) if embeddings else 0}")
        
            # Generate unique IDs for each chunk
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        
            # Ensure all metadata has document_id
            for meta in metadata:
                meta["document_id"] = document_id
        
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata,
                ids=chunk_ids
            )
        
            logger.info(f"Added {len(chunks)} chunks from document {document_id}")
        
            # Verify the chunks were added
            count_after = self.collection.count()
            logger.info(f"Collection now has {count_after} total chunks")
        
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents with better relevance filtering.
        """
        if not self.collection:
            self.initialize()

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
    
            # Search for more results to allow for better filtering
            search_kwargs = {
                "query_embeddings": query_embedding,
                "n_results": top_k * 3,  # Get more results for better filtering
            }
        
            # Add filter if provided
            if filter_metadata:
                search_kwargs["where"] = filter_metadata
        
            # Perform search
            results = self.collection.query(**search_kwargs)
    
            # Format results with better filtering
            formatted_results = []
            seen_content = set()
        
            if results['documents'] and results['documents'][0]:
                for doc, distance, metadata in zip(
                    results['documents'][0],
                    results['distances'][0],
                    results['metadatas'][0]
                ):
                    # Skip very short chunks
                    if len(doc.strip()) < 50:
                        continue
                    
                    # Calculate similarity score (convert distance to similarity)
                    similarity_score = max(0.0, 1.0 - (distance / 2.0))
                
                    # LOWER THE THRESHOLD - from 0.3 to 0.2 to catch more results
                    if similarity_score > 0.2:  # Reduced threshold
                        # Simple deduplication
                        content_hash = hash(doc[:100])  # Check first 100 chars
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            formatted_results.append((doc, similarity_score, metadata or {}))
                
                    # Stop when we have enough high-quality results
                    if len(formatted_results) >= top_k:
                        break
        
            logger.info(f"Found {len(formatted_results)} relevant chunks for query: '{query}'")
        
            # Debug logging - show what we found
            if formatted_results:
                logger.debug(f"Top results for '{query}':")
                for i, (chunk, score, meta) in enumerate(formatted_results):
                    logger.debug(f"  {i+1}. score={score:.3f}, doc={meta.get('filename', 'unknown')}")
                    logger.debug(f"     preview: {chunk[:100]}...")
            else:
                logger.debug(f"No results found for query: '{query}'")
                # Try to understand why - check what the top results were before filtering
                if results['documents'] and results['documents'][0]:
                    logger.debug(f"Raw results before filtering:")
                    for i, (doc, distance) in enumerate(zip(results['documents'][0][:3], results['distances'][0][:3])):
                        raw_score = 1.0 - (distance / 2.0)
                        logger.debug(f"  {i+1}. raw_score={raw_score:.3f}, preview: {doc[:80]}...")
        
            return formatted_results
    
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            return []
    
    def delete_document(self, document_id: str):
        """Delete all chunks from a specific document."""
        if not self.collection:
            self.initialize()
        
        try:
            # Delete by document_id metadata filter
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted all chunks for document {document_id}")
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        if not self.collection:
            self.initialize()
        
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "vector_store_path": config.vector_store_path
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"total_chunks": 0, "collection_name": self.collection_name}
    
    def list_documents(self) -> List[Dict]:
        """List all unique documents in the collection."""
        if not self.collection:
            self.initialize()
        
        try:
            # Get all documents to extract unique document_ids
            all_results = self.collection.get()
            documents = {}
            
            if all_results['metadatas']:
                for metadata in all_results['metadatas']:
                    if metadata and 'document_id' in metadata:
                        doc_id = metadata['document_id']
                        if doc_id not in documents:
                            documents[doc_id] = {
                                'document_id': doc_id,
                                'filename': metadata.get('filename', 'unknown'),
                                'source': metadata.get('source', 'unknown'),
                                'chunk_count': 0
                            }
                        documents[doc_id]['chunk_count'] += 1
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def reset_collection(self):
        """Reset the entire collection (for testing)."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "DocuChat document embeddings"}
            )
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            # Try to recover
            self._recover_from_schema_issue()

    # Add to ChromaManager class in chroma_manager.py
    def debug_collection_contents(self, limit: int = 5):
        """Debug method to see what's actually in the collection."""
        if not self.collection:
            self.initialize()
    
        try:
            # Get all documents from the collection
            all_results = self.collection.get(limit=limit)
        
            debug_info = {
                "total_chunks": len(all_results['ids']) if all_results['ids'] else 0,
                "sample_chunks": []
            }
        
            if all_results['documents']:
                for i, (doc_id, document, metadata) in enumerate(zip(
                    all_results['ids'][:limit],
                    all_results['documents'][:limit], 
                    all_results['metadatas'][:limit]
                )):
                    debug_info["sample_chunks"].append({
                        "chunk_id": doc_id,
                        "content_preview": document[:200] + "..." if len(document) > 200 else document,
                        "content_length": len(document),
                        "metadata": metadata
                    })
        
            return debug_info
        
        except Exception as e:
            return {"error": str(e)}
"""
ChromaDB vector store manager for ChromaDB 1.2.2.
"""
import os
import logging
from typing import List, Tuple, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import config

logger = logging.getLogger(__name__)

class ChromaManager:
    """Manages ChromaDB vector store with persistent storage."""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.collection_name = "docuchat_documents"
        
    def initialize(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize ChromaDB with persistent storage
            self.client = chromadb.PersistentClient(path=config.vector_store_path)
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(config.embedding.model_name)
            logger.info(f"Loaded embedding model: {config.embedding.model_name}")
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "DocuChat document embeddings"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def add_documents(self, chunks: List[str], metadata: List[Dict], document_id: str):
        """
        Add document chunks to ChromaDB.
        
        Args:
            chunks: List of text chunks
            metadata: List of metadata dicts for each chunk
            document_id: Unique identifier for the document
        """
        if not self.collection:
            self.initialize()
        
        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # Generate unique IDs for each chunk
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Ensure all metadata has document_id
            for meta in metadata:
                meta["document_id"] = document_id
            
            # Add to collection - ChromaDB 1.2.2 uses different parameter names
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata,
                ids=chunk_ids
            )
            
            logger.info(f"✅ Added {len(chunks)} chunks from document {document_id}")
            
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
        
            # Search for more results initially, then filter
            initial_results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k * 2,  # Get more results for filtering
                where=filter_metadata
            )
        
            # Format results with better filtering
            formatted_results = []
            seen_content = set()
        
            if initial_results['documents'] and initial_results['documents'][0]:
                for i, (doc, distance, metadata) in enumerate(zip(
                    initial_results['documents'][0],
                    initial_results['distances'][0],
                    initial_results['metadatas'][0]
                )):
                    # Skip if we've seen very similar content
                    content_hash = hash(doc[:50])  # Check first 50 chars for duplicates
                    if content_hash in seen_content:
                        continue
                    seen_content.add(content_hash)
                
                    # Calculate similarity score
                    similarity_score = max(0.0, 1.0 - (distance / 2.0))
                
                    # Filter out low-confidence results more aggressively
                    if similarity_score > 0.3:  # Increased threshold
                        formatted_results.append((doc, similarity_score, metadata or {}))
                
                    # Stop when we have enough high-quality results
                    if len(formatted_results) >= top_k:
                        break
        
            logger.info(f"Found {len(formatted_results)} relevant chunks for query: '{query}' (after filtering)")
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
        if not self.client:
            self.initialize()
        
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "DocuChat document embeddings"}
            )
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise

def main():
    """Test the ChromaDB manager with version 1.2.2."""
    print("🧪 Testing ChromaDB Manager (v1.2.2)...")
    
    manager = ChromaManager()
    manager.initialize()
    
    # Test data
    test_chunks = [
        "Cloud computing is the delivery of computing services over the internet.",
        "There are three main service models: IaaS, PaaS, and SaaS.",
        "Deployment models include public, private, and hybrid clouds."
    ]
    
    test_metadata = [
        {"source": "test_doc.pdf", "chunk_id": 0, "filename": "test_doc.pdf"},
        {"source": "test_doc.pdf", "chunk_id": 1, "filename": "test_doc.pdf"},
        {"source": "test_doc.pdf", "chunk_id": 2, "filename": "test_doc.pdf"}
    ]
    
    # Test adding documents
    manager.add_documents(test_chunks, test_metadata, "test_document_001")
    
    # Test search
    results = manager.search("What are the cloud service models?")
    print(f"Search results: {len(results)}")
    for chunk, score, meta in results:
        print(f"Score: {score:.3f} - {chunk[:50]}...")
    
    # Test stats
    stats = manager.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Test listing documents
    documents = manager.list_documents()
    print(f"Documents in collection: {len(documents)}")
    
    # Test document deletion
    if documents:
        doc_id = documents[0]['document_id']
        manager.delete_document(doc_id)
        print(f"Deleted document: {doc_id}")

if __name__ == "__main__":
    main()
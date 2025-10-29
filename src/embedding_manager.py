# src/embedding_manager.py - SIMPLIFIED VERSION
"""
Embedding manager - simplified since ChromaDB handles embeddings.
"""
import logging
from typing import List, Tuple, Optional, Dict
from src.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages ChromaDB vector store operations."""
    
    def __init__(self):
        self.chroma_manager = ChromaManager()
        self.initialized = False
        
    def initialize(self):
        """Initialize the embedding manager."""
        try:
            # ChromaManager is self-initializing
            self.initialized = True
            logger.info("Embedding manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding manager: {str(e)}")
            raise
    
    def add_to_index(self, chunks: List[str], metadata: List[Dict], document_id: str):
        """Add documents to ChromaDB index."""
        self.chroma_manager.add_documents(chunks, metadata, document_id)
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Tuple[str, float, Dict]]:
        """Search for similar chunks."""
        return self.chroma_manager.search(query, top_k, filter_metadata)
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        return self.chroma_manager.get_collection_stats()
    
    def list_documents(self) -> List[Dict]:
        """List all documents in the vector store."""
        return self.chroma_manager.list_documents()
    
    def delete_document(self, document_id: str):
        """Delete a document from the vector store."""
        self.chroma_manager.delete_document(document_id)
    
    @property
    def index(self):
        """Check if index is available."""
        return self.chroma_manager.collection is not None
"""
Updated embedding manager using ChromaDB.
"""
import os
import logging
from typing import List, Tuple, Optional, Dict

from src.chroma_manager import ChromaManager
from src.config import config

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages text embeddings and ChromaDB vector store."""
    
    def __init__(self):
        self.chroma_manager = ChromaManager()
        self.embedding_model = None
        
    def initialize_model(self):
        """Initialize the embedding model and ChromaDB."""
        try:
            self.chroma_manager.initialize()
            logger.info("Embedding manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding manager: {str(e)}")
            raise
    
    def create_embeddings(self, chunks: List[str]) -> List:
        """
        Create embeddings for text chunks.
        Note: ChromaDB handles embedding internally, but we keep this for compatibility.
        """
        return chunks
    
    def create_index(self, embeddings: List, chunks: List[str], metadata: List[Dict] = None):
        """
        Create index in ChromaDB.
        For compatibility with existing code.
        """
        if metadata is None:
            metadata = [{} for _ in range(len(chunks))]
        
        # Generate a document ID
        import uuid
        document_id = str(uuid.uuid4())[:8]
        
        self.chroma_manager.add_documents(chunks, metadata, document_id)
    
    def add_to_index(self, chunks: List[str], metadata: List[Dict], document_id: str):
        """
        Add documents to ChromaDB index.
        
        Args:
            chunks: List of text chunks
            metadata: Metadata for each chunk
            document_id: Unique document identifier
        """
        self.chroma_manager.add_documents(chunks, metadata, document_id)
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (chunk_text, similarity_score, metadata) tuples
        """
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
    
    # For backward compatibility
    @property
    def index(self):
        """For backward compatibility with existing code."""
        return self.chroma_manager.collection is not None
    
    @property 
    def chunks(self):
        """For backward compatibility - not used with ChromaDB."""
        return []
    
    @property
    def metadata(self):
        """For backward compatibility - not used with ChromaDB."""
        return []

def main():
    """Test the updated embedding manager."""
    print("Testing Updated Embedding Manager...")
    
    manager = EmbeddingManager()
    manager.initialize_model()
    
    # Test search
    results = manager.search("cloud computing", top_k=2)
    print(f"Found {len(results)} results")
    
    # Test stats
    stats = manager.get_stats()
    print(f"Vector store stats: {stats}")

if __name__ == "__main__":
    main()
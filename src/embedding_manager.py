"""
Handles text embedding and vector storage using FAISS.
"""
import os
import logging
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss

from src.config import config

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages text embeddings and FAISS vector store."""
    
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.chunks = []  # Store original text chunks
        self.metadata = []  # Store metadata for each chunk
        
    def initialize_model(self):
        """Initialize the embedding model."""
        try:
            logger.info(f"Loading embedding model: {config.embedding.model_name}")
            self.embedding_model = SentenceTransformer(config.embedding.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Create embeddings for text chunks.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            numpy array of embeddings
        """
        if self.embedding_model is None:
            self.initialize_model()
        
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def create_index(self, embeddings: np.ndarray, chunks: List[str], metadata: List[dict] = None):
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of embeddings
            chunks: Original text chunks
            metadata: Optional metadata for each chunk
        """
        if metadata is None:
            metadata = [{} for _ in range(len(chunks))]
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks = chunks
        self.metadata = metadata
        
        logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
    
    def save_index(self, file_path: str = None):
        """Save the FAISS index and associated data to disk."""
        if file_path is None:
            file_path = os.path.join(config.vector_store_path, "faiss_index")
        
        if self.index is None:
            logger.warning("No index to save")
            return
        
        # Save FAISS index
        faiss.write_index(self.index, file_path + ".index")
        
        # Save chunks and metadata (simplified - in production use proper serialization)
        import pickle
        with open(file_path + "_data.pkl", 'wb') as f:
            pickle.dump({'chunks': self.chunks, 'metadata': self.metadata}, f)
        
        logger.info(f"Saved index to {file_path}.index")
    
    def load_index(self, file_path: str = None):
        """Load FAISS index and associated data from disk."""
        if file_path is None:
            file_path = os.path.join(config.vector_store_path, "faiss_index")
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(file_path + ".index")
            
            # Load chunks and metadata
            import pickle
            with open(file_path + "_data.pkl", 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search for similar chunks to the query.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_text, similarity_score, metadata) tuples
        """
        if self.index is None or self.embedding_model is None:
            raise ValueError("Index or embedding model not initialized")
        
        # Create embedding for query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                results.append((
                    self.chunks[idx],
                    float(score),
                    self.metadata[idx] if idx < len(self.metadata) else {}
                ))
        
        return results

def main():
    """Test the embedding manager with a simple example."""
    import tempfile
    
    # Create test chunks
    test_chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Vector databases store embeddings for similarity search.",
        "RAG systems combine retrieval and generation for better answers."
    ]
    
    # Initialize and test
    manager = EmbeddingManager()
    embeddings = manager.create_embeddings(test_chunks)
    manager.create_index(embeddings, test_chunks)
    
    # Test search
    query = "What is neural networks?"
    results = manager.search(query, top_k=2)
    
    print(f"Query: '{query}'")
    print("Top results:")
    for i, (chunk, score, metadata) in enumerate(results):
        print(f"{i+1}. Score: {score:.4f}")
        print(f"   Text: {chunk}")
        print()

if __name__ == "__main__":
    main()
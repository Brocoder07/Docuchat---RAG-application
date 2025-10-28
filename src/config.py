"""
Configuration settings for the DocuChat RAG system.
"""
import os
from dataclasses import dataclass
from typing import List

# Calculate paths once at module level
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DOCS_DIR = os.path.join(DATA_DIR, "raw_documents")
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store")

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 800  # characters per chunk
    chunk_overlap: int = 100  # overlap between chunks

@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32

@dataclass
class ChromaConfig:
    """Configuration for ChromaDB."""
    collection_name: str = "docuchat_documents"
    persist_directory: str = VECTOR_STORE_PATH

@dataclass
class Config:
    """Main configuration class."""
    # Paths (now using module-level constants)
    project_root: str = PROJECT_ROOT
    data_dir: str = DATA_DIR
    raw_docs_dir: str = RAW_DOCS_DIR
    vector_store_path: str = VECTOR_STORE_PATH
    
    # Processing configs
    chunking: ChunkingConfig = None
    embedding: EmbeddingConfig = None
    chroma: ChromaConfig = None
    
    # Supported file types
    supported_extensions: List[str] = None
    
    def __post_init__(self):
        # Initialize with default instances if None
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.chroma is None:
            self.chroma = ChromaConfig()
        if self.supported_extensions is None:
            self.supported_extensions = ['.pdf', '.txt', '.docx', '.md', '.xlsx']
        
        # Create necessary directories
        os.makedirs(self.raw_docs_dir, exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)

# Global configuration instance
config = Config()
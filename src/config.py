# src/config.py - FIXED VERSION
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class OllamaConfig:
    """Ollama configuration."""
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b-instruct-q4_1")
    timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "30"))
    max_retries: int = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

@dataclass 
class APIConfig:
    """API configuration."""
    host: str = os.getenv("API_HOST", "127.0.0.1")
    port: int = int(os.getenv("API_PORT", "8000"))
    reload: bool = os.getenv("API_RELOAD", "true").lower() == "true"
    workers: int = int(os.getenv("API_WORKERS", "1"))

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

@dataclass
class EvaluationConfig:
    """Evaluation configuration for Week 2."""
    enable_evaluation: bool = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"
    evaluation_threshold: float = float(os.getenv("EVALUATION_THRESHOLD", "0.5"))
    track_performance: bool = os.getenv("TRACK_PERFORMANCE", "true").lower() == "true"

@dataclass
class EmbeddingConfig:
    """Embedding configuration - ADDED THIS CLASS"""
    model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    device: str = "cpu"
    batch_size: int = 32

@dataclass
class ChunkingConfig:
    """Chunking configuration."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    experimental_sizes: List[int] = field(default_factory=lambda: [500, 800, 1000, 1200])  # FIXED: Use field for mutable default

@dataclass
class Config:
    """Enhanced main configuration class."""
    # Paths
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(project_root, "data")
    raw_docs_dir: str = os.path.join(data_dir, "raw_documents")
    vector_store_path: str = os.path.join(data_dir, "vector_store")
    
    # Component configs
    ollama: OllamaConfig = None
    api: APIConfig = None
    logging: LoggingConfig = None
    evaluation: EvaluationConfig = None
    embedding: EmbeddingConfig = None  # ADDED THIS
    chunking: ChunkingConfig = None
    
    # ChromaDB
    collection_name: str = os.getenv("CHROMA_COLLECTION", "docuchat_documents")
    
    # Performance settings
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB default
    
    # Supported file types
    supported_extensions: List[str] = None
    
    def __post_init__(self):
        # Initialize component configs
        if self.ollama is None:
            self.ollama = OllamaConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.embedding is None:  # ADDED THIS
            self.embedding = EmbeddingConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.supported_extensions is None:
            self.supported_extensions = ['.pdf', '.txt', '.docx', '.md', '.xlsx']
        
        # Create necessary directories
        os.makedirs(self.raw_docs_dir, exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('docuchat.log')
            ]
        )

# Global configuration instance
config = Config()
"""
Unified configuration with Firebase Auth.
FIXED: All hardcoded secrets removed. Loading from environment variables.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging
from dotenv import load_dotenv
import json # 🚨 Import json

logger = logging.getLogger(__name__)
load_dotenv() 

@dataclass
class FirebaseConfig:
    """Firebase configuration."""
    SERVICE_ACCOUNT_KEY_PATH: str = "serviceAccountKey.json"

    FIREBASE_WEB_CONFIG: Dict = field(default_factory=lambda:
        json.loads(os.getenv("FIREBASE_WEB_CONFIG_JSON", "{}"))
    )

@dataclass
class ModelConfig:
    """Groq model configuration."""
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    MODEL: str = "llama-3.1-8b-instant"
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 1024
    TOP_P: float = 0.9

    def __post_init__(self):
        if not self.GROQ_API_KEY:
            logger.warning("❌ GROQ_API_KEY not found in environment variables")

@dataclass
class RAGConfig:
    """RAG pipeline configuration."""
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.3
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DEVICE: str = "cpu"

@dataclass
class APIConfig:
    """API configuration."""
    HOST: str = os.getenv("API_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"
    ALLOW_ORIGINS: List[str] = None

    def __post_init__(self):
        if self.ALLOW_ORIGINS is None:
            self.ALLOW_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]

@dataclass
class FileConfig:
    """File processing configuration."""
    ALLOWED_EXTENSIONS: List[str] = None
    MAX_FILE_SIZE_MB: int = 50
    UPLOAD_DIR: str = "data/uploads"

    def __post_init__(self):
        if self.ALLOWED_EXTENSIONS is None:
            self.ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.md']
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

@dataclass
class LoggingConfig:
    """Logging configuration."""
    LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def setup_logging(self):
        """Configure logging globally."""
        logging.basicConfig(
            level=getattr(logging, self.LEVEL.upper()),
            format=self.FORMAT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('docuchat.log', encoding='utf-8')
            ]
        )
        logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)

class Config:
    """
    Main configuration class following Singleton pattern.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize all configuration sections."""
        self.firebase = FirebaseConfig()
        self.model = ModelConfig()
        self.rag = RAGConfig()
        self.api = APIConfig()
        self.files = FileConfig()
        self.logging = LoggingConfig()

        self.logging.setup_logging()
        self._validate_config()

    def _validate_config(self):
        """Validate critical configuration values."""
        if not self.model.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required. Please set it in .env file")

        if self.rag.CHUNK_SIZE <= self.rag.CHUNK_OVERLAP:
            raise ValueError("Chunk size must be greater than chunk overlap")

        if not os.path.exists(self.firebase.SERVICE_ACCOUNT_KEY_PATH):
            raise FileNotFoundError(f"Firebase service account key not found at: {self.firebase.SERVICE_ACCOUNT_KEY_PATH}")

        if not self.firebase.FIREBASE_WEB_CONFIG:
            raise ValueError("FIREBASE_WEB_CONFIG_JSON is missing or empty in your .env file.")

        logging.info("✅ Configuration validated and loaded successfully")

# Global config instance
config = Config()
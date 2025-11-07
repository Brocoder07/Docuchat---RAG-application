"""
Unified configuration with Firebase Auth.
Made validation tolerant in dev; added ENV flags and admin list.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging
from dotenv import load_dotenv
import json

logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class FirebaseConfig:
    """Firebase configuration."""
    SERVICE_ACCOUNT_KEY_PATH: str = os.getenv("SERVICE_ACCOUNT_KEY_PATH", "serviceAccountKey.json")

    FIREBASE_WEB_CONFIG: Dict = field(default_factory=lambda:
        json.loads(os.getenv("FIREBASE_WEB_CONFIG_JSON", "{}"))
    )

@dataclass
class ModelConfig:
    """Groq model configuration."""
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
    MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
    TOP_P: float = float(os.getenv("GROQ_TOP_P", "0.9"))

    def __post_init__(self):
        if not self.GROQ_API_KEY:
            logger.warning("❌ GROQ_API_KEY not found in environment variables")

@dataclass
class RAGConfig:
    """RAG pipeline configuration."""
    # Interpreted as tokens when using tokenizer-based chunker
    CHUNK_SIZE: int = 500           # smaller chunk size for resumes
    CHUNK_OVERLAP: int = 100       # overlap to preserve context
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.3
    
    # -----------------------------------------------------------------
    # 🚨 UPDATED CONFIG FOR PROMPT EXPERIMENTATION
    # -----------------------------------------------------------------
    # Name of the prompt template to use from core/prompts.py
    # Options: "STRICT_CONTEXT_V1", "FRIENDLY_V1", "BALANCED_CONTEXT_V1", 
    #          "HYBRID_FRIENDLY_BALANCED_V1", "CREATIVE_GROUNDED_V1"
    RAG_PROMPT_TEMPLATE: str = os.getenv("RAG_PROMPT_TEMPLATE", "STRICT_CONTEXT_V1")
    # -----------------------------------------------------------------

    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DEVICE: str = "cpu"

    def __post_init__(self):
        if self.CHUNK_SIZE <= self.CHUNK_OVERLAP:
            logger.warning("RAGConfig: CHUNK_SIZE <= CHUNK_OVERLAP; adjusting overlap")
            self.CHUNK_OVERLAP = max(0, self.CHUNK_SIZE // 5)
            
        # Validate that the chosen prompt exists
        try:
            from core.prompts import PROMPT_REGISTRY
            if self.RAG_PROMPT_TEMPLATE not in PROMPT_REGISTRY:
                logger.error(f"❌ RAG_PROMPT_TEMPLATE '{self.RAG_PROMPT_TEMPLATE}' not found in PROMPT_REGISTRY!")
                raise ValueError("Invalid RAG_PROMPT_TEMPLATE configured")
        except ImportError:
            # This might happen on initial setup, just log a warning
            logger.warning("Could not import PROMPT_REGISTRY for validation yet.")
        except Exception as e:
            logger.error(f"Error validating RAG_PROMPT_TEMPLATE: {e}")
            raise

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
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")

    def __post_init__(self):
        if self.ALLOWED_EXTENSIONS is None:
            # streamlit wants extensions without leading dots sometimes; keep dots for backend checks.
            self.ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.md']
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

@dataclass
class LoggingConfig:
    """Logging configuration."""
    LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    LOG_FILE: str = os.path.join(os.getenv("LOG_DIR", "logs"), "docuchat.log")

    def setup_logging(self):
        """Configure logging globally."""
        os.makedirs(self.LOG_DIR, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.LEVEL.upper()),
            format=self.FORMAT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.LOG_FILE, encoding='utf-8')
            ]
        )
        # Silence chromadb telemetry if present
        try:
            import chromadb
            logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
        except Exception:
            pass

class Config:
    """
    Main configuration class following Singleton pattern.
    Validation is tolerant in non-production.
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
        # RAGConfig must be initialized after prompts.py is available
        # so we do this manually.
        
        self.api = APIConfig()
        self.files = FileConfig()
        self.logging = LoggingConfig()
        
        # Setup logging first
        self.logging.setup_logging()
        
        # Now initialize RAGConfig, which depends on core.prompts
        self.rag = RAGConfig()
        
        # Environment flags
        self.ENV = os.getenv("ENV", "development")  # "production" or "development"
        # When true, require firebase to initialize successfully at startup
        self.REQUIRE_FIREBASE = os.getenv("REQUIRE_FIREBASE", "false").lower() == "true"
        # Admin UIDs (comma separated) allowed for sensitive operations
        admin_uids = os.getenv("ADMIN_USER_IDS", "")
        self.ADMIN_USER_IDS = [x.strip() for x in admin_uids.split(",") if x.strip()]

        self._validate_config()

    def _validate_config(self):
        """Validate critical configuration values; be forgiving in development."""
        # GROQ API key
        if not self.model.GROQ_API_KEY:
            msg = "GROQ_API_KEY is not set in environment"
            if self.ENV == "production":
                raise ValueError(msg)
            else:
                logging.warning(msg + " — continuing in development mode")

        # Chunk size/overlap check (token-based expectation)
        if self.rag.CHUNK_SIZE <= self.rag.CHUNK_OVERLAP:
            raise ValueError("Chunk size must be greater than chunk overlap")

        # Firebase key existence only required when REQUIRE_FIREBASE True
        if self.REQUIRE_FIREBASE:
            if not os.path.exists(self.firebase.SERVICE_ACCOUNT_KEY_PATH):
                raise FileNotFoundError(f"Firebase service account key not found at: {self.firebase.SERVICE_ACCOUNT_KEY_PATH}")
            if not self.firebase.FIREBASE_WEB_CONFIG:
                raise ValueError("FIREBASE_WEB_CONFIG_JSON is missing or empty in your .env file.")

        logging.info("✅ Configuration validated and loaded successfully")

# Global config instance
config = Config()
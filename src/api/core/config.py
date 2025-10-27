"""
API configuration settings.
"""
import os
from typing import List

class APIConfig:
    """API configuration."""
    
    # FastAPI settings
    TITLE = "DocuChat API"
    DESCRIPTION = "FREE RAG-based Document Q&A System"
    VERSION = "1.0.0"
    
    # CORS settings
    ALLOW_ORIGINS = ["*"]  # For development
    ALLOW_CREDENTIALS = True
    ALLOW_METHODS = ["*"]
    ALLOW_HEADERS = ["*"]
    
    # File upload settings
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.md', '.xlsx']
    UPLOAD_DIR = "data/raw_documents"
    
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)

api_config = APIConfig()
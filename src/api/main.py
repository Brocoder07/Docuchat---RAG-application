"""
Main FastAPI application.
"""
from fastapi import FastAPI
import logging
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import debug
from src.api.core.config import api_config
from src.api.routes import health, documents, chat
from src.api.services.rag_service import rag_service

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=api_config.TITLE,
    description=api_config.DESCRIPTION,
    version=api_config.VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.ALLOW_ORIGINS,
    allow_credentials=api_config.ALLOW_CREDENTIALS,
    allow_methods=api_config.ALLOW_METHODS,
    allow_headers=api_config.ALLOW_HEADERS,
)

# Include routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(debug.router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to DocuChat - AI Document Q&A System",
        "version": api_config.VERSION,
        "cost": "$0.00",
        "features": [
            "Document upload (PDF, TXT, DOCX, MD, XLSX)",
            "AI-powered Q&A with Ollama LLM",
            "Local processing - 100% private",
            "Fast and intelligent answers"
        ]
    }

# Add proper error handling for startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        rag_service.initialize()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {str(e)}")
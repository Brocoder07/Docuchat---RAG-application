"""
FastAPI application with production-ready configuration.
FIXED: Disabled auto-reload to prevent recursive file watching
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html

from core.config import config
from api.routes import router

# Configure logging
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown."""
    # Startup
    logger.info("🚀 Starting DocuChat API...")
    
    try:
        from core.rag_pipeline import rag_pipeline
        
        # Initialize RAG pipeline with timeout
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor() as executor:
            future = executor.submit(rag_pipeline.initialize)
            try:
                # -----------------------------------------------------------------
                # 🚨 FIXED: Increased timeout to 90s for larger embedding model
                # -----------------------------------------------------------------
                initialized = future.result(timeout=90)  # 90 second timeout
                # -----------------------------------------------------------------
                
                if initialized:
                    logger.info("✅ RAG Pipeline initialized successfully")
                else:
                    logger.error("❌ RAG Pipeline initialization failed")
                    # Don't crash, but log heavily
            except TimeoutError:
                logger.error("❌ RAG Pipeline initialization timed out")
            except Exception as e:
                logger.error(f"❌ RAG Pipeline initialization error: {str(e)}")
                
    except Exception as e:
        logger.error(f"❌ Startup initialization failed: {str(e)}")
    
    yield  # App runs here
    
    # Shutdown
    logger.info("🛑 Shutting down DocuChat API...")

def create_application() -> FastAPI:
    """Application factory following best practices."""
    application = FastAPI(
        title="DocuChat API",
        description="AI-powered Document Q&A System with RAG Pipeline",
        version="2.0.0",
        docs_url=None,
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    application.include_router(router, prefix="/api/v1")
    
    return application

# Create application instance
app = create_application()

# Custom documentation endpoint
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with better styling."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="DocuChat API Documentation",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to DocuChat API",
        "version": "2.0.0",
        "description": "AI-powered Document Q&A System",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# Exception handlers (keep your existing ones)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.api.HOST,
        port=config.api.PORT,
        reload=False,  # 🚨 CRITICAL FIX: Disabled auto-reload
        log_level="info"
    )
"""
FastAPI application with production-ready configuration.
Now uses Firebase Admin for auth dependency.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html

from core.config import config
from api.routes import router as v1_router # 🚨 Renamed
# 🚨 NO auth_router, NO database imports

# Configure logging
logger = logging.getLogger(__name__)

# 🚨 NOTE: No need to create DB tables here anymore

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
                initialized = future.result(timeout=90)
                if initialized:
                    logger.info("✅ RAG Pipeline initialized successfully")
                else:
                    logger.error("❌ RAG Pipeline initialization failed")
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
    
    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # -----------------------------------------------------------------
    # 🚨 MODIFIED: Only one router
    # -----------------------------------------------------------------
    application.include_router(v1_router, prefix="/api/v1", tags=["RAG Pipeline"])
    # -----------------------------------------------------------------
    
    return application

# Create application instance
app = create_application()

# ... (keep /docs and / root endpoints) ...
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with better styling."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="DocuChat API Documentation",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.api.HOST,
        port=config.api.PORT,
        reload=False,
        log_level="info"
    )
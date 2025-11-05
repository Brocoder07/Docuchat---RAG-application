"""
FastAPI application with production-ready configuration and lazy Firebase init.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html

from core.config import config
from api.routes import router as v1_router
from core.firebase import init_firebase

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown."""
    logger.info("🚀 Starting DocuChat API...")

    # Initialize Firebase lazily if configured or required
    try:
        firebase_ok = init_firebase()
        if not firebase_ok:
            logger.warning("Firebase initialization failed or not configured. Auth endpoints will be disabled.")
    except Exception as e:
        logger.warning(f"Firebase initialization exception: {e}")

    # Initialize RAG pipeline in a thread to avoid blocking event loop
    try:
        from core.rag_pipeline import rag_pipeline
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(rag_pipeline.initialize)
            try:
                initialized = future.result(timeout=90)
                if initialized:
                    logger.info("✅ RAG Pipeline initialized successfully")
                else:
                    logger.error("❌ RAG Pipeline initialization failed")
            except Exception as e:
                logger.error(f"❌ RAG Pipeline initialization error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Startup initialization failed: {str(e)}")

    yield

    logger.info("🛑 Shutting down DocuChat API...")

def create_application() -> FastAPI:
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

    application.include_router(v1_router, prefix="/api/v1", tags=["RAG Pipeline"])
    return application

app = create_application()

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="DocuChat API Documentation",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

@app.get("/")
async def root():
    return {"message": "Welcome to DocuChat API", "version": "2.0.0", "docs": "/docs", "health": "/api/v1/health"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=config.api.HOST, port=config.api.PORT, reload=False, log_level="info")
"""
Enhanced error handling and performance monitoring utilities.
"""
import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

class RAGError(Exception):
    """Base exception for RAG pipeline errors."""
    pass

class DocumentProcessingError(RAGError):
    """Document processing related errors."""
    pass

class EmbeddingError(RAGError):
    """Embedding related errors."""
    pass

class LLMError(RAGError):
    """LLM related errors."""
    pass

class VectorStoreError(RAGError):
    """Vector store related errors."""
    pass

def handle_rag_errors(func: Callable) -> Callable:
    """Decorator to handle RAG pipeline errors consistently."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except DocumentProcessingError as e:
            logger.error(f"Document processing error in {func.__name__}: {str(e)}")
            raise
        except EmbeddingError as e:
            logger.error(f"Embedding error in {func.__name__}: {str(e)}")
            raise
        except LLMError as e:
            logger.error(f"LLM error in {func.__name__}: {str(e)}")
            raise
        except VectorStoreError as e:
            logger.error(f"Vector store error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise RAGError(f"Pipeline error: {str(e)}")
    return wrapper

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log execution time for performance monitoring."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry failed operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. Retrying...")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator
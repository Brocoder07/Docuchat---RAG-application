# src/embeddings_factory.py
"""
Provides a unified embedding interface for the RAG app.
Supports: SentenceTransformers (default) and can be extended for OpenAI later.
"""

from typing import List, Callable
import os

def get_embedder() -> Callable[[List[str]], List[List[float]]]:
    """
    Returns a function: embed(texts: List[str]) -> List[List[float]]
    Uses SentenceTransformer('all-MiniLM-L6-v2') by default.
    """
    backend = os.environ.get("EMBEDDING_BACKEND", "sentence_transformers").lower()

    if backend == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "SentenceTransformer not found. Run: pip install sentence-transformers"
            ) from e

        model = SentenceTransformer("all-MiniLM-L6-v2")

        def embed(texts: List[str]) -> List[List[float]]:
            """Compute embeddings using SentenceTransformer."""
            return model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            ).tolist()

        return embed

    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")

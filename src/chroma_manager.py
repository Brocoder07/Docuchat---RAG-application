# src/chroma_manager.py
import logging
from typing import List, Optional, Tuple, Dict, Any
from src.embeddings_factory import get_embedder
logger = logging.getLogger(__name__)

embed = get_embedder()

class ChromaManager:
    """
    Lightweight wrapper around a Chroma collection for embeddings + retrieval.

    This class will attempt to create a chromadb.Client() automatically if no client
    is provided. That makes initialization easier for callers that just want the default.
    """

    def __init__(self, client: Optional[Any] = None, collection_name: str = "docuchat_documents"):
        self.client = client
        self.collection_name = collection_name
        self.collection = None
        self._init_client_and_collection()

    def _init_client_and_collection(self):
        # If caller didn't provide a client, try to create one.
        if self.client is None:
            try:
                import chromadb
                # default client creation - adapt if your environment needs different args
                self.client = chromadb.Client()
                logger.info("Created chromadb.Client() automatically.")
            except Exception:
                logger.exception("chromadb client not available or failed to create. Pass a client explicitly to ChromaManager.")
                self.client = None

        if self.client is None:
            # can't proceed without a client; collection stays None
            logger.warning("ChromaManager created without a valid client; collection will not be initialized.")
            return

        try:
            # Try to get collection; adapt to your Chroma client accordingly
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info("Loaded existing collection: %s", self.collection_name)
            except Exception:
                # If collection doesn't exist, create it
                self.collection = self.client.create_collection(self.collection_name)
                logger.info("Created new collection: %s", self.collection_name)
        except Exception as e:
            logger.exception("Failed to initialize Chroma collection: %s", str(e))
            # keep self.collection as None and let callers handle errors

    def add_documents(self, docs: List[Dict[str, Any]]):
        """
        Add documents to the collection.
        docs: list of dicts with keys: 'id', 'text', 'metadata' (optional)
        """
        if not self.collection:
            raise RuntimeError("Chroma collection not initialized. Provide a client or ensure chromadb is installed.")
        try:
            ids = [d["id"] for d in docs]
            texts = [d["text"] for d in docs]
            metas = [d.get("metadata", {}) for d in docs]
            # Some Chroma clients use .add; adapt depending on client version
            if hasattr(self.collection, "add"):
                # common form: add(documents=texts, ids=ids, metadatas=metas)
                self.collection.add(documents=texts, ids=ids, metadatas=metas)
            else:
                # attempt generic ingestion fallback
                self.collection.insert(documents=texts, ids=ids, metadatas=metas)
            logger.info("Added %d documents to collection %s", len(docs), self.collection_name)
        except Exception:
            logger.exception("Error adding documents to Chroma collection.")
            raise

    def search(self, query: str, top_k: int = 5, filter: Optional[dict] = None) -> List[Tuple[str, Optional[float], Dict]]:
        """
        Search the chroma collection for the query.
        Default top_k increased from 3 -> 5 to improve recall for lecture / note documents.

        Returns:
            List of tuples: (text, distance_score_or_none, metadata_dict)
        """
        parsed: List[Tuple[str, Optional[float], Dict]] = []
        if not self.collection:
            logger.warning("ChromaManager.search called but no collection is initialized; returning empty list.")
            return parsed

        try:
            # Many chroma clients support query(query_texts=[...], n_results=k, where=filter)
            results = self.collection.query(query_texts=[query], n_results=top_k, where=filter)
        except TypeError:
            # Some clients use slightly different arg names
            try:
                results = self.collection.query([query], n_results=top_k, where=filter)
            except Exception as e:
                logger.exception("Chroma query failed: %s", str(e))
                return []
        except Exception as e:
            logger.exception("Chroma query failed: %s", str(e))
            return []

        # Try a few parsing strategies tolerant to client versions
        try:
            # Common newer shape: results['results'][0]['documents'], ['distances'], ['metadatas']
            res0 = results.get("results", [results])[0]
            docs = res0.get("documents", [])
            distances = res0.get("distances", []) or res0.get("scores", [])
            metadatas = res0.get("metadatas", []) or []
            for i, text in enumerate(docs):
                score = None
                if distances and i < len(distances):
                    score = distances[i]
                meta = metadatas[i] if i < len(metadatas) else {}
                parsed.append((text, score, meta))
        except Exception:
            # Fallback parsing for alternate response shapes
            try:
                docs = results.get("documents", [])
                metas = results.get("metadatas", [])
                dists = results.get("distances", []) or results.get("scores", [])
                for i, text in enumerate(docs):
                    meta = metas[i] if i < len(metas) else {}
                    score = dists[i] if i < len(dists) else None
                    parsed.append((text, score, meta))
            except Exception:
                # Last-resort: attempt to iterate as if results is a list of strings
                try:
                    for i, item in enumerate(results or []):
                        parsed.append((str(item), None, {}))
                except Exception:
                    logger.exception("Unable to parse results from chroma query response.")

        # Debug log retrieved chunk snippets
        logger.debug("ChromaManager.search retrieved %d entries for query: %s", len(parsed), query)
        for i, (text, score, meta) in enumerate(parsed):
            snippet = (text[:200] + "...") if isinstance(text, str) and len(text) > 200 else text
            logger.debug("  - chunk %d score=%s meta=%s snippet=%s", i + 1, repr(score), repr(meta), repr(snippet))

        return parsed

    def get_all_chunks(self):
        """
        Safer retrieval using explicit local embeddings for queries, avoiding chroma's ONNX auto-download.
        """
        try:
            # simple empty query text -> we can instead request all docs, but to use similarity we need embeddings
            # Use a dummy empty string embedding if you want top-k for empty query.
            query_texts = [""]
            # compute embeddings locally
            query_embeddings = embed(query_texts)  # list of lists
            results = self.collection.query(query_embeddings=query_embeddings, n_results=1000)
            # parse results into docs list as before
            docs = []
            docs_list = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else [None]*len(docs_list)
            ids = results.get("ids", [[]])[0] if results.get("ids") else [None]*len(docs_list)
            for i, text in enumerate(docs_list):
                docs.append({"id": ids[i] if i < len(ids) else None,
                            "document": text,
                            "metadata": metadatas[i] if i < len(metadatas) else None})
            return docs
        except Exception as e:
            logger.exception("Query with local embeddings failed: %s", e)
            # fallback to collection.get()
            try:
                fallback = self.collection.get()
                docs = []
                docs_list = fallback.get("documents", [])
                metadatas = fallback.get("metadatas", [])
                ids = fallback.get("ids", [])
                for i, text in enumerate(docs_list):
                    docs.append({"id": ids[i] if i < len(ids) else None,
                                "document": text,
                                "metadata": metadatas[i] if i < len(metadatas) else None})
                return docs
            except Exception as e2:
                logger.exception("Fallback collection.get() also failed: %s", e2)
                return []
## Quick context for AI coding agents

This repository implements DocuChat — a RAG (Retrieval-Augmented Generation) app that
lets users upload documents, stores embeddings in a local ChromaDB vector store,
and answers queries using a small local LLM (Ollama) with multi-stage fallbacks.

Keep suggestions focused, executable, and specific to files referenced below.

## Big-picture architecture
- Ingestion & chunking: `src/document_processor.py` — extracts text and creates overlapping chunks (see `src/config.py` for chunk sizes).
- Embeddings & vector store: `src/embedding_manager.py` → wraps ChromaDB logic (see `src/chroma_manager.py`). Persistent store directory: `data/vector_store`.
- RAG orchestration: `src/rag_pipeline.py` — coordinates processing, searching and source-tracking. Use this when changing query flows or metadata.
- LLM layer: `src/llm_integration.py` — prefers Ollama at `http://localhost:11434` with model `llama3.2:1b-instruct-q4_1`. Has multi-stage: direct LLM, synthesis, and fallback.
- API: `src/api` (entrypoint: `src/api.py` which runs Uvicorn with `src.api.main:app`).
- UI: Streamlit front-end at `src/frontend/app.py` which talks to the backend API via `src/frontend/services/api_client.py`.

Design rationale (from code): prefer small local models (Ollama) for privacy and latency, keep embeddings local (ChromaDB persisted), and implement conservative LLM prompts + multi-stage fallback to reduce hallucination.

## Key developer workflows & commands
- Install runtime deps listed in `requirements.txt` (Python 3.10+ recommended).
- Start backend API (development):
  - `python -m src.api` (this runs uvicorn configured in `src/api.py`)
  - Equivalent uvicorn invocation: `uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000`
- Start Streamlit UI (from repo root):
  - `streamlit run src/frontend/app.py`
- Quick component smoke-tests (no test framework required): several core modules include `if __name__ == '__main__'` tests, e.g. run:
  - `python -m src.rag_pipeline` or `python -m src.llm_integration` to exercise local code paths.

Notes: The project expects ChromaDB files under `data/vector_store` (config: `src/config.py`). A pre-existing sqlite file may exist under `streamlit/data/vector_store/chroma.sqlite3` in this workspace — verify `config.vector_store_path` if you change locations.

## Project-specific patterns & conventions (do not invent omissions)
- Configuration: use the single global `config` in `src/config.py`. Prefer reading/writing paths there rather than hardcoding.
- Embedding model: code uses SentenceTransformers via `config.embedding.model_name`. Keep changes to model name in `config`.
- Chroma v1.2.2 API conventions are used (see `src/chroma_manager.py`); when updating Chroma, check the API differences (`collection.query` keys, `get()` shape, etc.).
- LLM behavior: `src/llm_integration.py` intentionally enforces strict prompts and multi-stage fallbacks — edits must preserve those safety checks and the simple in-memory `answer_cache` logic.
- Source tracking: `src/rag_pipeline.py` builds metadata with `document_id`, `chunk_id`, `filename`, etc. Keep those keys stable if you change the vector schema (UI expects them).

## Integration points & external dependencies
- Ollama local service expected at `http://localhost:11434`. If not present, code falls back to rule-based synthesis. Keep this in mind when modifying `LLMIntegration.initialize`.
- ChromaDB persistent client uses the path `config.vector_store_path`. Backup or migration changes must maintain metadata keys used in `RAGPipeline._extract_source_info`.
- The frontend calls backend endpoints via `src/frontend/services/api_client.py` — when adding new endpoints, update that client and the Streamlit components under `src/frontend/components`.

## What to modify when adding features
- Adding new metadata fields: update `RAGPipeline.process_document` to populate them and adjust `src/chroma_manager.py` indexing and `RAGPipeline._extract_source_info` for display.
- Changing chunking: update `src/config.py` chunking defaults and `src/document_processor.py` chunking implementation together.
- Replacing LLM provider: keep the multi-stage pipeline shape from `src/llm_integration.py` (stage1→stage2→stage3). Implement a new `_call_<provider>_with_retry` and toggle in `initialize` with a clear fallback.

## Quick examples to reference in PRs
- To show you're familiar with the project, reference exact files and lines: e.g. "I updated source tracking in `src/rag_pipeline.py::_extract_source_info` to include `metadata['document_name']` and adapted `src/frontend/components/chat_area.py` to display `primary_source`."
- For LLM changes, point to `src/llm_integration.py::_build_dynamic_prompt` to explain prompt-preserving changes.

## Tests & validation
- There is no configured pytest entry in `requirements.txt`. Quick validation steps used by developers in this repo:
  1. Run `python -m src.chroma_manager` and `python -m src.rag_pipeline` to exercise collection and pipeline flows.
  2. Start backend (`python -m src.api`) and then Streamlit UI (`streamlit run src/frontend/app.py`) and interact with the UI.
  3. Verify `data/vector_store` files are created and that `RAGPipeline.query(...)` returns `source_info` with expected keys.

## When in doubt — focus edits on these invariants
- Keep metadata keys: `document_id`, `chunk_id`, `filename`, `source`.
- Ensure LLM path either uses Ollama or preserves the strict prompt + validation fallback behaviour.
- Preserve persistent Chroma collection name `docuchat_documents` or record the migration steps.

If anything above is unclear or you want me to expand examples (e.g., a checklist for PR reviewers or small unit tests to add), tell me which area to expand. I can iterate the instructions quickly.

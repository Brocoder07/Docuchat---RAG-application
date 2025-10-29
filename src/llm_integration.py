# src/llm_integration.py
import logging
import requests
import time
import traceback
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class LLMIntegration:
    """
    Integration helper for Ollama-based LLMs.

    - _call_ollama_api(question, relevant_chunks=None):
        - If relevant_chunks is None -> treat 'question' as a full prompt and call Ollama.
        - If relevant_chunks is provided -> build prompt from question + chunks and call Ollama.
    """

    def __init__(self, model_name: str = "llama3.2:1b-instruct-q4_1", temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature
        self.initialized = False

    def initialize(self, use_openai: bool = False):
        # Minimal initialization checks; in your project this may be more elaborate.
        logger.info("Ollama is running with model: %s", self.model_name)
        logger.info("Model %s is ready to use!", self.model_name)
        self.initialized = True

    def _build_optimized_context(self, relevant_chunks: List[Tuple[str, Optional[float], Dict]]) -> str:
        """
        Convert a list of (text, score, metadata) into a single context string.
        """
        parts = []
        for i, (text, score, meta) in enumerate(relevant_chunks):
            header = f"[CHUNK {i+1} | score={score}]"
            meta_line = f"[meta={meta}]" if meta else ""
            parts.append(f"{header} {meta_line}\n{text}")
        return "\n\n".join(parts)

    def _build_dynamic_prompt(self, question: str, context: str) -> str:
        """
        Build a deterministic, instruction-heavy prompt that asks the model to only use the context.
        """
        prompt_template = (
            "You are an assistant that answers using ONLY the information present in the provided Context.\n\n"
            "Context:\n{context}\n\n"
            "Instructions:\n"
            "1. Using only the text in Context above, answer the Question exactly.\n"
            "2. If the requested item is not present in the Context, explicitly say 'Not mentioned.'\n"
            "3. Do not hallucinate or invent any information not contained in the Context.\n"
            "4. When listing items, cite the CHUNK number shown in Context for each item.\n"
            "5. Keep answers concise and factual.\n\n"
            "6. If asked to summarize, provide a brief summary in 40-50 words, do not hallucinate or mix up content and create new stuff.\n\n"
            "7. Always format your answer in markdown.\n\n"
            "8. If asked to explain or asked \"what is\", provide a clear and simple explanation based on the Context, Recheck and Recheck the document properly before generating a response.\n\n"
            "Question: {question}\n\n"
            "Answer:\n"
        )
        return prompt_template.format(context=context, question=question)

    def _call_ollama_with_retry(self, prompt: str, max_retries: int = 3, base_timeout: int = 60) -> Optional[str]:
        """
        Low-level HTTP call to Ollama with retries and exponential backoff.
        Assumes Ollama is running locally on port 11434.
        """
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            # include other params (max_tokens, top_p, etc.) if needed
        }
        headers = {"Content-Type": "application/json"}
        last_exc = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug("Calling Ollama (attempt %d) with timeout=%ds", attempt, base_timeout)
                resp = requests.post(url, json=payload, headers=headers, timeout=base_timeout)
                resp.raise_for_status()
                # Response shapes may vary; attempt tolerant extraction
                try:
                    data = resp.json()
                except Exception:
                    text = resp.text
                    logger.debug("Ollama returned non-json response: %s", text[:500])
                    return text

                # Common shapes: {"response": "<text>"} or {"completions": [{"data": {"content":"..."}}]}
                if isinstance(data, dict):
                    if "response" in data:
                        return data["response"]
                    if "text" in data:
                        return data["text"]
                    # try completions
                    if "completions" in data:
                        comps = data["completions"]
                        if isinstance(comps, list) and comps:
                            possible = comps[0].get("data") or comps[0]
                            if isinstance(possible, dict):
                                # find textual fields
                                for k in ("content", "text", "output", "response"):
                                    if k in possible:
                                        return possible[k]
                            else:
                                # fallback to string conversion
                                return str(possible)
                    # fallback: return stringified json
                    return str(data)
                else:
                    # non-dict body; return string
                    return str(data)
            except requests.exceptions.RequestException as e:
                last_exc = e
                wait = 2 ** (attempt - 1)
                logger.warning("Ollama call attempt %d failed: %s (waiting %ds before retry)", attempt, str(e), wait)
                time.sleep(wait)

        logger.error("Ollama call failed after %d attempts. Last error: %s", max_retries, repr(last_exc))
        return None

    def _call_ollama_api(self, question: str, relevant_chunks: Optional[List[Tuple[str, Optional[float], Dict]]] = None) -> Optional[str]:
        """
        Call Ollama API with optimized prompt and parameters.

        Supports two calling modes:
         - _call_ollama_api(question, relevant_chunks): builds prompt from question+chunks
         - _call_ollama_api(prompt, None): prompt is already constructed and is passed directly.

        Returns the raw string output from the model, or None on failure.
        """
        try:
            # If relevant_chunks is None, assume `question` is already a full prompt.
            if relevant_chunks is None:
                logger.info("Using prompt-only mode for Ollama call (caller provided full prompt).")
                return self._call_ollama_with_retry(question)

            # Build optimized context and dynamic prompt from question + relevant chunks
            context = self._build_optimized_context(relevant_chunks)
            prompt = self._build_dynamic_prompt(question, context)

            logger.info("Calling Ollama with generated prompt (length: %d)", len(prompt))
            return self._call_ollama_with_retry(prompt)
        except Exception as e:
            logger.error("Error calling Ollama API: %s", str(e))
            logger.debug(traceback.format_exc())
            return None

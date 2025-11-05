"""
Groq LLM service with improved prompt roles and retry/backoff for HTTP calls.
"""
import logging
import requests
from typing import Dict, Any, Optional, List
import time

from core.config import config

logger = logging.getLogger(__name__)

class GroqLLMService:
    def __init__(self):
        self.initialized = False
        self.current_model = config.model.MODEL
        self.base_url = "https://api.groq.com/openai/v1"

        # System prompt for RAG
        self.system_rag = (
            "You are an expert Q&A assistant. Use only the provided context to answer. "
            "If the answer is not present in the context, say exactly: 'I could not find any relevant information in the document.'"
        )

        self.query_router_prompt = (
            "You are an expert query classifier. Classify ONLY as 'general' or 'specific'.\n\nQUESTION:\n{question}\n\nCATEGORY:"
        )

        self.hyde_prompt = (
            "You are a search query augmentor. Given a user's question, generate a short hypothetical passage that contains the ideal answer. "
            "Respond only with the hypothetical passage.\n\nQUESTION:\n{question}\n\nPASSAGE:"
        )

        self.direct_prompt_template = "{question}\n\nProvide a helpful and concise response (2-3 sentences):"

    def initialize(self) -> bool:
        if not config.model.GROQ_API_KEY:
            logger.error("❌ GROQ_API_KEY not found in environment")
            return False

        logger.info("🔄 Testing Groq API connection...")
        try:
            headers = {
                "Authorization": f"Bearer {config.model.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.current_model,
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 5
            }
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                self.initialized = True
                logger.info(f"✅ Groq service initialized with {self.current_model}")
                return True
            else:
                logger.error(f"❌ Groq API failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Groq initialization failed: {e}")
            return False

    def _post_with_retries(self, payload: Dict[str, Any], timeout: int = 30, max_retries: int = 3) -> Optional[requests.Response]:
        """
        Post helper with exponential backoff retries on 5xx and timeouts.
        """
        headers = {
            "Authorization": f"Bearer {config.model.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        attempt = 0
        backoff = 1.0
        while attempt < max_retries:
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
                if 200 <= resp.status_code < 300:
                    return resp
                if 500 <= resp.status_code < 600:
                    logger.warning(f"Server error {resp.status_code}, retrying after {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    attempt += 1
                    continue
                # For other codes, return immediately
                return resp
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt+1}/{max_retries}, retrying after {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                attempt += 1
            except requests.exceptions.RequestException as e:
                logger.error(f"RequestException while calling Groq: {e}")
                return None
        return None

    def route_query(self, question: str) -> str:
        if not self.initialized:
            logger.warning("Router not initialized, defaulting to 'specific'")
            return "specific"

        prompt = self.query_router_prompt.format(question=question)
        payload = {
            "model": self.current_model,
            "messages": [
                {"role": "system", "content": "Classify queries into 'general' or 'specific'."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 5,
            "temperature": 0.0
        }

        resp = self._post_with_retries(payload, timeout=5, max_retries=2)
        if not resp:
            return "specific"
        try:
            result_text = resp.json()['choices'][0]['message']['content'].strip().lower()
            if "general" in result_text:
                return "general"
            else:
                return "specific"
        except Exception:
            return "specific"

    def generate_hypothetical_query(self, question: str) -> Dict[str, Any]:
        if not self.initialized:
            return {"success": False, "query": question, "error": "Not initialized"}

        prompt = self.hyde_prompt.format(question=question)
        payload = {
            "model": self.current_model,
            "messages": [
                {"role": "system", "content": "Generate a short hypothetical passage for search augmentation."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.0
        }
        resp = self._post_with_retries(payload, timeout=10, max_retries=2)
        if not resp:
            return {"success": False, "query": question, "error": "HyDE request failed"}
        try:
            result = resp.json()
            hyde_query = result['choices'][0]['message']['content'].strip()
            return {"success": True, "query": hyde_query}
        except Exception as e:
            logger.error(f"HyDE parse failed: {e}")
            return {"success": False, "query": question, "error": str(e)}

    def generate_answer(self, question: str, context: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        if not self.initialized:
            return {"success": False, "answer": "Service not initialized", "error": "Not initialized"}

        try:
            messages = []
            # system role
            messages.append({"role": "system", "content": self.system_rag})

            # include limited chat history in user role context if provided
            history_block = ""
            if chat_history:
                for msg in chat_history[-5:]:
                    history_block += f"Human: {msg.get('question','')}\nAssistant: {msg.get('answer','')}\n"

            user_content = ""
            if history_block:
                user_content += f"CHAT HISTORY:\n{history_block}\n"
            if context:
                user_content += f"CONTEXT:\n{context}\n"
            user_content += f"QUESTION:\n{question}\n"

            messages.append({"role": "user", "content": user_content})

            payload = {
                "model": self.current_model,
                "messages": messages,
                "max_tokens": config.model.MAX_TOKENS,
                "temperature": config.model.TEMPERATURE,
                "top_p": config.model.TOP_P
            }

            resp = self._post_with_retries(payload, timeout=30, max_retries=3)
            if not resp:
                return {"success": False, "answer": "Failed to get response from AI service", "error": "Request failed"}

            result = resp.json()
            answer = result['choices'][0]['message']['content'].strip()
            tokens_used = result.get('usage', {}).get('total_tokens', 0)

            return {"success": True, "answer": answer, "model": self.current_model, "context_used": context is not None, "tokens_used": tokens_used}

        except Exception as e:
            logger.error(f"❌ generate_answer failed: {e}")
            return {"success": False, "answer": "Network error occurred", "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        return {"initialized": self.initialized, "current_model": self.current_model, "provider": "Groq API"}

# global instance
llm_service = GroqLLMService()
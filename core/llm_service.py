"""
Groq LLM service with improved prompt roles and retry/backoff for HTTP calls.
FIXED: Decoupled prompts into core/prompts.py for experimentation.
       Uses LangChain ChatPromptTemplate to format messages.
FIXED: Correctly map LangChain 'human'/'ai' types to Groq 'user'/'assistant' roles.
ADDED: Logic to pass 'is_summary' flag to prompt formatting.
"""
import logging
import requests
from typing import Dict, Any, Optional, List
import time

from core.config import config
from core.prompts import (
    PROMPT_REGISTRY, 
    QUERY_ROUTER_PROMPT, 
    HYDE_PROMPT
)

logger = logging.getLogger(__name__)

class GroqLLMService:
    
    def __init__(self):
        # ... (no changes)
        self.initialized = False
        self.current_model = config.model.MODEL
        self.base_url = "https://api.groq.com/openai/v1"

        try:
            template_name = config.rag.RAG_PROMPT_TEMPLATE
            self.rag_prompt = PROMPT_REGISTRY[template_name]
            logger.info(f"✅ Loaded RAG prompt template: {template_name}")
        except KeyError:
            logger.error(f"❌ Failed to load prompt '{template_name}'. Defaulting to STRICT_CONTEXT_V1.")
            self.rag_prompt = PROMPT_REGISTRY["STRICT_CONTEXT_V1"]
        except Exception as e:
            logger.error(f"Critical error loading prompts: {e}. Defaulting to STRICT_CONTEXT_V1.")
            self.rag_prompt = PROMPT_REGISTRY["STRICT_CONTEXT_V1"]
        
        self.query_router_prompt = QUERY_ROUTER_PROMPT
        self.hyde_prompt = HYDE_PROMPT

    def initialize(self) -> bool:
        # ... (no changes)
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
        # ... (no changes)
        headers = {
            "Authorization": f"Bearer {config.model.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        attempt = 0
        backoff = 1.0
        while attempt < max_retries:
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
                
                if 400 <= resp.status_code < 500:
                    logger.error(f"Client error {resp.status_code} calling Groq: {resp.text}")
                    return resp 
                
                if 200 <= resp.status_code < 300:
                    return resp
                
                if 500 <= resp.status_code < 600:
                    logger.warning(f"Server error {resp.status_code}, retrying after {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    attempt += 1
                    continue
                
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

    def _format_messages(self, prompt_template: Any, **kwargs) -> List[Dict[str, str]]:
        # ... (no changes)
        try:
            prompt_value = prompt_template.format_prompt(**kwargs)
            messages_objects = prompt_value.to_messages()
            messages_dicts = []
            for msg in messages_objects:
                role = ""
                if msg.type == 'system':
                    role = 'system'
                elif msg.type == 'human':
                    role = 'user'
                elif msg.type == 'ai':
                    role = 'assistant'
                
                if role:
                    messages_dicts.append({"role": role, "content": msg.content})
                else:
                    logger.warning(f"Unknown message type '{msg.type}' in _format_messages")

            return messages_dicts
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}", exc_info=True) 
            raise
    
    def route_query(self, question: str) -> str:
        # ... (no changes)
        if not self.initialized:
            logger.warning("Router not initialized, defaulting to 'specific'")
            return "specific"

        try:
            messages = self._format_messages(self.query_router_prompt, question=question)
            payload = {
                "model": self.current_model,
                "messages": messages,
                "max_tokens": 5,
                "temperature": 0.0
            }
        except Exception as e:
            logger.error(f"Failed to format route_query: {e}")
            return "specific"

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
        # ... (no changes)
        if not self.initialized:
            return {"success": False, "query": question, "error": "Not initialized"}

        try:
            messages = self._format_messages(self.hyde_prompt, question=question)
            payload = {
                "model": self.current_model,
                "messages": messages,
                "max_tokens": 150,
                "temperature": 0.0
            }
        except Exception as e:
            logger.error(f"Failed to format hyde_query: {e}")
            return {"success": False, "query": question, "error": str(e)}
        
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

    # -----------------------------------------------------------------
    # 🚨 START: MODIFIED generate_answer SIGNATURE
    # -----------------------------------------------------------------
    def generate_answer(self, 
                        question: str, 
                        context: Optional[str] = None, 
                        chat_history: Optional[List[Dict[str, str]]] = None,
                        is_summary: bool = False # 🚨 NEW FLAG
                       ) -> Dict[str, Any]:
    # -----------------------------------------------------------------
    # 🚨 END: MODIFIED generate_answer SIGNATURE
    # -----------------------------------------------------------------
        if not self.initialized:
            return {"success": False, "answer": "Service not initialized", "error": "Not initialized"}

        try:
            history_str = ""
            if chat_history:
                for msg in chat_history[-5:]:
                    history_str += f"Human: {msg.get('question','')}\nAssistant: {msg.get('answer','')}\n"
            
            # -----------------------------------------------------------------
            # 🚨 NEW: Modify question based on task
            # -----------------------------------------------------------------
            final_question = question
            if is_summary and "summary" not in question.lower() and "summarize" not in question.lower():
                # If the user just said "what is this", we prepend the
                # instruction to the context-less query
                final_question = f"Provide a high-level summary of the provided context. What is it about? Original query: {question}"
                logger.info("Auto-switching to summary instruction.")
            # -----------------------------------------------------------------

            messages = self._format_messages(
                self.rag_prompt,
                chat_history=history_str or "No history",
                context=context or "No context provided",
                question=final_question # Use the (potentially modified) question
            )

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

            if not (200 <= resp.status_code < 300):
                error_text = resp.text
                logger.error(f"Groq API returned error {resp.status_code}: {error_text}")
                return {"success": False, "answer": "AI service returned an error", "error": f"API Error {resp.status_code}: {error_text}"}

            result = resp.json()
            answer = result['choices'][0]['message']['content'].strip()
            tokens_used = result.get('usage', {}).get('total_tokens', 0)

            return {"success": True, "answer": answer, "model": self.current_model, "context_used": context is not None, "tokens_used": tokens_used}

        except Exception as e:
            logger.error(f"❌ generate_answer failed: {e}", exc_info=True)
            return {"success": False, "answer": "An internal error occurred", "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        # ... (no changes)
        return {"initialized": self.initialized, "current_model": self.current_model, "provider": "Groq API"}

# global instance
llm_service = GroqLLMService()
"""
Final working Groq service with updated model and HTTP requests.
"""
import logging
import requests
from typing import Dict, Any, Optional
from core.config import config

logger = logging.getLogger(__name__)

class GroqLLMService:
    def __init__(self):
        self.initialized = False
        self.current_model = "llama-3.1-8b-instant"  # Updated model
        self.base_url = "https://api.groq.com/openai/v1"
        
        # Optimized prompts
        self.rag_prompt = """You are a helpful AI assistant. Answer the question based ONLY on the provided context.

IMPORTANT RULES:
1. Use ONLY information from the context below
2. If the context doesn't contain the answer, say "I cannot find this information in the document"
3. Be concise and factual (2-3 sentences maximum)
4. Do not add any information not present in the context

CONTEXT:
{context}

QUESTION: 
{question}

ANSWER:"""
        
        self.direct_prompt = """{question}

Provide a helpful and concise response (2-3 sentences):"""
    
    def initialize(self) -> bool:
        if not config.model.GROQ_API_KEY:
            logger.error("❌ GROQ_API_KEY not found in .env file")
            return False
        
        logger.info("🔄 Testing Groq API connection...")
        
        try:
            headers = {
                "Authorization": f"Bearer {config.model.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.current_model,  # Use updated model
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
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
    
    def generate_answer(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        if not self.initialized:
            return {
                "success": False, 
                "answer": "Service not initialized",
                "error": "Not initialized"
            }
        
        try:
            # Build the prompt
            if context:
                prompt = self.rag_prompt.format(context=context, question=question)
            else:
                prompt = self.direct_prompt.format(question=question)
            
            headers = {
                "Authorization": f"Bearer {config.model.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.current_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config.model.MAX_TOKENS,
                "temperature": config.model.TEMPERATURE,
                "top_p": config.model.TOP_P
            }
            
            logger.info(f"🚀 Sending request to Groq API...")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content'].strip()
                
                return {
                    "success": True,
                    "answer": answer,
                    "model": self.current_model,
                    "context_used": context is not None,
                    "tokens_used": result.get('usage', {}).get('total_tokens', 0)
                }
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                logger.error(f"❌ {error_msg}")
                return {
                    "success": False,
                    "answer": "Failed to get response from AI service",
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "answer": "Network error occurred",
                "error": error_msg
            }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "current_model": self.current_model,
            "provider": "Groq API"
        }

# Global instance
llm_service = GroqLLMService()
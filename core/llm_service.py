"""
Final working Groq service with an intelligent Query Router and Conversation Memory.
"""
import logging
import requests
from typing import Dict, Any, Optional, List
from core.config import config

logger = logging.getLogger(__name__)

class GroqLLMService:
    def __init__(self):
        self.initialized = False
        self.current_model = "llama-3.1-8b-instant"  # Updated model
        self.base_url = "https://api.groq.com/openai/v1"
        
        self.query_router_prompt = """You are an expert query classifier. Your task is to classify a user's question into one of two categories based on its *intent*: 'general' or 'specific'.

- 'general': For broad, open-ended, or conceptual questions. These queries ask for summaries, explanations, or main points.
- 'specific': For queries seeking precise facts, definitions, keywords, or proper nouns.

IMPORTANT: Constraints like "in 20 words", "list 5 items", or "in 120-150 words" DO NOT change the query type. Classify based on the core question.

Here are examples:
Q: "summarize this document" -> A: "general"
Q: "what is this about?" -> A: "general"
Q: "explain the main points" -> A: "general"
Q: "summarize in 100 words" -> A: "general"
Q: "what is in the experience section?" -> A: "specific"
Q: "what is SaaS?" -> A: "specific"
Q: "who is Akshay Manjunath?" -> A: "specific"
Q: "what are the types of cloud?" -> A: "specific"

Respond with ONLY the category name ('general' or 'specific') and nothing else.

QUESTION:
{question}

CATEGORY:"""
        
        # -----------------------------------------------------------------
        # 🚨 MODIFIED: Updated prompt to include a {chat_history} block
        # -----------------------------------------------------------------
        self.rag_prompt = """You are an expert Q&A assistant. Your task is to answer the user's question based on the provided context.
Use the conversation history to understand follow-up questions.

IMPORTANT RULES:
1. You must ground your answer in the information found in the context.
2. Provide a comprehensive and detailed answer by synthesizing information from all relevant context chunks. Do not just quote one chunk.
3. If the user's question includes specific constraints (e.g., "answer in 20 words", "list 5 items"), you MUST follow those constraints.
4. If the context does not contain relevant information to answer the question, and only in that case, say "I could not find any relevant information in the document."
5. Do not add any information that is not present in the context.

CHAT HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION: 
{question}

ANSWER:"""
        # -----------------------------------------------------------------
        
        self.direct_prompt = """{question}

Provide a helpful and concise response (2-3 sentences):"""
        
        self.hyde_prompt = """You are a search query augmentor. Given a user's question, generate a short, hypothetical passage that contains the ideal answer. This passage will be used for a vector search.

Respond ONLY with the hypothetical passage, and nothing else.

QUESTION: 
{question}

PASSAGE:"""
    
    def initialize(self) -> bool:
        # ... (keep this method as is) ...
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
            
    def route_query(self, question: str) -> str:
        # ... (keep this method as is) ...
        if not self.initialized:
            logger.warning("Router not initialized, defaulting to 'specific'")
            return "specific"
        
        try:
            prompt = self.query_router_prompt.format(question=question)
            
            headers = {
                "Authorization": f"Bearer {config.model.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.current_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 5,    # Only needs one word
                "temperature": 0.0,
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content'].strip().lower()
                
                if "general" in result:
                    logger.info("Query route: general")
                    return "general"
                else:
                    logger.info("Query route: specific")
                    return "specific"
            else:
                logger.warning(f"Query router failed: {response.text}, defaulting to 'specific'")
                return "specific"
                
        except Exception as e:
            logger.warning(f"Query router exception: {e}, defaulting to 'specific'")
            return "specific"

    def generate_hypothetical_query(self, question: str) -> Dict[str, Any]:
        # ... (keep this method as is) ...
        if not self.initialized:
            return {
                "success": False, 
                "query": question,
                "error": "Not initialized"
            }
        
        try:
            prompt = self.hyde_prompt.format(question=question)
            
            headers = {
                "Authorization": f"Bearer {config.model.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.current_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,  # Short passage
                "temperature": 0.0, # Be factual
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                hyde_query = result['choices'][0]['message']['content'].strip()
                
                return {
                    "success": True,
                    "query": hyde_query,
                }
            else:
                error_msg = f"HyDE API error: {response.status_code} - {response.text}"
                logger.error(f"❌ {error_msg}")
                return {
                    "success": False,
                    "query": question, # Fallback to original question
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"HyDE Request failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "query": question, # Fallback to original question
                "error": error_msg
            }
    
    # -----------------------------------------------------------------
    # 🚨 MODIFIED: `generate_answer` now accepts and formats chat_history
    # -----------------------------------------------------------------
    def generate_answer(self, question: str, context: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        if not self.initialized:
            return {
                "success": False, 
                "answer": "Service not initialized",
                "error": "Not initialized"
            }
        
        try:
            # Build the prompt
            if context:
                # Format chat history
                history_str = "No history provided."
                if chat_history:
                    history_str = ""
                    for msg in chat_history:
                        # Use keys that match frontend state manager
                        history_str += f"Human: {msg['question']}\nAssistant: {msg['answer']}\n"
                
                prompt = self.rag_prompt.format(chat_history=history_str, context=context, question=question)
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
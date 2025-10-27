"""
LLM integration with Ollama local models - Virtual Environment Compatible
"""
import os
import requests
import logging
import time
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class LLMIntegration:
    """Handles answer generation using Ollama local LLM."""
    
    def __init__(self, use_ollama: bool = True):
        self.use_ollama = use_ollama
        self.model_name = "llama3.2:1b-instruct-q4_1"
        self.ollama_base_url = "http://localhost:11434"
        self.is_initialized = False
    
        if not use_ollama:
            self.model_name = "smart_rule_based"
    
    def initialize(self, use_openai: bool = False, api_key: str = None):
        """Initialize the LLM integration."""
        if self.use_ollama:
            if self._check_ollama_available():
                logger.info(f"✅ Ollama is running with model: {self.model_name}")
                
                # Verify our specific model is available
                available_models = self._get_available_models()
                if self.model_name in available_models:
                    logger.info(f"🚀 Model {self.model_name} is ready to use!")
                else:
                    logger.warning(f"❌ Model {self.model_name} not found in available models")
                    logger.info(f"📦 Available models: {available_models}")
                    self.use_ollama = False
            else:
                logger.warning("❌ Ollama service not accessible")
                logger.info("💡 Make sure Ollama is running as a Windows service")
                logger.info("📝 Using rule-based system instead")
                self.use_ollama = False
        
        self.is_initialized = True
        logger.info("LLM Integration initialized successfully")
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama check failed: {str(e)}")
            return False
    
    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model.get('name') for model in models]
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
        return []
    
    def generate_answer(self, question: str, context_chunks: List[Tuple[str, float, dict]]) -> str:
        """
        Generate an answer using Ollama LLM with context.
        """
        if not context_chunks:
            return "I couldn't find any relevant information in the documents to answer your question."
        
        # Try Ollama first if enabled
        if self.use_ollama:
            try:
                ollama_answer = self._call_ollama_api(question, context_chunks)
                if ollama_answer and self._validate_answer(ollama_answer):
                    logger.info("✅ Successfully generated answer using Ollama LLM")
                    return ollama_answer
                else:
                    logger.warning("Ollama returned invalid answer, using fallback")
            except Exception as e:
                logger.warning(f"Ollama API failed: {str(e)}, using rule-based fallback")
        
        # Fallback to rule-based system
        return self._rule_based_fallback(question, context_chunks)
    
    def _call_ollama_api(self, question: str, context_chunks: List[Tuple[str, float, dict]]) -> Optional[str]:
        """Call Ollama API with ultra-fast settings."""
    
        # Use ONLY the single best chunk for speed
        if context_chunks:
            best_chunk, best_score, best_meta = context_chunks[0]
            context = f"Content: {best_chunk[:400]}"  # Truncate to 400 chars
        else:
            return None
    
        # Ultra-fast prompt
        prompt = f"""Context: {context}

    Question: {question}

    Short answer (2-3 sentences max):"""
    
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.7,
                "num_predict": 150,  # Very short responses
                "top_k": 5,          # Very restricted sampling
                "repeat_penalty": 1.2
            }
        }

        try:
            logger.info(f"🤖 Calling Ollama with ULTRA-FAST settings...")
            start_time = time.time()
        
            response = requests.post(
                f"{self.ollama_base_url}/api/generate", 
                json=payload, 
                timeout=20  # Fail fast if slow
            )
        
            response_time = time.time() - start_time
            logger.info(f"⏱️ Ollama response time: {response_time:.2f}s")
        
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                return generated_text
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None
            
        except requests.exceptions.Timeout:
            logger.error("Ollama timeout - too slow")
            return None
        except Exception as e:
            logger.error(f"Ollama error: {str(e)}")
            return None
    
    def _clean_ollama_response(self, response: str) -> str:
        """Clean and format the Ollama response."""
        # Remove any thinking patterns or meta-commentary
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip common thinking prefixes
            if line and not line.lower().startswith(('i think', 'let me', 'based on', 'hmm', 'well,', 'so,', 'according to')):
                cleaned_lines.append(line)
        
        response = ' '.join(cleaned_lines)
        
        # Ensure proper sentence structure
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response.strip()
    
    def _validate_answer(self, answer: str) -> bool:
        """Validate that the generated answer is reasonable."""
        if not answer or len(answer.strip()) < 10:
            return False
        
        # Check for common error patterns
        error_indicators = [
            "error", "sorry,", "i cannot", "i'm unable", "model is loading",
            "please try again", "as an ai", "i don't have", "empty response"
        ]
        
        answer_lower = answer.lower()
        if any(indicator in answer_lower for indicator in error_indicators):
            return False
        
        return True
    
    def _rule_based_fallback(self, question: str, context_chunks: List[Tuple[str, float, dict]]) -> str:
        """Fallback to rule-based answer generation."""
        if not context_chunks:
            return "No relevant information found."
        
        # Filter for highly relevant chunks
        relevant_chunks = [(chunk, score, meta) for chunk, score, meta in context_chunks if score > 0.3]
        
        if not relevant_chunks:
            return "The documents don't contain specific information about this topic."
        
        # Use the best chunk
        best_chunk, best_score, best_meta = relevant_chunks[0]
        
        # Clean and format the chunk
        clean_chunk = self._clean_text(best_chunk)
        
        # Create a structured answer
        answer_parts = []
        
        if "what is" in question.lower() or "define" in question.lower():
            answer_parts.append("Based on the documents:")
        elif "how" in question.lower():
            answer_parts.append("According to the documents:")
        else:
            answer_parts.append("Relevant information from the documents:")
        
        answer_parts.append("")
        answer_parts.append(clean_chunk)
        
        if len(relevant_chunks) > 1:
            answer_parts.append("")
            answer_parts.append(f"[Found {len(relevant_chunks)} relevant sections in the documents]")
        
        return "\n".join(answer_parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean and format text for better readability."""
        import re
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Capitalize first letter
        text = text.strip()
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Ensure it ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text

def main():
    """Test the Ollama LLM integration."""
    print("🧪 Testing Ollama LLM Integration...")
    llm = LLMIntegration(use_ollama=True)
    llm.initialize()
    
    # Test with realistic chunks (like from your documents)
    test_chunks = [
        ("Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The core principle of AI is to create systems that can perform tasks that typically require human intelligence.", 0.95, {}),
        ("Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It works by using algorithms to analyze data, identify patterns, and make decisions with minimal human intervention.", 0.85, {}),
        ("Deep learning is a specialized subset of machine learning that uses neural networks with multiple layers. These networks are inspired by the human brain and can learn from vast amounts of data.", 0.75, {})
    ]
    
    test_questions = [
        "What is Artificial Intelligence?",
        "How does machine learning work?",
        "What is the difference between AI and machine learning?"
    ]
    
    for question in test_questions:
        print(f"\n" + "="*60)
        print(f"❓ QUESTION: {question}")
        print("="*60)
        
        answer = llm.generate_answer(question, test_chunks)
        print(f"🤖 ANSWER: {answer}")

if __name__ == "__main__":
    main()
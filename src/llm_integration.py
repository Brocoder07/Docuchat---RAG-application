"""
LLM integration with Ollama local models - Advanced RAG Pipeline
"""
import os
import requests
import logging
import time
import re
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class LLMIntegration:
    """Advanced LLM integration with multi-stage answer generation and validation."""
    
    def __init__(self, use_ollama: bool = True):
        self.use_ollama = use_ollama
        self.model_name = "llama3.2:1b-instruct-q4_1"
        self.ollama_base_url = "http://localhost:11434"
        self.is_initialized = False
        self.answer_cache = {}  # Simple cache for similar questions
    
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
                logger.info("📝 Using advanced rule-based system instead")
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
        Generate a comprehensive answer using Ollama LLM with context.
        """
        if not context_chunks:
            return "I cannot answer this question based on the provided documents."
    
        # Use lower threshold to capture more relevant content
        relevant_chunks = [(chunk, score, meta) for chunk, score, meta in context_chunks if score > 0.2]
    
        if not relevant_chunks:
            return "I cannot answer this question based on the provided documents."
    
        #ALWAYS try LLM first for better quality answers
        if self.use_ollama:
            try:
                ollama_answer = self._call_ollama_api(question, relevant_chunks)
                if ollama_answer and ollama_answer.strip():
                    logger.info("✅ Using LLM-generated answer")
                    return ollama_answer
            except Exception as e:
                logger.warning(f"Ollama API failed: {str(e)}, using fallback")
    
        #Fallback - use the improved fallback
        return self._detailed_rule_based_fallback(question, relevant_chunks)
    
    def _multi_stage_answer_generation(self, question: str, context_chunks: List[Tuple[str, float, dict]]) -> str:
        """Advanced multi-stage pipeline for robust answer generation."""
        
        # Stage 1: Try direct LLM answer with optimized prompt
        if self.use_ollama:
            llm_answer = self._stage1_direct_llm_answer(question, context_chunks)
            if self._validate_answer_quality(llm_answer, question):
                logger.info("✅ Stage 1: Direct LLM answer accepted")
                return llm_answer
        
        # Stage 2: Extract and synthesize key information
        synthesized_answer = self._stage2_information_synthesis(question, context_chunks)
        if self._validate_answer_quality(synthesized_answer, question):
            logger.info("✅ Stage 2: Synthesized answer accepted")
            return synthesized_answer
        
        # Stage 3: Smart fallback with context-aware formatting
        fallback_answer = self._stage3_smart_fallback(question, context_chunks)
        logger.info("🔄 Stage 3: Using smart fallback")
        return fallback_answer
    
    def _stage1_direct_llm_answer(self, question: str, context_chunks: List[Tuple[str, float, dict]]) -> Optional[str]:
        """Stage 1: Direct LLM answer generation with optimized prompts."""
        relevant_chunks = [(chunk, score, meta) for chunk, score, meta in context_chunks if score > 0.25]
        
        if not relevant_chunks:
            return None
        
        # Smart context selection
        context = self._build_optimized_context(relevant_chunks)
        
        # Dynamic prompt based on question type
        prompt = self._build_dynamic_prompt(question, context)
        
        try:
            response = self._call_ollama_with_retry(prompt)
            if response and self._is_structured_answer(response):
                return self._clean_and_format_answer(response)
        except Exception as e:
            logger.warning(f"Stage 1 LLM failed: {str(e)}")
        
        return None
    
    def _build_optimized_context(self, relevant_chunks: List[Tuple[str, float, dict]]) -> str:
        """Build optimized context by cleaning and selecting most relevant parts."""
        context_parts = []
        total_length = 0
        max_length = 1200  # Optimized for small model
        
        for chunk, score, meta in relevant_chunks[:2]:
            cleaned_chunk = self._clean_context_chunk(chunk)
            # Extract most relevant sentences based on content
            key_sentences = self._extract_key_sentences(cleaned_chunk, max_length - total_length)
            if key_sentences and total_length + len(key_sentences) <= max_length:
                context_parts.append(key_sentences)
                total_length += len(key_sentences)
        
        return "\n\n".join(context_parts)
    
    def _extract_key_sentences(self, text: str, max_length: int) -> str:
        """Extract key sentences from text, prioritizing definitions and lists."""
        sentences = re.split(r'[.!?]+', text)
        key_sentences = []
        current_length = 0
        
        # Prioritize sentences with key patterns
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Score sentences based on content importance
            score = self._score_sentence_importance(sentence)
            if score > 0.3 and current_length + len(sentence) <= max_length:
                key_sentences.append(sentence)
                current_length += len(sentence)
        
        return ". ".join(key_sentences) + "." if key_sentences else text[:max_length]
    
    def _score_sentence_importance(self, sentence: str) -> float:
        """Score sentence importance based on content patterns."""
        sentence_lower = sentence.lower()
        score = 0.0
        
        # Definitions and explanations
        if any(pattern in sentence_lower for pattern in ['is defined as', 'refers to', 'means that', 'can be defined']):
            score += 0.8
        # Lists and enumerations
        if any(pattern in sentence_lower for pattern in ['types of', 'categories', 'the following', 'includes']):
            score += 0.7
        # Key characteristics
        if any(pattern in sentence_lower for pattern in ['characteristics', 'features', 'advantages', 'benefits']):
            score += 0.6
        # Examples
        if 'for example' in sentence_lower or 'e.g.' in sentence_lower:
            score += 0.4
        
        return min(score, 1.0)
    
    def _build_dynamic_prompt(self, question: str, context: str) -> str:
        """Build dynamic prompt that forces complete, structured answers."""
        question_lower = question.lower()
    
        # ULTRA-STRICT prompt for complete answers
        strict_prompt = f"""You MUST answer the question using ONLY the information from the provided context. 

    CRITICAL RULES:
    1. Use ONLY the information provided in the context below
    2. Do not use any external knowledge or make up information
    3. If the context contains a list or categories, provide the COMPLETE list
    4. Structure your answer clearly with bullet points or numbered lists when appropriate
    5. Include ALL relevant details mentioned in the context
    6. If you cannot answer based on the context, say "I cannot answer this based on the provided documents."

    CONTEXT:
    {context}

    QUESTION: {question}

    ANSWER BASED STRICTLY ON THE CONTEXT:"""

        return strict_prompt
    
    def _call_ollama_with_retry(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """Call Ollama with retry logic for better reliability."""
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.7,
                        "num_predict": 250,
                        "top_k": 20,
                        "repeat_penalty": 1.2
                    }
                }
                
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    timeout=20
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('response', '').strip()
                    
                    # Quick validation
                    if answer and len(answer) > 10 and not answer.lower().startswith(('i cannot', 'sorry')):
                        return answer
                
            except Exception as e:
                logger.warning(f"Ollama call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Brief pause before retry
        
        return None
    
    def _is_structured_answer(self, answer: str) -> bool:
        """Check if answer is properly structured and not raw context."""
        if not answer or len(answer) < 15:
            return False
        
        # Check for raw context repetition patterns
        raw_patterns = [
            'consumer does not manage',
            'cloud infrastructure is operated',
            'page',
            '9 cloud services',
            '10 types of cloud'
        ]
        
        if any(pattern in answer.lower() for pattern in raw_patterns):
            return False
        
        return True
    
    def _stage2_information_synthesis(self, question: str, context_chunks: List[Tuple[str, float, dict]]) -> str:
        """Stage 2: Extract and synthesize information from multiple chunks."""
        relevant_chunks = [(self._clean_context_chunk(chunk), score, meta) 
                          for chunk, score, meta in context_chunks if score > 0.2]
        
        if not relevant_chunks:
            return "I cannot answer this question based on the provided documents."
        
        # Extract key information patterns
        extracted_info = self._extract_information_patterns(question, relevant_chunks)
        
        if extracted_info:
            return self._format_synthesized_answer(question, extracted_info)
        
        return self._stage3_smart_fallback(question, context_chunks)
    
    def _extract_information_patterns(self, question: str, chunks: List[Tuple[str, float, dict]]) -> Dict[str, Any]:
        """Extract structured information patterns from text chunks."""
        question_lower = question.lower()
        extracted = {
            "definitions": [],
            "lists": [],
            "examples": [],
            "characteristics": []
        }
        
        for chunk, score, meta in chunks:
            # Extract definitions
            definition_matches = re.findall(r'([A-Z][^.!?]*\b(?:is|are|refers to|means|defined as)[^.!?]*[.!?])', chunk, re.IGNORECASE)
            extracted["definitions"].extend(definition_matches[:2])
            
            # Extract list items
            list_items = re.findall(r'[•\-*]\s*([^.!?\n]+)', chunk)
            numbered_items = re.findall(r'\d+\.\s*([^.!?\n]+)', chunk)
            if list_items:
                extracted["lists"].extend(list_items[:5])
            if numbered_items:
                extracted["lists"].extend(numbered_items[:5])
            
            # Extract examples
            example_matches = re.findall(r'(?:for example|e\.g\.|such as)[^.!?]*[.!?]', chunk, re.IGNORECASE)
            extracted["examples"].extend(example_matches[:2])
            
            # Extract characteristics
            char_matches = re.findall(r'([A-Z][^.!?]*\b(?:characteristic|feature|advantage|benefit)[^.!?]*[.!?])', chunk, re.IGNORECASE)
            extracted["characteristics"].extend(char_matches[:3])
        
        # Remove duplicates and empty entries
        for key in extracted:
            extracted[key] = list(set([item.strip() for item in extracted[key] if item.strip()]))
        
        return extracted
    
    def _format_synthesized_answer(self, question: str, extracted_info: Dict[str, Any]) -> str:
        """Format extracted information into a coherent answer."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['list', 'types', 'categories', 'what are the']):
            if extracted_info["lists"]:
                return "Based on the documents:\n\n" + "\n".join([f"• {item}" for item in extracted_info["lists"][:8]])
        
        if any(word in question_lower for word in ['what is', 'define', 'explain']):
            if extracted_info["definitions"]:
                return "Based on the documents:\n\n" + extracted_info["definitions"][0]
        
        # General answer combining available information
        answer_parts = []
        if extracted_info["definitions"]:
            answer_parts.append(extracted_info["definitions"][0])
        if extracted_info["characteristics"]:
            answer_parts.append("\nKey characteristics:\n• " + "\n• ".join(extracted_info["characteristics"][:3]))
        if extracted_info["examples"]:
            answer_parts.append("\nExamples: " + extracted_info["examples"][0])
        
        if answer_parts:
            return "Based on the documents:\n\n" + "\n".join(answer_parts)
        
        return ""  # Fall back to stage 3
    
    def _stage3_smart_fallback(self, question: str, context_chunks: List[Tuple[str, float, dict]]) -> str:
        """Stage 3: Smart fallback with best-effort answer generation."""
        best_chunk, best_score, best_meta = context_chunks[0]
        cleaned_chunk = self._clean_context_chunk(best_chunk)
        
        # Extract the most relevant part of the chunk
        relevant_part = self._extract_most_relevant_part(cleaned_chunk, question)
        
        if relevant_part and len(relevant_part) > 50:
            return f"Based on the documents: {relevant_part}"
        else:
            return "I cannot answer this question based on the provided documents."
    
    def _extract_most_relevant_part(self, text: str, question: str) -> str:
        """Extract the part of text most relevant to the question."""
        sentences = re.split(r'[.!?]+', text)
        question_words = set(question.lower().split())
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
        
        return best_sentence if best_score > 0 else text[:300]  # Return first 300 chars if no good match
    
    def _clean_context_chunk(self, chunk: str) -> str:
        """Clean context chunk by removing artifacts."""
        # Remove page numbers, headers, etc.
        chunk = re.sub(r'^\s*\d+\s*$', '', chunk, flags=re.MULTILINE)
        chunk = re.sub(r'NPTEL ONLINE\s*CERTIFICATION COURSES', '', chunk)
        chunk = re.sub(r'\b(?:IT|IIT) KHARAGPUR\b', '', chunk)
        chunk = re.sub(r'\s+', ' ', chunk)
        return chunk.strip()
    
    def _clean_and_format_answer(self, answer: str) -> str:
        """Clean and format final answer."""
        if not answer:
            return answer
        
        # Remove thinking patterns
        lines = answer.split('\n')
        cleaned_lines = []
        
        thinking_patterns = [
            'i think', 'based on', 'according to', 'the context',
            'provide a clear', 'answer with a bulleted list'
        ]
        
        for line in lines:
            line = line.strip()
            if line and not any(line.lower().startswith(pattern) for pattern in thinking_patterns):
                cleaned_lines.append(line)
        
        answer = '\n'.join(cleaned_lines)
        
        # Ensure proper punctuation
        if answer and answer[-1] not in '.!?':
            answer += '.'
        
        return answer
    
    def _validate_answer_quality(self, answer: str, question: str) -> bool:
        """Validate that answer is high quality."""
        if not answer or len(answer.strip()) < 20:
            return False
        
        # Check for raw context repetition
        raw_patterns = [
            'page', 'consumer does not', 'cloud infrastructure is',
            '9 cloud services', '10 types of cloud'
        ]
        
        if any(pattern in answer.lower() for pattern in raw_patterns):
            return False
        
        # Check if answer actually addresses the question
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words.intersection(answer_words))
        
        return overlap >= 1  # At least one question word should be in answer
    
    def _get_cached_answer(self, question: str, context_chunks: List[Tuple[str, float, dict]]) -> Optional[str]:
        """Get cached answer for similar questions."""
        question_key = question.lower().strip()
        return self.answer_cache.get(question_key)
    
    def _cache_answer(self, question: str, context_chunks: List[Tuple[str, float, dict]], answer: str):
        """Cache answer for future similar questions."""
        question_key = question.lower().strip()
        self.answer_cache[question_key] = answer
        
        # Simple cache management (keep only last 50 entries)
        if len(self.answer_cache) > 50:
            first_key = next(iter(self.answer_cache))
            del self.answer_cache[first_key]

def main():
    """Test the advanced LLM integration."""
    print("🧪 Testing Advanced LLM Integration...")
    llm = LLMIntegration(use_ollama=True)
    llm.initialize()
    
    # Test with realistic chunks
    test_chunks = [
        ("Cloud computing deployment models include private cloud (operated solely for an organization), community cloud (shared by several organizations), public cloud (available to general public), and hybrid cloud (composition of multiple clouds).", 0.95, {}),
        ("Distributed computing involves multiple autonomous computers communicating through message passing. Key characteristics include fault tolerance, resource sharing, and scalability.", 0.85, {}),
        ("Artificial Intelligence refers to machines simulating human intelligence. Machine learning is a subset that enables learning from experience without explicit programming.", 0.75, {})
    ]
    
    test_questions = [
        "What are the types of cloud deployment models?",
        "What is distributed computing?",
        "Define artificial intelligence",
        "What are the key characteristics of distributed systems?"
    ]
    
    for question in test_questions:
        print(f"\n" + "="*60)
        print(f"❓ QUESTION: {question}")
        print("="*60)
        
        answer = llm.generate_answer(question, test_chunks)
        print(f"🤖 ANSWER: {answer}")

if __name__ == "__main__":
    main()
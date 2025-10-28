"""
LLM integration with Ollama local models - Fixed and Optimized
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
    
        # ALWAYS try LLM first for better quality answers
        if self.use_ollama:
            try:
                ollama_answer = self._call_ollama_api(question, relevant_chunks)
                if ollama_answer and ollama_answer.strip():
                    logger.info("✅ Using LLM-generated answer")
                    return ollama_answer
            except Exception as e:
                logger.warning(f"Ollama API failed: {str(e)}, using fallback")
    
        # Fallback - use the improved fallback
        return self._detailed_rule_based_fallback(question, relevant_chunks)
    
    def _call_ollama_api(self, question: str, relevant_chunks: List[Tuple[str, float, dict]]) -> Optional[str]:
        """
        Call Ollama API with optimized prompt and parameters.
        """
        try:
            # Build optimized context
            context = self._build_optimized_context(relevant_chunks)
            
            # Build dynamic prompt
            prompt = self._build_dynamic_prompt(question, context)
            
            # Call Ollama with retry
            return self._call_ollama_with_retry(prompt)
            
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            return None
    
    def _detailed_rule_based_fallback(self, question: str, relevant_chunks: List[Tuple[str, float, dict]]) -> str:
        """
        Advanced rule-based fallback when LLM is not available.
        Analyzes question type and extracts relevant information intelligently.
        """
        question_lower = question.lower().strip()
        
        # Extract all relevant text from chunks
        all_text = " ".join([self._clean_context_chunk(chunk) for chunk, _, _ in relevant_chunks])
        
        # Detect question type and extract accordingly
        if any(keyword in question_lower for keyword in ['what are', 'list', 'types of', 'categories']):
            return self._extract_list_answer(question, all_text, relevant_chunks)
        
        elif any(keyword in question_lower for keyword in ['what is', 'define', 'definition', 'meaning']):
            return self._extract_definition_answer(question, all_text, relevant_chunks)
        
        elif any(keyword in question_lower for keyword in ['how', 'explain', 'describe']):
            return self._extract_explanation_answer(question, all_text, relevant_chunks)
        
        elif any(keyword in question_lower for keyword in ['why', 'reason']):
            return self._extract_reasoning_answer(question, all_text, relevant_chunks)
        
        else:
            # General answer extraction
            return self._extract_general_answer(question, all_text, relevant_chunks)
    
    def _extract_list_answer(self, question: str, text: str, chunks: List[Tuple[str, float, dict]]) -> str:
        """Extract list-type answers (bullet points, enumeration)."""
        items = []
        
        # Find bulleted lists
        bullet_items = re.findall(r'[•\-*]\s*([^\n]+)', text)
        items.extend(bullet_items)
        
        # Find numbered lists
        numbered_items = re.findall(r'\d+[\.)]\s*([^\n]+)', text)
        items.extend(numbered_items)
        
        # Find colon-separated items (e.g., "Types: A, B, C")
        colon_lists = re.findall(r':\s*([^.!?\n]+(?:,\s*[^.!?\n]+)+)', text)
        for colon_list in colon_lists:
            items.extend([item.strip() for item in colon_list.split(',')])
        
        # Remove duplicates and clean
        unique_items = []
        seen = set()
        for item in items:
            cleaned = item.strip()
            if cleaned and len(cleaned) > 3 and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                unique_items.append(cleaned)
        
        if unique_items:
            if len(unique_items) <= 3:
                return "Based on the documents:\n\n" + "\n".join([f"• {item}" for item in unique_items[:10]])
            else:
                return "Based on the documents, here are the key points:\n\n" + "\n".join([f"• {item}" for item in unique_items[:10]])
        
        # Fallback to general extraction
        return self._extract_general_answer(question, text, chunks)
    
    def _extract_definition_answer(self, question: str, text: str, chunks: List[Tuple[str, float, dict]]) -> str:
        """Extract definition-type answers."""
        # Find definition patterns
        definition_patterns = [
            r'([A-Z][^.!?]*\b(?:is|are|refers to|means|defined as|can be defined as)[^.!?]*[.!?])',
            r'([A-Z][^.!?]*\b(?::\s*[A-Z])[^.!?]*[.!?])',
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Return the first good definition
                for match in matches[:2]:
                    if len(match) > 30:  # Ensure it's a substantial definition
                        return f"Based on the documents:\n\n{match.strip()}"
        
        # Fallback: return most relevant sentence
        sentences = re.split(r'[.!?]+', text)
        question_words = set(question.lower().split())
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
        
        if best_sentence:
            return f"Based on the documents:\n\n{best_sentence}."
        
        return "I cannot find a clear definition in the provided documents."
    
    def _extract_explanation_answer(self, question: str, text: str, chunks: List[Tuple[str, float, dict]]) -> str:
        """Extract explanation-type answers."""
        # Get most relevant chunk
        best_chunk = self._clean_context_chunk(chunks[0][0])
        
        # Extract key sentences
        sentences = re.split(r'[.!?]+', best_chunk)
        key_sentences = []
        
        for sentence in sentences[:5]:  # First 5 sentences
            sentence = sentence.strip()
            if len(sentence) > 30:
                key_sentences.append(sentence)
        
        if key_sentences:
            explanation = ". ".join(key_sentences[:3]) + "."
            return f"Based on the documents:\n\n{explanation}"
        
        return self._extract_general_answer(question, text, chunks)
    
    def _extract_reasoning_answer(self, question: str, text: str, chunks: List[Tuple[str, float, dict]]) -> str:
        """Extract reasoning/causation answers."""
        # Look for reasoning patterns
        reasoning_patterns = [
            r'([^.!?]*\b(?:because|since|due to|as a result|therefore|thus)[^.!?]*[.!?])',
            r'([^.!?]*\b(?:reason|cause|purpose)[^.!?]*[.!?])',
        ]
        
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                reasons = [m.strip() for m in matches if len(m.strip()) > 20]
                if reasons:
                    return f"Based on the documents:\n\n{reasons[0]}"
        
        return self._extract_general_answer(question, text, chunks)
    
    def _extract_general_answer(self, question: str, text: str, chunks: List[Tuple[str, float, dict]]) -> str:
        """Extract general answer when specific patterns don't match."""
        # Get the best chunk
        best_chunk = self._clean_context_chunk(chunks[0][0])
        
        # Extract most relevant part
        relevant_part = self._extract_most_relevant_part(best_chunk, question)
        
        if relevant_part and len(relevant_part) > 50:
            return f"Based on the documents:\n\n{relevant_part}"
        else:
            # Last resort: return first few sentences of best chunk
            sentences = re.split(r'[.!?]+', best_chunk)
            first_sentences = [s.strip() for s in sentences[:2] if len(s.strip()) > 20]
            if first_sentences:
                return f"Based on the documents:\n\n{'. '.join(first_sentences)}."
            
            return "I cannot provide a clear answer based on the provided documents."
    
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
                        return self._clean_and_format_answer(answer)
                
            except Exception as e:
                logger.warning(f"Ollama call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return None
    
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
        
        return best_sentence if best_score > 0 else text[:300]


def main():
    """Test the fixed LLM integration."""
    print("🧪 Testing Fixed LLM Integration...")
    llm = LLMIntegration(use_ollama=True)
    llm.initialize()
    
    # Test with realistic chunks
    test_chunks = [
        ("Cloud computing deployment models include private cloud (operated solely for an organization), community cloud (shared by several organizations), public cloud (available to general public), and hybrid cloud (composition of multiple clouds).", 0.95, {}),
        ("Distributed computing involves multiple autonomous computers communicating through message passing.", 0.85, {}),
    ]
    
    test_questions = [
        "What are the types of cloud deployment models?",
        "What is distributed computing?",
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ QUESTION: {question}")
        print('='*60)
        
        answer = llm.generate_answer(question, test_chunks)
        print(f"🤖 ANSWER: {answer}")


if __name__ == "__main__":
    main()
"""
LLM integration for generating intelligent answers from retrieved chunks.
"""
import logging
import re
from typing import List, Tuple, Optional
from src.config import config

logger = logging.getLogger(__name__)

class LLMIntegration:
    """Handles answer generation from retrieved chunks."""
    
    def __init__(self):
        self.is_initialized = True  # Always ready for rule-based
        self.model_name = "smart_rule_based"
    
    def initialize(self, use_openai: bool = False, api_key: str = None):
        """Initialize - for compatibility, but we're always ready."""
        self.is_initialized = True
        logger.info("Smart rule-based answer generator ready")
    
    def generate_answer(self, question: str, context_chunks: List[Tuple[str, float, dict]]) -> str:
        """
        Generate an answer using smart rule-based approach.
        
        Args:
            question: User's question
            context_chunks: List of (chunk_text, similarity_score, metadata)
            
        Returns:
            Generated answer
        """
        if not context_chunks:
            return "I couldn't find any relevant information in the documents to answer your question."
        
        try:
            return self._smart_rule_based_answer(question, context_chunks)
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return self._fallback_answer(context_chunks)
    
    def _smart_rule_based_answer(self, question: str, context_chunks: List[Tuple[str, float, dict]]) -> str:
        """Smart rule-based answer that extracts and formats relevant information."""
        if not context_chunks:
            return "No relevant information found."
        
        # Filter for highly relevant chunks
        relevant_chunks = [(chunk, score, meta) for chunk, score, meta in context_chunks if score > 0.3]
        
        if not relevant_chunks:
            return "The documents don't contain specific information about this topic."
        
        question_lower = question.lower()
        
        # Try to extract the most relevant sentence or phrase
        best_answer = self._extract_best_answer(question_lower, relevant_chunks)
        
        if best_answer:
            return best_answer
        else:
            # Fallback to well-formatted context
            return self._format_context_answer(question, relevant_chunks)
    
    def _extract_best_answer(self, question_lower: str, relevant_chunks: List[Tuple[str, float, dict]]) -> str:
        """Extract the best matching sentence or phrase."""
        all_text = " ".join([chunk for chunk, score, meta in relevant_chunks])
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', all_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Score sentences based on relevance to question
        scored_sentences = []
        for sentence in sentences:
            score = self._score_sentence_relevance(sentence.lower(), question_lower)
            if score > 0:
                scored_sentences.append((sentence, score))
        
        if scored_sentences:
            # Get the best sentence
            best_sentence, best_score = max(scored_sentences, key=lambda x: x[1])
            
            # Format the answer nicely
            if "what is" in question_lower or "what are" in question_lower:
                return f"Based on the document:\n\n{best_sentence}."
            elif "how" in question_lower:
                return f"According to the document:\n\n{best_sentence}."
            else:
                return f"Relevant information from the document:\n\n{best_sentence}."
        
        return None
    
    def _score_sentence_relevance(self, sentence: str, question: str) -> float:
        """Score how relevant a sentence is to the question."""
        score = 0
        
        # Keywords from question
        question_words = set(question.split())
        sentence_words = set(sentence.split())
        
        # Common words to ignore
        common_words = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'does', 'do', 'can', 'could'}
        question_words = question_words - common_words
        
        # Score based on keyword matches
        matches = question_words.intersection(sentence_words)
        if matches:
            score += len(matches) * 2
        
        # Bonus for definition-like sentences
        if any(word in sentence for word in ['is defined as', 'refers to', 'means that', 'is a']):
            score += 3
        
        # Bonus for explanation-like sentences
        if any(word in sentence for word in ['uses', 'works by', 'employs', 'utilizes']):
            score += 2
        
        return score
    
    def _format_context_answer(self, question: str, relevant_chunks: List[Tuple[str, float, dict]]) -> str:
        """Format a nice answer from the context chunks."""
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
    
    def _fallback_answer(self, context_chunks: List[Tuple[str, float, dict]]) -> str:
        """Simple fallback answer."""
        if not context_chunks:
            return "No information available."
        
        best_chunk, best_score, best_meta = context_chunks[0]
        clean_chunk = self._clean_text(best_chunk)
        
        return f"Based on the document:\n\n{clean_chunk}"

def main():
    """Test the smart rule-based integration."""
    llm = LLMIntegration()
    
    # Test with realistic chunks
    test_chunks = [
        ("Deep learning is a subset of machine learning that uses neural networks with multiple layers to analyze various factors of data. It is particularly useful for image recognition and natural language processing tasks.", 0.85, {}),
        ("Machine learning enables computers to learn without explicit programming by using algorithms that can analyze data and learn from it to make predictions or decisions.", 0.72, {})
    ]
    
    questions = [
        "What is deep learning?",
        "How does machine learning work?",
        "What are neural networks used for?"
    ]
    
    for question in questions:
        answer = llm.generate_answer(question, test_chunks)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("-" * 60)

if __name__ == "__main__":
    main()
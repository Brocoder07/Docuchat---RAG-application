"""
Universal hallucination detection that works across all document types.
"""
import re
from typing import List, Dict, Set, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class UniversalHallucinationDetector:
    """Detects hallucinations in LLM responses regardless of document type."""
    
    def __init__(self):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Universal hallucination patterns
        self.hallucination_indicators = {
            'speculation_phrases': {
                'probably', 'likely', 'might be', 'could be', 'would be',
                'i think', 'i believe', 'in my opinion', 'usually', 'typically',
                'generally', 'always', 'never', 'every', 'all'
            },
            'vague_quantifiers': {
                'several', 'many', 'few', 'some', 'various', 'multiple'
            },
            'subjective_language': {
                'amazing', 'terrible', 'perfect', 'best', 'worst', 'excellent'
            },
            'unsupported_claims': {
                'as we know', 'it is known', 'research shows', 'studies prove'
            }
        }
    
    def detect_hallucinations(self, answer: str, context_chunks: List, 
                            question: str = None) -> Dict[str, any]:
        """Comprehensive hallucination detection."""
        
        detectors = [
            self._detect_speculation,
            self._detect_context_mismatch,
            self._detect_entity_invention,
            self._detect_factual_contradiction,
            self._detect_unsupported_claims
        ]
        
        results = {}
        for detector in detectors:
            detector_name = detector.__name__.replace('_detect_', '')
            results[detector_name] = detector(answer, context_chunks, question)
        
        # Calculate overall hallucination score
        results['overall_hallucination_score'] = self._calculate_overall_score(results)
        results['is_hallucinating'] = results['overall_hallucination_score'] > 0.7
        
        return results
    
    def _detect_speculation(self, answer: str, context_chunks: List, question: str) -> Dict:
        """Detect speculative language in the answer."""
        answer_lower = answer.lower()
        
        speculation_count = 0
        detected_phrases = []
        
        for phrase in self.hallucination_indicators['speculation_phrases']:
            if phrase in answer_lower:
                speculation_count += 1
                detected_phrases.append(phrase)
        
        speculation_score = min(1.0, speculation_count / 5)  # Normalize
        
        return {
            'speculation_count': speculation_count,
            'detected_phrases': detected_phrases,
            'score': speculation_score
        }
    
    def _detect_context_mismatch(self, answer: str, context_chunks: List, question: str) -> Dict:
        """Detect when answer doesn't match the retrieved context."""
        if not context_chunks or not answer.strip():
            return {'score': 1.0, 'details': 'No context or answer available'}
        
        try:
            # Semantic similarity check
            answer_embedding = self.similarity_model.encode([answer])
            context_texts = [chunk[0] for chunk in context_chunks if chunk[0].strip()]
            
            if not context_texts:
                return {'score': 1.0, 'details': 'No valid context texts'}
            
            context_embeddings = self.similarity_model.encode(context_texts)
            similarities = np.dot(answer_embedding, context_embeddings.T)[0]
            max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0
            
            # Keyword grounding check
            answer_entities = self._extract_entities(answer)
            context_entities = self._extract_entities_from_chunks(context_chunks)
            
            novel_entities = answer_entities - context_entities
            
            # Filter out common words that aren't really entities
            common_words = {'the', 'and', 'or', 'but', 'with', 'for', 'from', 'this', 'that'}
            novel_entities = novel_entities - common_words
            
            mismatch_score = min(1.0, (1.0 - max_similarity) + (len(novel_entities) * 0.1))
            
            return {
                'semantic_similarity': max_similarity,
                'novel_entities': list(novel_entities),
                'score': mismatch_score
            }
            
        except Exception as e:
            logger.error(f"Error in context mismatch detection: {str(e)}")
            return {'score': 0.5, 'details': f'Error: {str(e)}'}
    
    def _detect_entity_invention(self, answer: str, context_chunks: List, question: str) -> Dict:
        """Detect when the LLM invents entities not in context."""
        answer_entities = self._extract_entities(answer)
        context_entities = self._extract_entities_from_chunks(context_chunks)
        
        invented_entities = answer_entities - context_entities
        
        # Filter out common words that aren't really entities
        common_words = {
            'the', 'and', 'or', 'but', 'with', 'for', 'from', 'this', 'that',
            'which', 'what', 'when', 'where', 'why', 'how', 'can', 'could', 'would', 'should'
        }
        invented_entities = invented_entities - common_words
        
        invention_score = len(invented_entities) / len(answer_entities) if answer_entities else 0
        
        return {
            'invented_entities': list(invented_entities),
            'invention_score': invention_score,
            'score': min(1.0, invention_score * 2)  # Amplify the score
        }
    
    def _detect_factual_contradiction(self, answer: str, context_chunks: List, question: str) -> Dict:
        """Detect factual contradictions (simplified version)."""
        # This is a simplified version - in production you'd use more sophisticated methods
        answer_sentences = re.split(r'[.!?]+', answer)
        contradiction_score = 0.0
        
        # Simple check for obvious contradictions
        contradiction_indicators = [
            ('always', 'never'),
            ('all', 'none'),
            ('every', 'no'),
        ]
        
        for always, never in contradiction_indicators:
            if always in answer.lower() and never in answer.lower():
                contradiction_score = 0.8
                break
        
        return {
            'contradiction_detected': contradiction_score > 0,
            'score': contradiction_score
        }
    
    def _detect_unsupported_claims(self, answer: str, context_chunks: List, question: str) -> Dict:
        """Detect unsupported claims and subjective language."""
        answer_lower = answer.lower()
        
        unsupported_count = 0
        detected_claims = []
        
        for category, phrases in self.hallucination_indicators.items():
            if category in ['subjective_language', 'unsupported_claims']:
                for phrase in phrases:
                    if phrase in answer_lower:
                        unsupported_count += 1
                        detected_claims.append(phrase)
        
        claim_score = min(1.0, unsupported_count / 3)  # Normalize
        
        return {
            'unsupported_count': unsupported_count,
            'detected_claims': detected_claims,
            'score': claim_score
        }
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract potential entities from text."""
        if not text:
            return set()
            
        entities = set()
        
        # Proper nouns (capitalized sequences)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update(proper_nouns)
        
        # Technical terms (mixed case with specific patterns)
        tech_terms = re.findall(r'\b[A-Za-z]+(?:[A-Z][a-z]+)+\b', text)  # CamelCase
        entities.update(tech_terms)
        
        # Lowercase everything for comparison
        entities = set(entity.lower() for entity in entities)
        
        return entities
    
    def _extract_entities_from_chunks(self, chunks: List) -> Set[str]:
        """Extract entities from all context chunks."""
        all_entities = set()
        
        for chunk in chunks:
            if isinstance(chunk, tuple) and len(chunk) > 0:
                chunk_text = chunk[0]
                chunk_entities = self._extract_entities(chunk_text)
                all_entities.update(chunk_entities)
        
        return all_entities
    
    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate overall hallucination score from individual detectors."""
        if not results:
            return 0.0
        
        scores = []
        weights = {
            'speculation': 0.2,
            'context_mismatch': 0.4,  # Most important
            'entity_invention': 0.3,
            'factual_contradiction': 0.05,
            'unsupported_claims': 0.05
        }
        
        for detector_name, result in results.items():
            if detector_name in weights and isinstance(result, dict) and 'score' in result:
                scores.append(result['score'] * weights[detector_name])
        
        return min(1.0, sum(scores)) if scores else 0.0
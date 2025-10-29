"""
Universal context verification to prevent hallucinations for ANY document type.
"""
import re
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class UniversalContextVerifier:
    """Verifies that LLM answers are grounded in the retrieved context."""
    
    def __init__(self):
        self.hallucination_indicators = [
            # Generic unsupported claims
            r'\b(?:research shows|studies prove|it is known|as we know|experts say)\b',
            # Vague quantifiers without specific references
            r'\b(?:many|several|various|multiple|some|few)\s+(?:people|studies|experts)\b',
            # Speculative language
            r'\b(?:probably|likely|might be|could be|would be|i think|i believe)\b',
            # Absolute statements without evidence
            r'\b(?:always|never|every|all|none)\b',
            # External references not in context
            r'\b(?:according to|as per|based on)\s+(?!the context|the document|the provided|chunk)',
        ]
    
    def verify_answer_grounding(self, answer: str, context_chunks: List[Tuple[str, float, Dict]]) -> Dict[str, Any]:
        """
        Verify that the answer is properly grounded in the context.
        Returns verification results with confidence score.
        """
        if not answer or not context_chunks:
            return {
                "is_grounded": False,
                "confidence": 0.0,
                "issues": ["No answer or context provided"],
                "suggested_correction": "I cannot verify this information."
            }
        
        # Extract all context text
        context_text = " ".join([chunk[0] for chunk in context_chunks]).lower()
        answer_lower = answer.lower()
        
        verification_results = {
            "is_grounded": True,
            "confidence": 1.0,
            "issues": [],
            "missing_citations": [],
            "hallucination_flags": []
        }
        
        # Check for hallucination indicators
        self._check_hallucination_indicators(answer, verification_results)
        
        # Verify key claims are in context
        self._verify_key_claims(answer, context_text, verification_results)
        
        # Check for proper citations
        self._check_citations(answer, context_chunks, verification_results)
        
        # Calculate overall confidence
        verification_results["confidence"] = self._calculate_confidence(verification_results)
        verification_results["is_grounded"] = verification_results["confidence"] > 0.6
        
        return verification_results
    
    def _check_hallucination_indicators(self, answer: str, results: Dict):
        """Check for language that indicates potential hallucinations."""
        for pattern in self.hallucination_indicators:
            matches = re.findall(pattern, answer.lower())
            if matches:
                results["hallucination_flags"].extend(matches)
                results["issues"].append(f"Used speculative language: {matches}")
    
    def _verify_key_claims(self, answer: str, context_text: str, results: Dict):
        """Verify that key factual claims in the answer appear in context."""
        # Extract factual statements (sentences without speculative language)
        sentences = re.split(r'[.!?]+', answer)
        key_claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                not any(indicator in sentence.lower() for indicator in ['i think', 'probably', 'might be'])):
                key_claims.append(sentence)
        
        # Check if key claims are supported by context
        unsupported_claims = []
        for claim in key_claims[:5]:  # Check first 5 key claims
            claim_clean = re.sub(r'\s+', ' ', claim.lower().strip())
            if claim_clean and claim_clean not in context_text:
                # Check for partial matches
                words = claim_clean.split()
                if len(words) > 3:
                    # Require at least 50% of key words to be in context
                    matching_words = sum(1 for word in words if len(word) > 3 and word in context_text)
                    if matching_words / len(words) < 0.5:
                        unsupported_claims.append(claim)
        
        if unsupported_claims:
            results["issues"].append(f"Unsupported claims: {unsupported_claims}")
            results["missing_citations"] = unsupported_claims
    
    def _check_citations(self, answer: str, context_chunks: List[Tuple], results: Dict):
        """Check if the answer properly cites the context chunks."""
        # Look for chunk references in the answer
        chunk_refs = re.findall(r'chunk\s*(\d+)', answer.lower())
        if not chunk_refs:
            results["issues"].append("No specific chunk references found")
        
        # Verify referenced chunks exist
        valid_refs = []
        for ref in chunk_refs:
            chunk_num = int(ref)
            if 1 <= chunk_num <= len(context_chunks):
                valid_refs.append(ref)
            else:
                results["issues"].append(f"Reference to non-existent chunk {chunk_num}")
        
        if not valid_refs and len(context_chunks) > 1:
            results["issues"].append("Answer doesn't cite specific sources")
    
    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate overall confidence score."""
        confidence = 1.0
        
        # Penalize for each issue
        penalties = {
            "hallucination_flags": 0.2,
            "unsupported_claims": 0.3, 
            "missing_citations": 0.1,
            "no_chunk_references": 0.1
        }
        
        for issue in results["issues"]:
            for penalty_type, penalty in penalties.items():
                if penalty_type in issue.lower():
                    confidence -= penalty
        
        return max(0.0, min(1.0, confidence))
    
    def generate_safe_response(self, original_answer: str, verification_results: Dict) -> str:
        """Generate a safe response when hallucinations are detected."""
        if verification_results["confidence"] > 0.7:
            return original_answer
        
        issues = verification_results["issues"]
        safe_prefix = "Based on the provided documents, "
        
        if verification_results["confidence"] < 0.4:
            return f"{safe_prefix}I cannot find sufficient information to answer this question accurately. The available content doesn't contain the specific details you're looking for."
        
        # For medium confidence, be cautious but provide the answer
        caution_note = " Please note that some information may not be fully supported by the available documents."
        return safe_prefix + original_answer + caution_note
"""
Builds context that adapts to ANY document type while preventing hallucination.
"""
from typing import List, Dict, Tuple
import re
import logging

logger = logging.getLogger(__name__)

class AdaptiveContextBuilder:
    """Builds context that works for resumes, reports, articles, etc."""
    
    def __init__(self):
        self.document_patterns = {
            'resume': [
                r'\b(?:resume|cv|experience|education|skills|projects)\b',
                r'\b\d{4}\s*[-–]\s*(?:present|\d{4})\b'
            ],
            'technical_doc': [
                r'\b(?:api|endpoint|database|server|client|framework)\b',
                r'\b(?:GET|POST|PUT|DELETE|PATCH)\b'
            ],
            'research_paper': [
                r'\b(?:abstract|introduction|methodology|conclusion|references)\b',
                r'\b\d+\.\d+\b'  # Version numbers
            ],
            'business_report': [
                r'\b(?:executive summary|recommendations|q[1-4]|fy\d{4})\b',
                r'\$?\d+(?:,\d+)*(?:\.\d+)?%?'  # Financial numbers
            ]
        }
    
    def build_universal_context(self, chunks: List, question: str) -> str:
        """Build context that works for any document type."""
        
        if not chunks:
            return "No context available."
        
        # Extract universal structure
        structure = self._analyze_document_structure(chunks)
        
        # Build context with clear boundaries
        context = self._build_structured_context(chunks, structure)
        
        # Add universal anti-hallucination rules
        context += self._add_universal_rules(structure, question)
        
        return context
    
    def _analyze_document_structure(self, chunks: List) -> Dict:
        """Analyze document structure without assuming document type."""
        all_text = " ".join([chunk[0] for chunk in chunks if isinstance(chunk, tuple) and len(chunk) > 0])
        
        structure = {
            'has_headings': self._detect_headings(all_text),
            'has_lists': self._detect_lists(all_text),
            'has_dates': self._detect_dates(all_text),
            'has_proper_nouns': self._detect_proper_nouns(all_text),
            'estimated_doc_type': self._estimate_document_type(all_text),
            'content_units': len(chunks)
        }
        
        return structure
    
    def _detect_headings(self, text: str) -> bool:
        """Detect if text contains heading-like structures."""
        heading_patterns = [
            r'\n\s*#+\s+.+',
            r'\n\s*[A-Z][A-Za-z\s]+\n\s*[-=]+\s*\n',
            r'\n\s*[A-Z][A-Z ]+:\s*\n',
        ]
        
        for pattern in heading_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _detect_lists(self, text: str) -> bool:
        """Detect if text contains list structures."""
        list_patterns = [
            r'\n\s*[•\-*]\s+',
            r'\n\s*\d+\.\s+',
            r'\n\s*[a-z]\)\s+',
        ]
        
        for pattern in list_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _detect_dates(self, text: str) -> bool:
        """Detect if text contains dates."""
        date_patterns = [
            r'\b\d{4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _detect_proper_nouns(self, text: str) -> bool:
        """Detect if text contains proper nouns."""
        proper_noun_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        return bool(re.search(proper_noun_pattern, text))
    
    def _estimate_document_type(self, text: str) -> str:
        """Estimate document type based on content patterns."""
        scores = {}
        
        for doc_type, patterns in self.document_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                score += len(matches)
            scores[doc_type] = score
        
        # Return the type with highest score, or 'general' if no clear winner
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            return best_type[0] if best_type[1] > 0 else 'general'
        
        return 'general'
    
    def _build_structured_context(self, chunks: List, structure: Dict) -> str:
        """Build structured context based on detected document features."""
        context_parts = []
        
        # Add document type hint
        doc_type = structure.get('estimated_doc_type', 'general')
        context_parts.append(f"DOCUMENT TYPE: {doc_type.upper()}")
        context_parts.append("CONTENT UNITS:")
        context_parts.append("")
        
        # Add chunks with their natural boundaries preserved
        for i, chunk_data in enumerate(chunks[:6]):  # Limit context size for efficiency
            if isinstance(chunk_data, tuple) and len(chunk_data) >= 2:
                chunk, score = chunk_data[0], chunk_data[1]
                context_parts.append(f"--- UNIT {i+1} (relevance: {score:.3f}) ---")
                
                # Clean and truncate if too long
                clean_chunk = self._clean_chunk_text(chunk)
                if len(clean_chunk) > 500:
                    clean_chunk = clean_chunk[:497] + "..."
                
                context_parts.append(clean_chunk)
                context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean chunk text for better context."""
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        return text
    
    def _add_universal_rules(self, structure: Dict, question: str) -> str:
        """Add universal rules that prevent hallucination."""
        
        rules = [
            "\n" + "="*50,
            "CRITICAL ANSWERING INSTRUCTIONS:",
            "="*50,
            "1. Use ONLY the information provided in the context above",
            "2. Do not add, infer, or assume any information not explicitly stated",
            "3. If information is missing, say 'The context does not provide information about X'",
            "4. Keep different content units separate - do not mix information across boundaries",
            "5. Be precise and factual - avoid speculative language",
            "6. If unsure, acknowledge the limitation rather than guessing",
        ]
        
        # Add document-type specific rules
        doc_type = structure.get('estimated_doc_type', 'general')
        if doc_type == 'resume':
            rules.extend([
                "7. Keep projects, roles, and companies completely separate",
                "8. Only mention technologies, skills, or experiences explicitly listed",
                "9. Do not infer relationships between different companies or roles",
            ])
        elif doc_type == 'technical_doc':
            rules.extend([
                "7. Be precise about technical specifications and APIs",
                "8. Do not infer undocumented features or behaviors",
                "9. Only mention parameters, endpoints, or methods that are explicitly described",
            ])
        elif doc_type == 'research_paper':
            rules.extend([
                "7. Only reference findings, methods, or conclusions explicitly stated",
                "8. Do not extrapolate or generalize beyond what's written",
                "9. Be precise about study parameters and results",
            ])
        elif doc_type == 'business_report':
            rules.extend([
                "7. Only reference financial numbers, metrics, or dates explicitly provided",
                "8. Do not infer trends or performance beyond what's stated",
                "9. Be precise about timeframes and quantitative data",
            ])
        
        # Add question-specific guidance
        if question:
            question_lower = question.lower()
            if any(word in question_lower for word in ['compare', 'difference', 'versus']):
                rules.append("10. Only compare elements that are explicitly mentioned together in the context")
            if any(word in question_lower for word in ['why', 'reason', 'because']):
                rules.append("10. Only provide reasons that are explicitly stated in the context")
        
        rules.append("\nANSWER:")
        
        return "\n".join(rules)
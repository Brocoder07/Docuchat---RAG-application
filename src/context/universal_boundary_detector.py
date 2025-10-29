"""
Universal boundary detection that works across all document types.
"""
import re
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)

class UniversalBoundaryDetector:
    """Detects natural boundaries in ANY document to prevent content mixing."""
    
    def __init__(self):
        # Universal boundary markers that work across document types
        self.boundary_patterns = {
            'headings': [
                r'\n\s*#+\s+.+',  # Markdown headings
                r'\n\s*[A-Z][A-Za-z\s]+\n\s*[-=]+\s*\n',  # Underlined headings
                r'\n\s*\b(?:CHAPTER|SECTION|PROJECT|COMPANY|ROLE)\b.*\n',
            ],
            'list_boundaries': [
                r'\n\s*[•\-*]\s+',  # Bullet points
                r'\n\s*\d+\.\s+',   # Numbered lists
            ],
            'structural_boundaries': [
                r'\n\s*\n',  # Multiple newlines
                r'---+\s*\n',  # Horizontal rules
            ]
        }
    
    def detect_content_units(self, text: str) -> List[Dict]:
        """Detect natural content units in ANY document."""
        units = []
        
        # Split by major boundaries first
        major_chunks = self._split_by_major_boundaries(text)
        
        for chunk in major_chunks:
            # Further split by sub-boundaries if needed
            sub_units = self._split_by_sub_boundaries(chunk)
            units.extend(sub_units)
        
        return units
    
    def _split_by_major_boundaries(self, text: str) -> List[str]:
        """Split text by major structural boundaries."""
        # Combine all heading patterns
        heading_pattern = '|'.join(self.boundary_patterns['headings'])
        
        # Split at headings but keep them with their content
        chunks = []
        current_chunk = ""
        
        lines = text.split('\n')
        for line in lines:
            if re.match(heading_pattern, '\n' + line) if line else False:
                # New major section
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_sub_boundaries(self, text: str) -> List[Dict]:
        """Split text by sub-boundaries like lists and paragraphs."""
        units = []
        
        # Split by multiple newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para in paragraphs:
            if para.strip():
                units.append({
                    'content': para.strip(),
                    'type': 'paragraph',
                    'boundary_markers': self._extract_boundary_markers(para)
                })
        
        return units
    
    def extract_entity_boundaries(self, text: str) -> Dict[str, Set]:
        """Extract entities and their natural boundaries from any text."""
        entities = {
            'proper_nouns': self._extract_proper_nouns(text),
            'technical_terms': self._extract_technical_terms(text),
            'numeric_entities': self._extract_numeric_entities(text),
            'date_entities': self._extract_date_entities(text),
        }
        
        return entities
    
    def _extract_proper_nouns(self, text: str) -> Set[str]:
        """Extract proper nouns that might represent entities."""
        # Simple proper noun detection (capitalized words in specific contexts)
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Project|App|System|Platform|Tool)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|LLC|Corp|Company)\b',
            r'\b(?:The\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Simple proper nouns
        ]
        
        entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)
        
        return entities
    
    def _extract_technical_terms(self, text: str) -> Set[str]:
        """Extract technical terms and technologies."""
        # Common technical patterns
        patterns = [
            r'\b[A-Za-z]+(?:\s+[A-Z][a-z]+)*\s+(?:API|SDK|Framework|Library|Toolkit)\b',
            r'\b(?:Python|Java|JavaScript|Kotlin|Swift|Go|Rust|C\+\+|C#)\b',
            r'\b(?:React|Vue|Angular|Django|Flask|Spring|FastAPI|Express)\b',
            r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch)\b',
        ]
        
        entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update(matches)
        
        return entities
    
    def _extract_numeric_entities(self, text: str) -> Set[str]:
        """Extract numeric entities like percentages, measurements, etc."""
        patterns = [
            r'\b\d+(?:\.\d+)?%',  # Percentages
            r'\$\d+(?:,\d+)*(?:\.\d+)?',  # Currency
            r'\b\d+(?:\.\d+)?\s*(?:GB|MB|KB|TB)',  # Storage sizes
        ]
        
        entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)
        
        return entities
    
    def _extract_date_entities(self, text: str) -> Set[str]:
        """Extract date entities."""
        patterns = [
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
        ]
        
        entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)
        
        return entities
    
    def _extract_boundary_markers(self, text: str) -> List[str]:
        """Extract boundary markers from text."""
        markers = []
        
        for boundary_type, patterns in self.boundary_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    markers.append(boundary_type)
                    break
        
        return markers
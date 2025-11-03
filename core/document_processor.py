"""
Intelligent document processor with multiple PDF backend support.
Senior Engineer Principle: Robust extraction with multiple fallbacks.
"""
import logging
import os
import re
from typing import List, Dict, Optional
import pandas as pd

from core.config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Smart document processor with format detection and optimized chunking.
    Senior Engineer Principle: Handle multiple file types gracefully.
    """
    
    def __init__(self):
        self.supported_extensions = config.files.ALLOWED_EXTENSIONS
    
    def process_file(self, file_path: str, filename: str) -> Dict[str, any]:
        """
        Process any supported file type with proper error handling.
        Returns structured response with chunks and metadata.
        """
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "chunks": []
            }
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension not in self.supported_extensions:
            return {
                "success": False,
                "error": f"Unsupported file type: {file_extension}",
                "chunks": []
            }
        
        try:
            logger.info(f"📄 Processing document: {filename}")
            
            # Extract text based on file type
            if file_extension == '.pdf':
                text = self._extract_pdf_text(file_path)
            elif file_extension == '.txt':
                text = self._extract_text_file(file_path)
            elif file_extension == '.docx':
                text = self._extract_docx_file(file_path)
            elif file_extension == '.md':
                text = self._extract_markdown_file(file_path)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_extension}",
                    "chunks": []
                }
            
            if not text or not text.strip():
                return {
                    "success": False,
                    "error": "No text content extracted from document",
                    "chunks": []
                }
            
            # Smart chunking
            chunks = self._smart_chunking(text)
            
            logger.info(f"✅ Successfully processed {filename}: {len(chunks)} chunks")
            
            return {
                "success": True,
                "chunks": chunks,
                "total_characters": len(text),
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunks": []
            }
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF with multiple fallback libraries."""
        text = ""
        
        # Try pdfplumber first (best for structured text)
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            if text.strip():
                logger.debug("Used pdfplumber for PDF extraction")
                return self._clean_text(text)
        except ImportError:
            logger.warning("pdfplumber not available")
        except Exception as e:
            logger.debug(f"pdfplumber failed: {str(e)}")
        
        # Try PyMuPDF (fitz) as second option
        try:
            import fitz
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n\n"
            doc.close()
            if text.strip():
                logger.debug("Used PyMuPDF for PDF extraction")
                return self._clean_text(text)
        except ImportError:
            logger.warning("PyMuPDF not available")
        except Exception as e:
            logger.debug(f"PyMuPDF failed: {str(e)}")
        
        # Try PyPDF2 as last resort
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            if text.strip():
                logger.debug("Used PyPDF2 for PDF extraction")
                return self._clean_text(text)
        except ImportError:
            logger.warning("PyPDF2 not available")
        except Exception as e:
            logger.debug(f"PyPDF2 failed: {str(e)}")
        
        raise Exception("No PDF extraction library available. Install pdfplumber, PyMuPDF, or PyPDF2")
    
    def _extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    
    def _extract_docx_file(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text
        except ImportError:
            raise Exception("python-docx not installed. Run: pip install python-docx")
    
    def _extract_markdown_file(self, file_path: str) -> str:
        """Extract text from Markdown file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text while preserving structure."""
        # Remove excessive whitespace but preserve paragraphs
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)([A-Z][a-z])', r'\1 \2', text)  # Add space between words
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        
        return text.strip()
    
    def _smart_chunking(self, text: str) -> List[str]:
        """
        Intelligent chunking that preserves semantic boundaries.
        Senior Engineer Principle: Balance chunk size with content coherence.
        """
        if not text:
            return []
        
        # First, try to split by major sections (headings, etc.)
        sections = self._split_into_sections(text)
        
        chunks = []
        for section in sections:
            if len(section) <= config.rag.CHUNK_SIZE:
                # Section is small enough to be one chunk
                if section.strip():
                    chunks.append(section.strip())
            else:
                # Split large sections by sentences/paragraphs
                section_chunks = self._split_section(section)
                chunks.extend(section_chunks)
        
        # Final validation
        valid_chunks = [chunk for chunk in chunks if chunk and len(chunk) > 50]
        
        logger.debug(f"Created {len(valid_chunks)} chunks from {len(text)} characters")
        return valid_chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Better section splitting for resumes."""
        # Split by major resume sections (case insensitive)
        sections = re.split(r'\n\s*(EDUCATION|TECHNICAL SKILLS|EXPERIENCE|PROJECTS|PUBLICATIONS|CERTIFICATIONS)\s*\n', text, flags=re.IGNORECASE)
        
        # Reconstruct sections with their headers
        refined_sections = []
        i = 0
        while i < len(sections):
            if sections[i].strip() and i + 1 < len(sections):
                # This is content, previous was header
                header = sections[i-1] if i > 0 else "HEADER"
                content = sections[i]
                refined_sections.append(f"{header}\n{content}")
                i += 1
            elif sections[i].strip():
                refined_sections.append(sections[i])
            i += 1
        
        return refined_sections
    
    def _split_section(self, text: str) -> List[str]:
        """Split a section into chunks of appropriate size."""
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_chunk and len(current_chunk) + len(sentence) > config.rag.CHUNK_SIZE:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - config.rag.CHUNK_OVERLAP)
                current_chunk = current_chunk[overlap_start:] + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

# Global document processor instance
document_processor = DocumentProcessor()
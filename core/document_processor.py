# core/document_processor.py (FULL file - replace your existing file)
"""
Intelligent document processor with heading normalization and robust chunking.
- Normalizes common resume headings (case-insensitive).
- Inserts explicit separators to ensure the splitter groups headings with their content.
- Uses a smaller chunk size (taken from config.rag) to create multiple chunks for resumes.
"""
import logging
import os
import re
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Smart document processor with format detection and optimized chunking.
    """
    def __init__(self):
        self.supported_extensions = config.files.ALLOWED_EXTENSIONS

        # Common resume/section headings to normalize (case-insensitive)
        self.headings = [
            r'EDUCATION',
            r'TECHNICAL SKILLS',
            r'SKILLS',
            r'EXPERIENCE',
            r'PROJECTS',
            r'PUBLICATIONS',
            r'CERTIFICATIONS',
            r'ACHIEVEMENTS',
            r'INTERESTS',
            r'CONTACT',
            r'WORK EXPERIENCE',
            r'PROFILE',
            r'SUMMARY',
        ]

        # Build separators so that headings are treated as section boundaries.
        # We include blank-line separators and single newline fallback.
        self.resume_separators = [
            "\n\n",  # prefer double newlines
            "\n",    # fallback single newline
            " ",     # fallback space
            ""       # ultimate fallback
        ]

        # create the splitter with values from config.rag
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.CHUNK_SIZE,
            chunk_overlap=config.rag.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
            separators=self.resume_separators
        )

    def process_file(self, file_path: str, filename: str) -> Dict[str, any]:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path}", "chunks": []}

        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in self.supported_extensions:
            return {"success": False, "error": f"Unsupported file type: {file_extension}", "chunks": []}

        try:
            logger.info(f"📄 Processing document: {filename}")

            # Extract raw text based on file type
            if file_extension == '.pdf':
                text = self._extract_pdf_text(file_path)
            elif file_extension == '.txt':
                text = self._extract_text_file(file_path)
            elif file_extension == '.docx':
                text = self._extract_docx_file(file_path)
            elif file_extension == '.md':
                text = self._extract_markdown_file(file_path)
            else:
                return {"success": False, "error": f"Unsupported file type: {file_extension}", "chunks": []}

            if not text or not text.strip():
                return {"success": False, "error": "No text content extracted from document", "chunks": []}

            # Normalize text to improve splitting:
            normalized = self._normalize_headings(text)

            # Use the splitter to produce chunks
            chunks = self.text_splitter.split_text(normalized)

            # Ensure returned chunks are strings and trimmed
            cleaned_chunks = []
            for c in chunks:
                if c is None:
                    continue
                s = c.strip()
                if s:
                    cleaned_chunks.append(s)

            if not cleaned_chunks:
                return {"success": False, "error": "No chunks generated after cleaning", "chunks": []}

            logger.info(f"✅ Successfully processed {filename}: {len(cleaned_chunks)} chunks")
            return {"success": True, "chunks": cleaned_chunks, "total_characters": len(normalized), "filename": filename}

        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "chunks": []}

    def _normalize_headings(self, text: str) -> str:
        """
        Normalize common section headings to a consistent form and ensure
        they are surrounded by blank lines so the text splitter can work.
        """
        # Replace Windows line endings
        t = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive repeated newlines (keep at most 2)
        t = re.sub(r'\n{3,}', '\n\n', t)

        # Normalize headings: find headings as whole words (case-insensitive)
        for heading in self.headings:
            # regex: beginning of line, optional bullets/spaces, heading, word boundary
            pattern = re.compile(rf'(?im)^[\s\-\•\*]*({heading})\s*:?$', re.IGNORECASE | re.MULTILINE)
            # replace with standardized boxed heading plus double newline before and after
            t = pattern.sub(lambda m: f"\n\n{m.group(1).upper()}\n", t)

        # Also convert lines that look like "EDUCATION\n•" or "EDUCATION\n• ..." into "EDUCATION\n"
        t = re.sub(r'(?m)^(EDUCATION|EXPERIENCE|PROJECTS|TECHNICAL SKILLS|CERTIFICATIONS|PUBLICATIONS|SUMMARY|PROFILE)\s*\n[\s\-\•\*]+', r'\1\n', t)

        # Ensure there is a final newline
        if not t.endswith("\n"):
            t += "\n"

        return t

    # --- Extraction helpers (unchanged logic but with clear exceptions) ---

    def _extract_pdf_text(self, file_path: str) -> str:
        text = ""
        # pdfplumber preferred
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
            logger.debug(f"pdfplumber failed: {e}")

        # Try PyMuPDF
        try:
            import fitz
            doc = fitz.open(file_path)
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n\n"
            doc.close()
            if text.strip():
                logger.debug("Used PyMuPDF for PDF extraction")
                return self._clean_text(text)
        except ImportError:
            logger.warning("PyMuPDF not available")
        except Exception as e:
            logger.debug(f"PyMuPDF failed: {e}")

        # Try PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            for page in reader.pages:
                p_text = page.extract_text()
                if p_text:
                    text += p_text + "\n\n"
            if text.strip():
                logger.debug("Used PyPDF2 for PDF extraction")
                return self._clean_text(text)
        except ImportError:
            logger.warning("PyPDF2 not available")
        except Exception as e:
            logger.debug(f"PyPDF2 failed: {e}")

        raise Exception("No PDF extraction library available. Install pdfplumber, PyMuPDF, or PyPDF2")

    def _extract_text_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _extract_docx_file(self, file_path: str) -> str:
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text and paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text
        except ImportError:
            raise Exception("python-docx not installed. Run: pip install python-docx")

    def _extract_markdown_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _clean_text(self, text: str) -> str:
        # preserve paragraphs, remove excessive whitespace
        txt = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        txt = re.sub(r'[ \t]+', ' ', txt)
        # fix hyphenation across line breaks
        txt = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', txt)
        # ensure consistent newlines
        txt = txt.replace('\r\n', '\n').replace('\r', '\n')
        return txt.strip()

# Global document processor instance
document_processor = DocumentProcessor()
"""
Basic document processing functionality.
"""
import os
import logging
from typing import List, Optional
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and basic text extraction."""
    
    def __init__(self):
        self.supported_extensions = config.supported_extensions
    
    def load_document(self, file_path: str) -> Optional[str]:
        """
        Load a document and extract its text content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.txt':
                return self._load_text_file(file_path)
            elif file_extension == '.pdf':
                return self._load_pdf_file(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return None
    
    def _load_text_file(self, file_path: str) -> str:
        """Load text from a .txt file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _load_pdf_file(self, file_path: str) -> str:
        """Load text from a PDF file."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            logger.error("PyPDF2 not installed. Please install it with: pip install pypdf2")
            return ""
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = config.chunking.chunk_size
        if chunk_overlap is None:
            chunk_overlap = config.chunking.chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size")
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position for next chunk
            start += chunk_size - chunk_overlap
            
            # If we're at the end, break
            if start >= len(text):
                break
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks

def main():
    """Test the document processor with a simple example."""
    processor = DocumentProcessor()
    
    # Create a test text file
    test_file = os.path.join(config.raw_docs_dir, "test_document.txt")
    with open(test_file, 'w') as f:
        f.write("This is a test document. " * 50)  # Create some repetitive text
    
    # Test loading
    text = processor.load_document(test_file)
    if text:
        print(f"Loaded text length: {len(text)} characters")
        
        # Test chunking
        chunks = processor.chunk_text(text)
        print(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"Chunk {i+1}: {chunk[:100]}...")
    else:
        print("Failed to load document")

if __name__ == "__main__":
    main()
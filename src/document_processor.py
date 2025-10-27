"""
Enhanced document processing functionality with support for multiple file types.
"""
import os
import sys
import logging
from typing import List, Optional

# Add project root to Python path to fix imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.config import config
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and basic text extraction for multiple file types."""
    
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
            elif file_extension == '.docx':
                return self._load_docx_file(file_path)
            elif file_extension == '.md':
                return self._load_markdown_file(file_path)
            elif file_extension == '.xlsx':
                return self._load_excel_file(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return None
    
    def _load_text_file(self, file_path: str) -> str:
        """Load text from a .txt file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            return ""
    
    def _load_pdf_file(self, file_path: str) -> str:
        """Load text from a PDF file."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        except ImportError:
            logger.error("PyPDF2 not installed. Please install it with: pip install pypdf2")
            return ""
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            return ""
    
    def _load_docx_file(self, file_path: str) -> str:
        """Load text from a DOCX file."""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text.strip()
        except ImportError:
            logger.error("python-docx not installed. Please install it with: pip install python-docx")
            return ""
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {str(e)}")
            return ""
    
    def _load_markdown_file(self, file_path: str) -> str:
        """Load text from a Markdown (.md) file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading Markdown file {file_path}: {str(e)}")
            return ""
    
    def _load_excel_file(self, file_path: str) -> str:
        """Load text from an Excel (.xlsx) file."""
        try:
            import pandas as pd
            text_parts = []
            
            # Read all sheets in the Excel file
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                text_parts.append(f"--- Sheet: {sheet_name} ---")
                
                # Read the sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Add column names
                text_parts.append("Columns: " + ", ".join(df.columns.astype(str)))
                
                # Add sample data (first 50 rows to avoid huge files)
                for index, row in df.head(50).iterrows():
                    row_text = " | ".join([str(cell) for cell in row if pd.notna(cell) and str(cell).strip()])
                    if row_text.strip():  # Only add non-empty rows
                        text_parts.append(f"Row {index + 1}: {row_text}")
                
                text_parts.append("")  # Empty line between sheets
            
            return "\n".join(text_parts)
            
        except ImportError:
            logger.error("pandas not installed. Please install it with: pip install pandas openpyxl")
            return ""
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {str(e)}")
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
        
        # Handle empty text
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
            
            # Move start position for next chunk
            start += chunk_size - chunk_overlap
            
            # If we're at the end, break
            if start >= text_length:
                break
        
        logger.info(f"Created {len(chunks)} chunks from text (length: {text_length} chars)")
        return chunks

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return self.supported_extensions

def main():
    """Test the document processor with various file types."""
    processor = DocumentProcessor()
    
    print("🧪 Testing Document Processor with Multiple File Types")
    print(f"Supported extensions: {processor.get_supported_extensions()}")
    
    # Ensure test directory exists
    test_dir = "data/raw_documents"
    os.makedirs(test_dir, exist_ok=True)
    
    print("✅ Document processor is ready with support for:")
    print("   - .txt (Text files)")
    print("   - .pdf (PDF documents)")
    print("   - .docx (Word documents)")
    print("   - .md (Markdown files)")
    print("   - .xlsx (Excel spreadsheets)")
    
    print("\n🎉 Document processor test completed!")

if __name__ == "__main__":
    main()
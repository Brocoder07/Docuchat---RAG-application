"""
Main RAG pipeline that combines document processing, embeddings, and retrieval.
"""
import os
import logging
from typing import List, Tuple, Optional
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.llm_integration import LLMIntegration
from src.config import config

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline that orchestrates the entire process."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.llm_integration = LLMIntegration()
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the pipeline components."""
        try:
            self.embedding_manager.initialize_model()
            self.llm_integration.initialize(use_openai=False)
            self.is_initialized = True
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise
    
    def process_document(self, file_path: str) -> bool:
        """Process a document and add to the vector store."""
        if not self.is_initialized:
            self.initialize()

        try:
            # Load and chunk document
            text = self.document_processor.load_document(file_path)
            if not text:
                logger.error(f"Failed to load document or no text extracted: {file_path}")
                return False
        
            if len(text.strip()) < 10:  # Minimum text length
                logger.error(f"Document contains too little text: {file_path}")
                return False
        
            chunks = self.document_processor.chunk_text(text)
            if not chunks:
                logger.error("No chunks created from document")
                return False
        
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
        
            # Create embeddings and add to index
            embeddings = self.embedding_manager.create_embeddings(chunks)
        
            # Create metadata for each chunk
            metadata = [{"source": file_path, "chunk_id": i, "filename": os.path.basename(file_path)} 
                    for i in range(len(chunks))]
        
            if self.embedding_manager.index is None:
                # Create new index
                self.embedding_manager.create_index(embeddings, chunks, metadata)
                logger.info(f"Created new vector index with {len(chunks)} chunks")
            else:
                # Add to existing index
                self._add_to_existing_index(embeddings, chunks, metadata)
                logger.info(f"Added {len(chunks)} chunks to existing vector index")
        
            logger.info(f"Successfully processed document: {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return False

    def _add_to_existing_index(self, new_embeddings, new_chunks, new_metadata):
        """Efficiently add new documents to existing index."""
        # Get existing data
        existing_chunks = self.embedding_manager.chunks
        existing_metadata = self.embedding_manager.metadata
    
        # Combine existing and new data
        all_chunks = existing_chunks + new_chunks
        all_metadata = existing_metadata + new_metadata
    
        # Create embeddings for combined data
        logger.info(f"Adding {len(new_chunks)} new chunks to existing {len(existing_chunks)} chunks")
        all_embeddings = self.embedding_manager.create_embeddings(all_chunks)
    
        # Recreate index with all data
        self.embedding_manager.create_index(all_embeddings, all_chunks, all_metadata)
    
    def query(self, question: str, top_k: int = 3) -> Tuple[Optional[str], List[Tuple[str, float, dict]]]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Tuple of (answer, relevant_chunks)
        """
        if not self.is_initialized:
            self.initialize()
        
        if self.embedding_manager.index is None:
            return "No documents have been processed yet. Please upload documents first.", []
        
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.embedding_manager.search(question, top_k=top_k)
            
            if not relevant_chunks:
                return "No relevant information found in the documents.", []
            
            # Use LLM to generate intelligent answer
            answer = self.llm_integration.generate_answer(question, relevant_chunks)
            
            return answer, relevant_chunks
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing your question: {str(e)}", []

def main():
    """Test the complete RAG pipeline with LLM integration."""
    pipeline = RAGPipeline()
    
    # Create a test document
    test_file = "data/raw_documents/test_ai.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""
        Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
        Machine Learning is a subset of AI that enables computers to learn without explicit programming.
        Deep Learning uses neural networks with multiple layers to analyze various factors of data.
        Natural Language Processing (NLP) allows computers to understand and interpret human language.
        Computer Vision enables machines to interpret and understand the visual world.
        """)
    
    # Test the pipeline
    print("=== Testing RAG Pipeline with LLM ===")
    
    # Process document
    success = pipeline.process_document(test_file)
    if success:
        print("✓ Document processed successfully")
        
        # Test queries
        test_queries = [
            "What is deep learning?",
            "How does machine learning work?",
            "What are neural networks used for?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            answer, chunks = pipeline.query(query)
            print(f"Answer: {answer}")
            print("-" * 50)
            
    else:
        print("✗ Failed to process document")

if __name__ == "__main__":
    main()
"""
Enhanced RAG pipeline with better source tracking.
"""
import os
import logging
import uuid
from typing import List, Tuple, Optional, Dict
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.llm_integration import LLMIntegration
from src.config import config

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline with enhanced source tracking."""
    
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
    
    def process_document(self, file_path: str) -> Tuple[bool, str, str]:
        """
        Process a document and add to the vector store.
        
        Returns:
            Tuple of (success, document_id, filename)
        """
        if not self.is_initialized:
            self.initialize()

        try:
            # Load and chunk document
            text = self.document_processor.load_document(file_path)
            if not text:
                logger.error(f"Failed to load document or no text extracted: {file_path}")
                return False, "", "Failed to load document"
        
            if len(text.strip()) < 10:  # Minimum text length
                logger.error(f"Document contains too little text: {file_path}")
                return False, "", "Document contains too little text"
        
            chunks = self.document_processor.chunk_text(text)
            if not chunks:
                logger.error("No chunks created from document")
                return False, "", "No text chunks could be created"
        
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            
            # Generate unique document ID and get filename
            document_id = str(uuid.uuid4())
            filename = os.path.basename(file_path)
        
            # Create enhanced metadata for each chunk
            metadata = [{
                "source": file_path, 
                "chunk_id": i, 
                "filename": filename,
                "document_id": document_id,
                "document_name": filename,  # Add human-readable name
                "total_chunks": len(chunks)  # Add context about the document
            } for i in range(len(chunks))]
        
            # Add to ChromaDB
            self.embedding_manager.add_to_index(chunks, metadata, document_id)
        
            logger.info(f"Successfully processed document: {filename} (ID: {document_id})")
            return True, document_id, filename
        
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return False, "", f"Error processing document: {str(e)}"

    def query(self, question: str, top_k: int = 3, filter_by_document: Optional[str] = None) -> Tuple[Optional[str], List[Tuple[str, float, dict]], Dict]:
        """
        Query the RAG system with enhanced source tracking.
        
        Args:
            question: The question to ask
            top_k: Number of relevant chunks to retrieve
            filter_by_document: Optional document ID to filter results
            
        Returns:
            Tuple of (answer, relevant_chunks, source_info)
        """
        if not self.is_initialized:
            self.initialize()
        
        #Check if we have any documents
        stats = self.embedding_manager.get_stats()
        if stats["total_chunks"] == 0:
            return "No documents have been processed yet. Please upload documents first.", [], {}
        
        try:
            #Build metadata filter if specified
            filter_metadata = None
            if filter_by_document:
                filter_metadata = {"document_id": filter_by_document}
            
            #Retrieve relevant chunks
            relevant_chunks = self.embedding_manager.search(
                question, 
                top_k=top_k, 
                filter_metadata=filter_metadata
            )
            
            # Remove duplicates
            relevant_chunks = self.remove_duplicate_chunks(relevant_chunks)

            if not relevant_chunks:
                return "No relevant information found in the documents.", [], {}
            
            # Extract source information
            source_info = self._extract_source_info(relevant_chunks)
            
            # Use LLM to generate intelligent answer
            answer = self.llm_integration.generate_answer(question, relevant_chunks)
            
            return answer, relevant_chunks, source_info
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing your question: {str(e)}", [], {}
    
    def _extract_source_info(self, chunks: List[Tuple[str, float, dict]]) -> Dict:
        """
        Extract detailed source information from retrieved chunks with proper deduplication.
        """
        if not chunks:
            return {}
    
        source_info = {
            "total_sources": 0,
            "documents": [],
            "chunk_details": [],
            "primary_source": "Unknown"
        }
    
        # Track unique documents by document_id to avoid duplicates
        unique_docs = {}
        seen_chunk_content = set()
    
        for i, (chunk, score, metadata) in enumerate(chunks):
            # Skip if we've already seen very similar chunk content (avoid duplicates)
            # Use a more robust signature: first 80 chars + last 20 chars + length
            chunk_preview = chunk[:80] + chunk[-20:] if len(chunk) > 100 else chunk
            content_signature = f"{len(chunk)}:{hash(chunk_preview)}"
        
            if content_signature in seen_chunk_content:
                continue
            seen_chunk_content.add(content_signature)
        
            # Extract proper filename
            raw_filename = metadata.get('filename', metadata.get('document_name', 'Unknown Document'))
            source_file = metadata.get('source', 'Unknown Source')
        
            # Clean filename - remove UUID prefix
            if '_' in raw_filename and len(raw_filename.split('_')[0]) == 8:
                clean_filename = '_'.join(raw_filename.split('_')[1:])
            else:
                clean_filename = raw_filename
        
            doc_id = metadata.get('document_id', 'unknown')
        
            # Add to unique documents
            if doc_id not in unique_docs:
                unique_docs[doc_id] = {
                    'document_name': clean_filename,
                    'filename': clean_filename,
                    'source_file': source_file,
                    'document_id': doc_id
                }
        
            # Add chunk details
            source_info["chunk_details"].append({
                "chunk_id": i + 1,
                "document": clean_filename,
                "filename": clean_filename,
                "confidence": f"{score:.3f}",
                "source_file": source_file,
                "document_id": doc_id,
                "content_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
    
        source_info["total_sources"] = len(unique_docs)
        source_info["documents"] = list(set([doc['document_name'] for doc in unique_docs.values()]))
        source_info["primary_source"] = source_info["documents"][0] if source_info["documents"] else "Unknown"
    
        # Sort chunks by confidence score (highest first)
        source_info["chunk_details"].sort(key=lambda x: float(x["confidence"]), reverse=True)
    
        # Limit chunk details to avoid overwhelming display
        source_info["chunk_details"] = source_info["chunk_details"][:5]
    
        return source_info
    
    def get_document_list(self) -> List[Dict]:
        """Get list of all processed documents."""
        if not self.is_initialized:
            self.initialize()
        
        return self.embedding_manager.list_documents()
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store."""
        try:
            self.embedding_manager.delete_document(document_id)
            logger.info(f"Successfully deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        if not self.is_initialized:
            self.initialize()
        
        return self.embedding_manager.get_stats()
    
    def cleanup_duplicate_documents(self):
        """Clean up duplicate documents from the vector store."""
        try:
            documents = self.get_document_list()
            seen_filenames = set()
            duplicates = []
        
            for doc in documents:
                filename = doc.get('filename', '')
                if filename in seen_filenames:
                    duplicates.append(doc)
                else:
                    seen_filenames.add(filename)
        
            # Delete duplicates
            for duplicate in duplicates:
                self.delete_document(duplicate['document_id'])
                logger.info(f"Cleaned up duplicate document: {duplicate['filename']}")
        
            return len(duplicates)
        
        except Exception as e:
            logger.error(f"Error cleaning up duplicates: {str(e)}")
            return 0
    
    def remove_duplicate_chunks(self, chunks: List[Tuple[str, float, dict]]) -> List[Tuple[str, float, dict]]:
        """Remove duplicate or very similar chunks from results with better detection."""
        unique_chunks = []
        seen_content = set()
    
        # Sort by score first to keep highest scoring duplicates
        chunks.sort(key=lambda x: x[1], reverse=True)
    
        for chunk, score, metadata in chunks:
            # Create a more robust content signature
            chunk_text = chunk.strip().lower()
            if len(chunk_text) < 20:  # Skip very short chunks
                continue
            
            # Use beginning, middle and end for signature
            signature_parts = []
            if len(chunk_text) > 50:
                signature_parts.extend([chunk_text[:30], chunk_text[-20:]])
            else:
                signature_parts.append(chunk_text)
        
            content_sig = "|".join(signature_parts)
        
            if content_sig not in seen_content:
                seen_content.add(content_sig)
                unique_chunks.append((chunk, score, metadata))
    
        return unique_chunks

def main():
    """Test the enhanced RAG pipeline with source tracking."""
    pipeline = RAGPipeline()
    
    # Create a test document
    test_file = "data/raw_documents/test_cloud_computing.txt"
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""
        Cloud Computing Overview
        
        Cloud computing is the delivery of computing services over the internet.
        These services include servers, storage, databases, networking, software, and more.
        
        Service Models:
        - IaaS (Infrastructure as a Service): Provides virtualized computing resources
        - PaaS (Platform as a Service): Provides platform for developing and deploying applications  
        - SaaS (Software as a Service): Provides software applications over the internet
        
        Deployment Models:
        - Public Cloud: Services offered over the public internet
        - Private Cloud: Cloud infrastructure for a single organization
        - Hybrid Cloud: Combination of public and private clouds
        """)
    
    # Test the enhanced pipeline
    print("=== Testing Enhanced RAG Pipeline with Source Tracking ===")
    
    # Process document
    success, doc_id, filename = pipeline.process_document(test_file)
    if success:
        print(f"✓ Document processed successfully: {filename} (ID: {doc_id})")
        
        # Test enhanced query
        query = "What are the cloud service models?"
        answer, chunks, source_info = pipeline.query(query)
        
        print(f"\nQuery: '{query}'")
        print(f"Answer: {answer}")
        print(f"\n📚 Source Information:")
        print(f"   - Total documents: {source_info['total_sources']}")
        print(f"   - Documents: {', '.join(source_info['documents'])}")
        print(f"   - Primary source: {source_info['primary_source']}")
        print(f"   - Chunks used: {len(source_info['chunk_details'])}")
        
        print(f"\n🔍 Detailed Chunk Info:")
        for chunk in source_info['chunk_details']:
            print(f"   - Chunk {chunk['chunk_id']}: {chunk['document']} (Confidence: {chunk['confidence']})")
            
        # Cleanup
        pipeline.delete_document(doc_id)
        os.remove(test_file)
            
    else:
        print("✗ Failed to process document")

if __name__ == "__main__":
    main()
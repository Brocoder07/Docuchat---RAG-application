"""
Clean up duplicate documents from the vector store.
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.rag_pipeline import RAGPipeline

def cleanup_duplicates():
    """Clean up duplicate documents."""
    print("🧹 Cleaning up duplicate documents...")
    
    pipeline = RAGPipeline()
    pipeline.initialize()
    
    # Get current documents
    documents = pipeline.get_document_list()
    print(f"Current documents: {len(documents)}")
    
    for doc in documents:
        print(f" - {doc.get('filename')} (ID: {doc.get('document_id')})")
    
    # Clean duplicates
    removed_count = pipeline.cleanup_duplicate_documents()
    
    # Get updated documents
    documents_after = pipeline.get_document_list()
    print(f"\n✅ Removed {removed_count} duplicates")
    print(f"Remaining documents: {len(documents_after)}")
    
    for doc in documents_after:
        print(f" - {doc.get('filename')}")

if __name__ == "__main__":
    cleanup_duplicates()
"""
Test the full ChromaDB migration with RAG pipeline.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_full_migration():
    """Test the complete RAG pipeline with ChromaDB."""
    print("🧪 Testing Full ChromaDB Migration...")
    
    try:
        from src.rag_pipeline import RAGPipeline
        
        # Initialize RAG pipeline
        pipeline = RAGPipeline()
        pipeline.initialize()
        
        print("✅ RAG Pipeline initialized with ChromaDB")
        
        # Test document processing
        test_file = "data/raw_documents/test_migration.txt"
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("""
            Cloud Computing Migration Test
            
            This document tests the ChromaDB migration.
            Cloud computing provides scalable resources.
            Service models include IaaS, PaaS, and SaaS.
            """)
        
        # Process document
        success, doc_id = pipeline.process_document(test_file)
        if not success:
            print("❌ Failed to process document")
            return False
            
        print(f"✅ Document processed successfully (ID: {doc_id})")
        
        # Test query
        answer, chunks = pipeline.query("What are the cloud service models?")
        print(f"✅ Query successful: {len(chunks)} chunks found")
        print(f"Answer: {answer}")
        
        # Test document listing
        documents = pipeline.get_document_list()
        print(f"✅ Document listing: {len(documents)} documents")
        
        # Test stats
        stats = pipeline.get_stats()
        print(f"✅ System stats: {stats}")
        
        # Cleanup
        pipeline.delete_document(doc_id)
        os.remove(test_file)
        
        print("🎉 Full ChromaDB migration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Full migration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_migration()
    sys.exit(0 if success else 1)
"""
Test script for ChromaDB migration.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_chroma_migration():
    """Test that ChromaDB migration works correctly."""
    print("🧪 Testing ChromaDB Migration...")
    
    try:
        # Import after path is set
        from src.chroma_manager import ChromaManager
        
        # Initialize ChromaDB
        manager = ChromaManager()
        manager.initialize()
        
        # Test basic operations
        test_chunks = ["Test document for migration validation."]
        test_metadata = [{"source": "migration_test.pdf", "chunk_id": 0, "filename": "migration_test.pdf"}]
        
        manager.add_documents(test_chunks, test_metadata, "migration_test_001")
        
        # Test search
        results = manager.search("test document")
        
        # Test stats
        stats = manager.get_collection_stats()
        
        print("✅ ChromaDB migration successful!")
        print(f"   - Collection: {stats['collection_name']}")
        print(f"   - Total chunks: {stats['total_chunks']}")
        print(f"   - Search results: {len(results)}")
        
        # Cleanup
        manager.delete_document("migration_test_001")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("💡 Make sure ChromaDB is installed: pip install chromadb==1.2.2")
        return False
    except Exception as e:
        print(f"❌ ChromaDB migration failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_chroma_migration()
    sys.exit(0 if success else 1)
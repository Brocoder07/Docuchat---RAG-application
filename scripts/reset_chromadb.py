"""
ChromaDB Reset Script - Completely clears vector store and document tracking.
Senior Engineer Principle: Safe, comprehensive reset with backups.
"""
import os
import shutil
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging for the reset script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('reset_chromadb.log', mode='w', encoding='utf-8')
        ]
    )

def reset_chromadb():
    """Completely reset ChromaDB vector store and related data."""
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 Starting ChromaDB Reset Process")
    logger.info("=" * 50)
    
    # Define paths to reset
    paths_to_reset = [
        "data/vector_store",  # ChromaDB data
        "data/uploads",       # Uploaded files
        "docuchat.log",       # Application logs
        "test_results.log",   # Test logs
        "reset_chromadb.log"  # This script's logs
    ]
    
    try:
        # Stop any running services first
        logger.info("⏹️  Please ensure all DocuChat services are stopped (Ctrl+C in terminals)")
        
        # Reset ChromaDB vector store
        vector_store_path = project_root / "data" / "vector_store"
        if vector_store_path.exists():
            logger.info(f"🗑️  Deleting vector store: {vector_store_path}")
            shutil.rmtree(vector_store_path)
            logger.info("✅ Vector store deleted successfully")
        else:
            logger.info("ℹ️  Vector store directory not found (already clean)")
        
        # Clear uploads directory (keep the directory structure)
        uploads_path = project_root / "data" / "uploads"
        if uploads_path.exists():
            logger.info(f"🗑️  Clearing uploads directory: {uploads_path}")
            for item in uploads_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.info("✅ Uploads directory cleared successfully")
        else:
            uploads_path.mkdir(parents=True, exist_ok=True)
            logger.info("✅ Created uploads directory structure")
        
        # Clear log files
        log_files = [
            project_root / "docuchat.log",
            project_root / "test_results.log", 
            project_root / "reset_chromadb.log"
        ]
        
        for log_file in log_files:
            if log_file.exists():
                logger.info(f"🗑️  Deleting log file: {log_file}")
                log_file.unlink()
        
        # Reset session state by clearing any session files
        session_files_path = project_root / "frontend" / ".streamlit"
        if session_files_path.exists():
            logger.info(f"🗑️  Clearing Streamlit session cache")
            # Streamlit stores session state in memory, but clear any cached files
            for pattern in ["*.pyc", "__pycache__"]:
                for item in session_files_path.rglob(pattern):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
        
        logger.info("✅ ChromaDB reset completed successfully!")
        
        # Create fresh directories
        (project_root / "data" / "vector_store").mkdir(parents=True, exist_ok=True)
        (project_root / "data" / "uploads").mkdir(parents=True, exist_ok=True)
        (project_root / "logs").mkdir(exist_ok=True)
        
        logger.info("📁 Fresh directory structure created")
        
        # Display next steps
        logger.info("\n" + "=" * 50)
        logger.info("🎯 NEXT STEPS:")
        logger.info("1. Start the backend: python -m api.main")
        logger.info("2. Start the frontend: streamlit run frontend/app.py")
        logger.info("3. Upload your documents again")
        logger.info("4. Test with simple questions first")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Reset failed: {str(e)}")
        return False

def soft_reset():
    """
    Soft reset - only clears the collection without deleting files.
    Useful for development without losing upload directory structure.
    """
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 Starting Soft Reset (Collection Only)")
    
    try:
        # Import ChromaDB and reset just the collection
        import chromadb
        from core.config import config
        
        # Initialize client and delete collection
        client = chromadb.PersistentClient(path="data/vector_store")
        
        try:
            client.delete_collection("docuchat_documents")
            logger.info("✅ Collection 'docuchat_documents' deleted")
        except Exception as e:
            logger.warning(f"Collection may not exist: {e}")
        
        # Recreate collection
        client.create_collection(
            name="docuchat_documents",
            metadata={"description": "DocuChat document chunks"}
        )
        logger.info("✅ New collection 'docuchat_documents' created")
        
        # Clear the RAG pipeline's processed documents list
        try:
            from core.rag_pipeline import rag_pipeline
            rag_pipeline.processed_documents = []
            rag_pipeline.query_history = []
            logger.info("✅ RAG pipeline memory cleared")
        except Exception as e:
            logger.warning(f"Could not clear RAG pipeline: {e}")
        
        logger.info("✅ Soft reset completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Soft reset failed: {str(e)}")
        return False

def main():
    """Main function with command line options."""
    import argparse
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Reset ChromaDB Vector Store")
    parser.add_argument(
        "--soft", 
        action="store_true",
        help="Soft reset (keep data structure, only clear collection)"
    )
    parser.add_argument(
        "--hard",
        action="store_true", 
        help="Hard reset (delete everything including uploads and logs)"
    )
    
    args = parser.parse_args()
    
    print("🚀 DocuChat ChromaDB Reset Tool")
    print("=" * 50)
    
    if args.soft:
        print("🔄 Performing SOFT reset (collection only)...")
        success = soft_reset()
    elif args.hard:
        print("💀 Performing HARD reset (complete wipe)...")
        success = reset_chromadb()
    else:
        # Interactive mode
        print("Choose reset type:")
        print("1. SOFT reset - Clear only ChromaDB collection (recommended)")
        print("2. HARD reset - Delete everything including uploads and logs")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            success = soft_reset()
        elif choice == "2":
            print("⚠️  WARNING: This will delete ALL uploaded documents and logs!")
            confirm = input("Type 'YES' to confirm: ").strip()
            if confirm == "YES":
                success = reset_chromadb()
            else:
                print("❌ Reset cancelled")
                return
        else:
            print("❌ Invalid choice")
            return
    
    if success:
        print("✅ Reset completed successfully!")
        print("\n🎯 Next: Restart your backend and frontend services")
    else:
        print("❌ Reset failed - check reset_chromadb.log for details")

if __name__ == "__main__":
    main()
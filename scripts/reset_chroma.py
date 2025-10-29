"""
Script to reset ChromaDB database for schema compatibility.
"""
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_chroma_db():
    """Reset ChromaDB database to fix schema issues."""
    
    # Path to ChromaDB data
    vector_store_path = "data/vector_store"
    
    if os.path.exists(vector_store_path):
        try:
            # Remove the entire vector store directory
            shutil.rmtree(vector_store_path)
            logger.info(f"✅ Successfully removed ChromaDB database: {vector_store_path}")
            
            # Recreate the directory
            os.makedirs(vector_store_path, exist_ok=True)
            logger.info("✅ Recreated empty vector store directory")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error resetting ChromaDB: {str(e)}")
            return False
    else:
        logger.info("ℹ️  ChromaDB database doesn't exist yet, no need to reset")
        return True

if __name__ == "__main__":
    print("🔄 Resetting ChromaDB database to fix schema compatibility...")
    if reset_chroma_db():
        print("✅ ChromaDB reset successful! You can now restart the server.")
    else:
        print("❌ ChromaDB reset failed!")
# cleanup_vector_store.py
import os
import shutil

def clear_vector_store():
    """Clear the vector store to remove duplicates."""
    vector_store_path = "data/vector_store"
    raw_docs_path = "data/raw_documents"
    
    if os.path.exists(vector_store_path):
        shutil.rmtree(vector_store_path)
        print(f"✅ Cleared vector store: {vector_store_path}")
        os.makedirs(vector_store_path)
    
    if os.path.exists(raw_docs_path):
        for file in os.listdir(raw_docs_path):
            file_path = os.path.join(raw_docs_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"✅ Cleared raw documents: {raw_docs_path}")
    
    print("🎉 Vector store cleared! You can now upload fresh documents.")

if __name__ == "__main__":
    clear_vector_store()
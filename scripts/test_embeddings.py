"""
Test script to verify embeddings are working.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from src.config import config

def test_embeddings():
    print("🧪 Testing embedding generation...")
    
    # Test the embedding model
    model = SentenceTransformer(config.embedding.model_name)
    
    test_texts = [
        "This is a test about projects and technology",
        "Machine learning and artificial intelligence",
        "Android development with Kotlin and Jetpack Compose"
    ]
    
    embeddings = model.encode(test_texts)
    print(f"✅ Embeddings generated successfully")
    print(f"   Input texts: {len(test_texts)}")
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Sample embedding: {embeddings[0][:5]}...")
    
    # Test similarity
    query = "projects"
    query_embedding = model.encode([query])
    similarities = model.similarity(query_embedding, embeddings)[0]
    
    print(f"✅ Similarity test:")
    for i, (text, sim) in enumerate(zip(test_texts, similarities)):
        print(f"   '{text}' -> similarity: {sim:.3f}")

if __name__ == "__main__":
    test_embeddings()
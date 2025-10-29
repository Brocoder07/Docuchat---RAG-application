"""
Debug routes for troubleshooting RAG issues.
"""
from fastapi import APIRouter
from src.api.services.rag_service import rag_service

router = APIRouter(prefix="/debug", tags=["debug"])

@router.get("/documents")
async def debug_documents():
    """Debug endpoint to see all documents and their chunks."""
    try:
        documents = rag_service.rag_pipeline.get_document_list()
        stats = rag_service.rag_pipeline.get_stats()
        
        # Get sample chunks from each document
        document_details = []
        for doc in documents:
            # Try to search for any content from this document
            sample_query = "technology"  # Generic term that should match something
            results = rag_service.rag_pipeline.embedding_manager.search(
                sample_query, 
                top_k=2, 
                filter_metadata={"document_id": doc['document_id']}
            )
            
            document_details.append({
                "document": doc,
                "sample_chunks": [chunk[0][:100] + "..." for chunk in results[:2]] if results else [],
                "chunks_found": len(results)
            })
        
        return {
            "total_documents": len(documents),
            "total_chunks": stats.get("total_chunks", 0),
            "documents": document_details,
            "collection_stats": stats
        }
        
    except Exception as e:
        return {"error": str(e)}

@router.get("/search-test")
async def search_test(query: str = "projects"):
    """Test search with different queries."""
    try:
        results = rag_service.rag_pipeline.embedding_manager.search(query, top_k=5)
        
        return {
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "chunk_preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    "score": score,
                    "metadata": metadata
                }
                for chunk, score, metadata in results
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}

@router.get("/collection-contents")
async def debug_collection_contents(limit: int = 5):
    """Debug endpoint to see what's actually stored in ChromaDB."""
    try:
        contents = rag_service.rag_pipeline.embedding_manager.chroma_manager.debug_collection_contents(limit)
        return contents
    except Exception as e:
        return {"error": str(e)}
    
# Add to debug.py
@router.post("/reset-collection")
async def reset_collection():
    """Reset the entire ChromaDB collection."""
    try:
        rag_service.rag_pipeline.embedding_manager.chroma_manager.reset_collection()
        return {"message": "Collection reset successfully"}
    except Exception as e:
        return {"error": str(e)}
    
@router.get("/chunks-by-document")
async def debug_chunks_by_document(document_id: str = None):
    """Debug endpoint to see all chunks for a specific document."""
    try:
        # Get all chunks from ChromaDB
        all_chunks = rag_service.rag_pipeline.chroma_manager.get_all_chunks()
        
        if document_id:
            # Filter by document
            filtered_chunks = [
                chunk for chunk in all_chunks 
                if chunk.get('metadata', {}).get('document_id') == document_id
            ]
        else:
            filtered_chunks = all_chunks
        
        # Format for display
        chunk_details = []
        for chunk in filtered_chunks:
            metadata = chunk.get('metadata', {})
            chunk_details.append({
                'document_id': metadata.get('document_id'),
                'source': metadata.get('source'),
                'chunk_index': metadata.get('chunk_index'),
                'content_preview': chunk.get('text', '')[:100] + "...",
                'full_content': chunk.get('text', '')[:500]
            })
        
        return {
            "total_chunks": len(filtered_chunks),
            "chunks": chunk_details
        }
        
    except Exception as e:
        return {"error": str(e)}
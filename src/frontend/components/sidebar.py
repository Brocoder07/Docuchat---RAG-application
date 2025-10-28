"""
Enhanced sidebar component with better document tracking.
"""
import streamlit as st
from src.frontend.services.api_client import api_client
from src.frontend.config.settings import config
from src.frontend.utils.session_state import update_documents_count

def render_sidebar():
    """Render the sidebar with document upload and system info."""
    with st.sidebar:
        st.header("📁 Document Management")
        render_file_upload()
        st.divider()
        render_system_info()
        st.divider()
        render_document_list()

def render_file_upload():
    """Render the file upload section."""
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=config.ALLOWED_FILE_TYPES,
        help=f"Supported formats: {', '.join(config.ALLOWED_FILE_TYPES).upper()}"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            result = api_client.upload_document(
                uploaded_file.getvalue(), 
                uploaded_file.name
            )
            
            if result["success"]:
                st.success(f"✅ {result['data']['message']}")
                # Update documents count
                health_info = api_client.get_health_info()
                if health_info:
                    update_documents_count(health_info["documents_loaded"])
            else:
                st.error(f"❌ {result['error']}")

def render_system_info():
    """Render system information section."""
    st.header("📊 System Info")
    
    health_info = api_client.get_health_info()
    
    if health_info:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents Loaded", health_info["documents_loaded"])
        with col2:
            model_name = health_info.get("model", "ollama-llama3.2:1b")
            st.metric("AI Model", f"{model_name} ✓")
        
        # Show collection info if available
        if health_info.get("collection_name"):
            st.caption(f"Collection: {health_info['collection_name']}")
        
        st.metric("Cost", "$0.00")
        status = health_info.get("status", "unknown")
        if status == "healthy":
            st.metric("Status", "🟢 Healthy")
        elif status == "error":
            st.metric("Status", "🔴 Error")
        else:
            st.metric("Status", "🟡 Unknown")
    else:
        st.warning("Unable to fetch system information")

def render_document_list():
    """Render the list of processed documents."""
    st.header("📄 Processed Documents")
    
    # Add a refresh button
    if st.button("🔄 Refresh Document List", use_container_width=True):
        st.rerun()
    
    # Get document list from API
    result = api_client.list_documents()
    
    if result["success"] and result["data"]["documents"]:
        documents = result["data"]["documents"]
        for doc in documents:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{doc['filename']}**")
                    st.caption(f"Chunks: {doc['chunks']}")
                with col2:
                    st.caption(f"ID: {doc['document_id'][:8]}...")
        st.caption(f"Total: {len(documents)} documents")
    else:
        st.info("No documents processed yet. Upload a document to get started!")
    
    # Clear chat history button
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()
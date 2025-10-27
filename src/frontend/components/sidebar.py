"""
Sidebar component for document management and system info.
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
                st.success(result["data"]["message"])
                # Update documents count
                health_info = api_client.get_health_info()
                if health_info:
                    update_documents_count(health_info["documents_loaded"])
            else:
                st.error(result["error"])

def render_system_info():
    """Render system information section."""
    st.header("📊 System Info")
    
    health_info = api_client.get_health_info()
    
    if health_info:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents Loaded", health_info["documents_loaded"])
        with col2:
            # Show actual model name instead of "Rule-Based"
            model_name = health_info.get("model", "ollama-llama3.2:1b")
            st.metric("AI Model", f"{model_name} ✓")
        
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
    
    # Clear chat history button
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
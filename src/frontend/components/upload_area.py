"""
File upload component.
"""
import streamlit as st
from src.frontend.services.api_client import api_client
from src.frontend.config.settings import config

def render_upload_area():
    """Render the file upload area."""
    uploaded_file = st.file_uploader(
        "📤 Upload Document",
        type=config.ALLOWED_FILE_TYPES,
        help=f"Supported formats: {', '.join(config.ALLOWED_FILE_TYPES).upper()}",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        return _handle_file_upload(uploaded_file)
    
    return None

def _handle_file_upload(uploaded_file):
    """Handle file upload and processing."""
    with st.spinner(f"Processing {uploaded_file.name}..."):
        result = api_client.upload_document(
            uploaded_file.getvalue(), 
            uploaded_file.name
        )
        
        if result["success"]:
            st.success("✅ Document processed successfully!")
            return True
        else:
            st.error(f"❌ {result['error']}")
            return False
    
    return False
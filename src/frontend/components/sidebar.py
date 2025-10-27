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
        _render_file_upload()
        st.divider()
        _render_system_info()

def _render_file_upload():
    """Render the file upload section with proper state management."""
    # Initialize session state for tracking uploads
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=config.ALLOWED_FILE_TYPES,
        help=f"Supported formats: {', '.join(config.ALLOWED_FILE_TYPES).upper()}",
        key="document_uploader"
    )
    
    # Process new uploads only
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Check if this exact file was already processed in this session
        if file_id not in st.session_state.uploaded_files:
            _process_uploaded_file(uploaded_file, file_id)
        else:
            st.info("📄 This file has already been processed in this session.")

def _process_uploaded_file(uploaded_file, file_id):
    """Process the uploaded file and track it in session state."""
    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            result = api_client.upload_document(
                uploaded_file.getvalue(), 
                uploaded_file.name
            )
            
            # Remove debug output for cleaner UI
            # st.write("Debug - API Response:", result)
            
            if result.get("success", False):
                # Mark file as processed in this session
                st.session_state.uploaded_files[file_id] = {
                    "name": uploaded_file.name,
                    "timestamp": st.session_state.get('script_run_count', 0)
                }
                
                # Get the success message from response data
                response_data = result.get("data", {})
                success_message = response_data.get("message", "Document processed successfully")
                
                st.success(f"✅ {success_message}")
                
                # Update documents count
                health_info = api_client.get_health_info()
                if health_info and isinstance(health_info, dict):
                    update_documents_count(health_info.get("documents_loaded", 0))
                    
                # Small delay to show success message
                st.rerun()
                
            else:
                error_msg = result.get('error', 'Unknown error occurred')
                if "already been processed" in str(error_msg):
                    st.warning("⚠️ This file was already processed in a previous session.")
                    # Still mark it as processed to prevent repeated attempts
                    st.session_state.uploaded_files[file_id] = {
                        "name": uploaded_file.name,
                        "timestamp": st.session_state.get('script_run_count', 0)
                    }
                else:
                    st.error(f"❌ {error_msg}")
                    
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
            # Show more detailed error for debugging
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")

def _render_system_info():
    """Render system information section."""
    st.header("📊 System Info")
    
    health_info = api_client.get_health_info()
    
    if health_info and isinstance(health_info, dict):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents Loaded", health_info.get("documents_loaded", 0))
        with col2:
            st.metric("AI Model", "Rule-Based ✓")
        
        st.metric("Cost", "$0.00")
        st.metric("Status", "🟢 Healthy")
        
        # Show session info
        st.divider()
        st.caption(f"Files in this session: {len(st.session_state.uploaded_files)}")
    else:
        st.warning("Unable to fetch system information")
    
    # Clear chat history button
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
    
    with col2:
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.uploaded_files = {}
            st.session_state.chat_history = []
            st.success("New session started! You can re-upload files.")
            st.rerun()
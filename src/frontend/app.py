"""
Main Streamlit application - now clean and modular!
"""
import streamlit as st

from src.frontend.config.settings import config
from src.frontend.utils.session_state import init_session_state, set_api_health_status
from src.frontend.services.api_client import api_client
from src.frontend.components.sidebar import render_sidebar
from src.frontend.components.chat_area import render_chat_area

def main():
    """Main Streamlit application."""
    # Configure the page
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout=config.LAYOUT,
        initial_sidebar_state=config.INITIAL_SIDEBAR_STATE
    )
    
    # Initialize session state
    init_session_state()
    
    # Initialize uploaded files tracking if not exists
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    
    # Check API health
    api_healthy = api_client.check_health()
    set_api_health_status(api_healthy)
    
    # Render header
    _render_header()
    
    # Show error if API is not available
    if not api_healthy:
        _render_api_error()
        return
    
    # Render sidebar (document upload, system info)
    render_sidebar()
    
    # Render main chat area
    render_chat_area()
    
    # Handle pending quick questions
    _handle_pending_questions()

def _handle_pending_questions():
    """Handle quick questions that were clicked."""
    if 'pending_question' in st.session_state:
        question = st.session_state.pending_question
        # This will trigger the chat input in the next render
        st.session_state.auto_question = question

def _render_header():
    """Render the page header."""
    st.title("📚 DocuChat - FREE Document Q&A")
    st.markdown("**Ask questions about your documents - Zero costs, 100% private!**")
    st.divider()

def _render_api_error():
    """Render API error message."""
    st.error("🚨 Backend API is not running!")
    
    st.info("**To start the backend API:**")
    st.code("python -m src.api.main")
    
    st.info("**The API will be available at:**")
    st.code("http://localhost:8000")
    
    st.info("**Once the API is running, refresh this page.**")

if __name__ == "__main__":
    main()
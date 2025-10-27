"""
Main Streamlit application - Fixed imports
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

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
    
    # Check API health
    api_healthy = api_client.check_health()
    set_api_health_status(api_healthy)
    
    # Render header
    _render_header()
    
    # Show error if API is not available
    if not api_healthy:
        _render_api_error()
        return
    
    # Render main interface - sidebar first, then chat area
    render_sidebar()
    render_chat_area()  # This now handles quick questions in sidebar

def _render_header():
    """Render the page header."""
    st.title("📚 DocuChat - AI Document Q&A")
    st.markdown("**Ask questions about your documents - Powered by Ollama LLM**")
    st.divider()

def _render_api_error():
    """Render API error message."""
    st.error("🚨 Backend API is not running!")
    
    st.info("**To start the backend API:**")
    st.code("python -m src.api")
    
    st.info("**The API will be available at:**")
    st.code("http://localhost:8000")
    
    st.info("**Once the API is running, refresh this page.**")

if __name__ == "__main__":
    main()
"""
Session state management utilities.
"""
import streamlit as st

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'documents_loaded': 0,
        'chat_history': [],
        'api_healthy': False,
        'uploaded_files': {},
        'script_run_count': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Increment script run count to track reruns
    st.session_state.script_run_count = st.session_state.get('script_run_count', 0) + 1

def clear_chat_history():
    """Clear chat history from session state."""
    st.session_state.chat_history = []

def add_to_chat_history(question: str, answer: str, confidence: str = None):
    """Add a message to chat history."""
    st.session_state.chat_history.append((question, answer, confidence))

def get_chat_history():
    """Get the current chat history."""
    return st.session_state.chat_history

def update_documents_count(count: int):
    """Update the documents loaded count."""
    if isinstance(count, int):
        st.session_state.documents_loaded = count
    else:
        # If count is not an integer, try to convert it
        try:
            st.session_state.documents_loaded = int(count)
        except (ValueError, TypeError):
            st.session_state.documents_loaded = 0

def set_api_health_status(healthy: bool):
    """Set API health status."""
    st.session_state.api_healthy = healthy
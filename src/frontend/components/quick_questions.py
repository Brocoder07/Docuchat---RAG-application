"""
Quick questions component.
"""
import streamlit as st
from src.frontend.config.settings import config

def render_quick_questions():
    """Render the quick questions panel."""
    st.header("🎯 Quick Questions")
    
    for question in config.QUICK_QUESTIONS:
        if st.button(question, key=f"quick_{question}", use_container_width=True):
            # Set the question in session state to be picked up by chat input
            st.session_state.pending_question = question
            st.rerun()

def get_pending_question() -> str:
    """Get and clear any pending quick question."""
    return st.session_state.pop('pending_question', None)
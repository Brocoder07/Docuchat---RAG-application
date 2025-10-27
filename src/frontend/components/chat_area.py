"""
Main chat area component.
"""
import streamlit as st
from src.frontend.services.api_client import api_client
from src.frontend.utils.session_state import add_to_chat_history, get_chat_history

def render_chat_area():
    """Render the main chat area."""
    # Render quick questions in sidebar instead
    _render_quick_questions_sidebar()
    
    # Main chat interface - NO COLUMNS around chat_input
    _render_chat_interface()

def _render_chat_interface():
    """Render the chat interface."""
    st.header("💬 Chat with Your Documents")
    
    # Display chat history
    _render_chat_history()
    
    # Query input - MUST be at main level, not inside columns/sidebar
    _render_chat_input()

def _render_chat_history():
    """Render the chat history."""
    chat_history = get_chat_history()
    
    for i, (question, answer, confidence) in enumerate(chat_history):
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            st.write(answer)
            if confidence:
                st.caption(f"Confidence: {confidence}")
            st.divider()

def _render_chat_input():
    """Render the chat input and handle queries."""
    # Chat input must be at the main level, not inside any layout containers
    question = st.chat_input("Ask a question about your documents...")
    
    if question:
        # Add user question to chat immediately
        with st.chat_message("user"):
            st.write(question)
        
        # Get and display answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                result = api_client.query_documents(question)
            
            if result["success"]:
                data = result["data"]
                st.write(data["answer"])
                
                # Show confidence and sources
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption(f"Confidence: {data['confidence']}")
                with col_b:
                    st.caption(f"Sources: {data['relevant_sources']}")
                
                # Add to chat history
                add_to_chat_history(question, data["answer"], data["confidence"])
            else:
                st.error(f"Error: {result['error']}")

def _render_quick_questions_sidebar():
    """Render quick questions in the sidebar."""
    with st.sidebar:
        st.header("🎯 Quick Questions")
        
        # Define some default quick questions
        quick_questions = [
            "What is this document about?",
            "Can you summarize the main points?",
            "What are the key topics discussed?",
            "Are there any important definitions?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}", use_container_width=True):
                # Set the question to trigger chat input
                st.session_state.pending_question = question
                st.rerun()

def get_pending_question() -> str:
    """Get and clear any pending quick question."""
    return st.session_state.pop('pending_question', None)
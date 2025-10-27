"""
Main chat area component.
"""
import streamlit as st
from src.frontend.services.api_client import api_client
from src.frontend.utils.session_state import add_to_chat_history, get_chat_history

def render_chat_area():
    """Render the main chat area."""
    # Render quick questions in sidebar first
    render_quick_questions_sidebar()
    
    # Then render main chat interface (outside of columns)
    render_chat_interface()

def render_quick_questions_sidebar():
    """Render quick questions in the sidebar."""
    with st.sidebar:
        st.header("🎯 Quick Questions")
        
        # Define quick questions
        quick_questions = [
            "What is this document about?",
            "Can you summarize the main points?",
            "What are the key topics discussed?",
            "Are there any important definitions?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

def render_chat_interface():
    """Render the chat interface."""
    st.header("💬 Chat with Your Documents")
    
    # Display chat history
    render_chat_history()
    
    # Query input - MUST be at the bottom and not inside any containers
    render_chat_input()

def render_chat_history():
    """Render the chat history."""
    chat_history = get_chat_history()
    
    for i, (question, answer, confidence) in enumerate(chat_history):
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            st.write(answer)
            if confidence:
                st.caption(f"Confidence: {confidence}")

def render_chat_input():
    """Render the chat input and handle queries."""
    # Check if there's a pending question from quick questions
    pending_question = st.session_state.pop('pending_question', None)
    
    # Use chat_input at the bottom (not inside any layout containers)
    if pending_question:
        # If there's a pending question, process it immediately
        with st.chat_message("user"):
            st.write(pending_question)
        
        # Get and display answer with better loading
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing documents and generating answer..."):
                result = api_client.query_documents(pending_question)
            
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
                add_to_chat_history(pending_question, data["answer"], data["confidence"])
            else:
                st.error(f"Error: {result['error']}")
    
    # Regular chat input
    question = st.chat_input("Ask a question about your documents...")
    
    if question:
        # Add user question to chat immediately
        with st.chat_message("user"):
            st.write(question)
        
        # Get and display answer with better loading
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing documents and generating answer..."):
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
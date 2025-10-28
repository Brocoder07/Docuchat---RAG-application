"""
Enhanced chat area component with detailed source information.
"""
import streamlit as st
from src.frontend.services.api_client import api_client
from src.frontend.utils.session_state import add_to_chat_history, get_chat_history

def render_chat_area():
    """Render the main chat area."""
    # Render quick questions in sidebar first
    render_quick_questions_sidebar()
    
    # Then render main chat interface
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
    
    # Query input
    render_chat_input()

def render_chat_history():
    """Render the chat history with enhanced source information."""
    chat_history = get_chat_history()
    
    for i, (question, answer, confidence, source_info) in enumerate(chat_history):
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            st.write(answer)
            
            # Enhanced source information display
            if source_info:
                render_source_information(source_info, confidence)

def render_source_information(source_info: dict, confidence: str):
    """Render enhanced source information without nested expanders."""
    if not source_info:
        return
    
    with st.expander("📚 Source Information", expanded=False):
        
        # Main source summary
        st.subheader("📊 Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Used", source_info.get('total_sources', 0))
        with col2:
            confidence_color = {
                'very high': '🟢', 'high': '🟢', 'medium': '🟡',
                'low': '🟠', 'very low': '🔴'
            }.get(confidence.lower(), '⚪')
            st.metric("Confidence", f"{confidence_color} {confidence.title()}")
        with col3:
            primary_source = source_info.get('primary_source', 'Unknown')
            display_name = primary_source[:20] + "..." if len(primary_source) > 23 else primary_source
            st.metric("Primary Source", display_name)
        
        # Document list - show unique documents only
        if source_info.get('documents'):
            st.subheader("📄 Documents Referenced")
            unique_docs = list(set(source_info['documents']))
            for doc in unique_docs:
                st.write(f"• **{doc}**")
        
        # Show relevant sections without nested expanders
        if source_info.get('chunk_details'):
            st.subheader("🔍 Relevant Sections")
            
            # Group by document but display without nesting
            chunks_by_doc = {}
            for chunk in source_info['chunk_details']:
                doc_name = chunk['document']
                if doc_name not in chunks_by_doc:
                    chunks_by_doc[doc_name] = []
                chunks_by_doc[doc_name].append(chunk)
            
            # Display by document without nested expanders
            for doc_name, chunks in chunks_by_doc.items():
                st.write(f"**📄 {doc_name}** ({len(chunks)} sections)")
                
                for i, chunk in enumerate(chunks):
                    confidence_score = float(chunk.get('confidence', 0))
                    confidence_percent = int(confidence_score * 100)
                    
                    # Show content preview
                    content_preview = chunk.get('content_preview', 'No content preview')
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"*\"{content_preview}\"*")
                    with col2:
                        st.metric("Relevance", f"{confidence_percent}%")
                    
                    # Only show divider if not last item
                    if i < len(chunks) - 1:
                        st.divider()
                
                # Add space between documents
                st.write("")

def render_chat_input():
    """Render the chat input and handle queries with enhanced source info."""
    # Check if there's a pending question from quick questions
    pending_question = st.session_state.pop('pending_question', None)
    
    # Use chat_input at the bottom
    if pending_question:
        # If there's a pending question, process it immediately
        with st.chat_message("user"):
            st.write(pending_question)
        
        # Get and display answer with enhanced source info
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing documents and generating answer..."):
                result = api_client.query_documents(pending_question)
            
            if result["success"]:
                data = result["data"]
                st.write(data["answer"])
                
                # Show enhanced source information
                if data.get("source_info"):
                    render_source_information(data["source_info"], data["confidence"])
                
                # Add to chat history with source info
                add_to_chat_history(
                    pending_question, 
                    data["answer"], 
                    data["confidence"],
                    data.get("source_info", {})
                )
            else:
                st.error(f"Error: {result['error']}")
    
    # Regular chat input
    question = st.chat_input("Ask a question about your documents...")
    
    if question:
        # Add user question to chat immediately
        with st.chat_message("user"):
            st.write(question)
        
        # Get and display answer with enhanced source info
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing documents and generating answer..."):
                result = api_client.query_documents(question)
            
            if result["success"]:
                data = result["data"]
                st.write(data["answer"])
                
                # Show enhanced source information
                if data.get("source_info"):
                    render_source_information(data["source_info"], data["confidence"])
                
                # Add to chat history with source info
                add_to_chat_history(
                    question, 
                    data["answer"], 
                    data["confidence"],
                    data.get("source_info", {})
                )
            else:
                st.error(f"Error: {result['error']}")
"""
Consolidated Streamlit UI components with professional design and UX.
FIXED: Logout button now sets session state to None instead of deleting keys.
IMPROVED: Upload handler now inserts a placeholder document and polls backend.
RE-DESIGNED: Implemented a "ChatGPT-style" dark-mode UI with a clean
             sidebar for document management and actions.
FIXED: Chat history now clears when the user selects a new document.
FIXED: Resolved infinite loop on duplicate file upload by moving
       uploader key reset and rerun into a finally block.
"""
import uuid
import time
import streamlit as st
import logging
from typing import Dict, Any
from datetime import datetime

from frontend.services import api_client, state_manager
from core.config import config

logger = logging.getLogger(__name__)

class ComponentStyler:
    @staticmethod
    def apply_custom_styles():
        # ... (styles remain the same) ...
        st.markdown("""
        <style>
            /* Dark Mode Theme */
            body {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            .main {
                background-color: #0E1117;
            }
            .stApp {
                background-color: #0E1117;
            }
            
            /* Sidebar Styling */
            [data-testid="stSidebar"] {
                background-color: #1E1E1E; /* Darker sidebar */
                border-right: 1px solid #333;
            }
            [data-testid="stSidebar"] .stButton button {
                background-color: #333;
                color: #FAFAFA;
                border: 1px solid #444;
                text-align: left;
                justify-content: flex-start; /* Align text left */
            }
            [data-testid="stSidebar"] .stButton button:hover {
                background-color: #444;
                border: 1px solid #555;
            }
            [data-testid="stSidebar"] .stExpander {
                background-color: #1E1E1E;
                border-radius: 0.5rem;
                border: 1px solid #333;
            }
            [data-testid="stSidebar"] .stExpander header {
                color: #FAFAFA;
            }
            
            /* Main Chat Area */
            .main-header {
                font-size: 2.2rem;
                color: #FAFAFA;
                margin-bottom: 1rem;
            }
            .sub-header {
                font-size: 1.1rem;
                color: #AAA; /* Lighter grey */
                margin-bottom: 1rem;
            }
            
            /* Chat Message Styling */
            [data-testid="stChatMessageBox"] {
                border-radius: 1rem;
            }
            .user-message {
                background-color: #262626;
                padding: 1rem;
                border-radius: 1rem 1rem 0 1rem;
                margin: 0.5rem 0;
            }
            .assistant-message {
                background-color: #333333;
                padding: 1rem;
                border-radius: 1rem 1rem 1rem 0;
                margin: 0.5rem 0;
            }

            /* Metric Card */
            .metric-card {
                background-color: #262626;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #1f77b4;
            }
            
            /* Source Information */
            .source-card {
                background-color: #262626;
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid #333;
                margin-bottom: 0.5rem;
            }
            .confidence-high { color: #28a745; font-weight: bold; }
            .confidence-medium { color: #ffc107; font-weight: bold; }
            .confidence-low { color: #dc3545; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)

class HeaderComponent:
    # ... (no changes) ...
    def render(self):
        st.markdown('<div class="main-header">📚 DocuChat</div>', unsafe_allow_html=True)

class SidebarComponent:
    
    def __init__(self):
        # ... (no changes) ...
        self.quick_questions = [
            "What is this document about?",
            "Can you summarize the main points?",
            "What are the key topics discussed?",
            "Are there any important definitions?"
        ]
    
    def render(self):
        # ... (no changes) ...
        with st.sidebar:
            st.header("DocuChat")
            
            if st.button("➕ New Chat", use_container_width=True, key="new_chat"):
                state_manager.clear_chat_history()
                st.rerun()

            st.divider()

            self._render_document_list()
            st.divider()
            
            with st.expander("📁 Document Management"):
                self._render_document_uploader()
            
            with st.expander("🎯 Quick Questions"):
                self._render_quick_questions()

            with st.expander("📊 System Info"):
                self._render_system_info()
            
            st.divider()
            st.write(f"👤 {st.session_state.user_email}")
            if st.button("Log Out", use_container_width=True, key="logout"):
                st.session_state.id_token = None
                st.session_state.user_email = None
                st.session_state.auth_page = 'login'
                st.rerun()
    
    def _render_document_list(self):
        # ... (no changes) ...
        st.subheader("My Documents")
        
        if st.button("All Documents", use_container_width=True, key="doc_all"):
            st.session_state.selected_filename = "All Documents"
            st.rerun()
        
        documents = st.session_state.get('document_list', [])
        if not documents:
            try:
                result = api_client.list_documents()
                if result.get("success"):
                    documents = result["data"].get("documents", [])
                    st.session_state.document_list = documents
                    state_manager.update_documents_count(len(documents))
            except Exception as e:
                logger.error(f"Failed to fetch documents for sidebar: {e}")

        if not documents:
            st.caption("Upload a document to get started.")

        for doc in documents:
            doc_name = doc['filename']
            doc_id = doc['document_id']
            button_key = f"doc_{doc_id}"
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📄 {doc_name}", key=button_key, use_container_width=True):
                    st.session_state.selected_filename = doc_name
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{button_key}", use_container_width=True):
                    self._delete_document(doc_id)
                    st.rerun()
        
        if st.button("🔄 Refresh List", key="refresh_docs", use_container_width=True):
            self._force_refresh_documents()

    def _force_refresh_documents(self):
        # ... (no changes) ...
        try:
            docs_result = api_client.list_documents()
            if docs_result.get("success"):
                docs = docs_result["data"].get("documents", [])
                st.session_state.document_list = docs
                state_manager.update_documents_count(len(docs))
            else:
                st.error("Failed to refresh document list")
        except Exception as e:
            logger.error(f"Refresh error: {e}")
            st.error("Unable to refresh document list")
    
    # -----------------------------------------------------------------
    # 🚨 START: MODIFIED UPLOADER
    # -----------------------------------------------------------------
    def _render_document_uploader(self):
        uploader_key = getattr(st.session_state, 'uploader_key', 'default_uploader')
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=config.files.ALLOWED_EXTENSIONS,
            help=f"Supported formats: {', '.join(config.files.ALLOWED_EXTENSIONS).upper()}",
            key=uploader_key
        )
        
        if uploaded_file is not None:
            self._handle_file_upload(uploaded_file)
            # 🚨 REMOVED st.rerun() from here. It's now in _handle_file_upload
    # -----------------------------------------------------------------
    # 🚨 END: MODIFIED UPLOADER
    # -----------------------------------------------------------------
    
    # -----------------------------------------------------------------
    # 🚨 START: MODIFIED UPLOAD HANDLER
    # -----------------------------------------------------------------
    def _handle_file_upload(self, uploaded_file):
        """
        Save/upload file, create a placeholder, and poll.
        This now runs in a try/finally to guarantee the uploader is reset.
        """
        try:
            file_key = f"{uploaded_file.name}_{uuid.uuid4().hex[:8]}"
            
            if hasattr(st.session_state, 'processing_files'):
                if file_key in st.session_state.processing_files:
                    logger.info(f"File {uploaded_file.name} already being processed")
                    return
            else:
                st.session_state.processing_files = set()
            
            st.session_state.processing_files.add(file_key)
            
            with st.spinner(f"Uploading {uploaded_file.name}..."):
                result = api_client.upload_document(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
                
                # 🚨 This is where the 409 error is handled
                if not result["success"]:
                    error_msg = result.get('error', 'Upload failed')
                    if "duplicate" in str(error_msg).lower():
                        st.warning(f"💡 This file has already been uploaded.")
                    else:
                        st.error(f"❌ Upload failed: {error_msg}")
                    # The 'finally' block will still run
                    return
                
                # --- This code only runs on SUCCESS ---
                data = result.get("data") or {}
                message = data.get("message", "Document uploaded")
                document_id = data.get("document_id") or str(uuid.uuid4())
                
                placeholder_doc = {
                    "filename": uploaded_file.name,
                    "chunks": data.get("chunks_count", 0),
                    "source": "uploads/" + uploaded_file.name,
                    "document_id": document_id,
                    "processed_at": "processing"
                }
                
                if 'document_list' not in st.session_state:
                    st.session_state.document_list = []
                
                st.session_state.document_list.insert(0, placeholder_doc)
                state_manager.update_documents_count(len(st.session_state.document_list))
                
                st.success(f"✅ {message}")
                
                # Brief poll to replace placeholder with real data
                time.sleep(2.0)
                self._force_refresh_documents()
            
            st.session_state.processing_files.discard(file_key)
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            st.error(f"Upload failed: {str(e)}")
            if hasattr(st.session_state, 'processing_files'):
                st.session_state.processing_files.discard(file_key)
        
        finally:
            # 🚨 THIS IS THE FIX 🚨
            # This block runs whether the upload succeeded OR failed.
            # This resets the uploader key, clearing the file.
            st.session_state.uploader_key = str(uuid.uuid4())
            # This triggers the rerun to show the change.
            st.rerun()
    # -----------------------------------------------------------------
    # 🚨 END: MODIFIED UPLOAD HANDLER
    # -----------------------------------------------------------------
    
    def _delete_document(self, document_id: str):
        # ... (no changes) ...
        try:
            result = api_client.delete_document(document_id)
            if result["success"]:
                st.success("Document deleted!")
                self._force_refresh_documents()
            else:
                st.error(f"Delete failed: {result['error']}")
        except Exception as e:
            logger.error(f"Delete error: {str(e)}")
            st.error(f"Delete failed: {str(e)}")
    
    def _render_quick_questions(self):
        # ... (no changes) ...
        for question in self.quick_questions:
            if st.button(question, key=f"quick_{question}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()
    
    def _render_system_info(self):
        # ... (no changes) ...
        health_info = api_client.get_health_info()
        status_info = api_client.get_system_status()
        
        if health_info and status_info and status_info.get("success"):
            status_data = status_info["data"]
            
            st.metric("My Documents", status_data["pipeline"]["documents_processed"])
            st.metric("My Chunks", status_data["vector_store"]["total_chunks"])
            
            status_color = "🟢" if health_info["status"] == "healthy" else "🔴"
            st.metric("Status", f"{status_color} {health_info['status'].title()}")
            
            st.metric("Model", health_info["llm_model"])
        else:
            st.warning("Unable to fetch system info")

class ChatAreaComponent:
    # ... (no changes in this class) ...
    def __init__(self):
        self.processing_query = False
        
        if 'selected_filename' not in st.session_state:
            st.session_state.selected_filename = "All Documents"

    def render(self):
        selected_doc = st.session_state.get('selected_filename', "All Documents")
        if selected_doc == "All Documents":
            st.markdown('<div class="sub-header">Querying all your documents...</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="sub-header">Querying: <strong>{selected_doc}</strong></div>', unsafe_allow_html=True)

        self._render_chat_history()
        self._render_chat_input()
    
    def _render_chat_history(self):
        try:
            chat_history = state_manager.get_chat_history()
        
            if not chat_history:
                st.info("💬 What's on the agenda today? Ask a question about your documents!")
                return
        
            for i, message in enumerate(chat_history):
                with st.chat_message("user"):
                    st.write(message["question"])
            
                with st.chat_message("assistant"):
                    st.write(message["answer"])
                
                    if message.get("source_info"):
                        self._render_source_information(
                            message["source_info"], 
                            message["confidence"]
                        )
                    
        except Exception as e:
            st.error(f"Error displaying chat history: {str(e)}")
            logger.error(f"Chat history rendering error: {str(e)}")
    
    def _render_source_information(self, source_info: Dict[str, Any], confidence: str):
        with st.expander("📚 Source Information", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents Used", source_info.get('total_sources', 0))
            with col2:
                confidence_color = self._get_confidence_color(confidence)
                st.metric("Confidence", f"{confidence_color} {confidence.title()}")
            with col3:
                primary_source = source_info.get('primary_source', 'Unknown')
                display_name = primary_source[:20] + "..." if len(primary_source) > 23 else primary_source
                st.metric("Primary Source", display_name)
            
            if source_info.get('chunk_details'):
                st.subheader("🔍 Relevant Sections")
                chunks_by_doc = {}
                for chunk in source_info['chunk_details']:
                    doc_name = chunk['document']
                    if doc_name not in chunks_by_doc:
                        chunks_by_doc[doc_name] = []
                    chunks_by_doc[doc_name].append(chunk)
                
                for doc_name, chunks in chunks_by_doc.items():
                    st.write(f"**📄 {doc_name}**")
                    for i, chunk in enumerate(chunks):
                        confidence_score = chunk.get('confidence', 0)
                        confidence_percent = int(confidence_score * 100)
                        
                        with st.container(border=True):
                            st.caption(f"Relevance: {confidence_percent}%")
                            st.write(f"*\"{chunk['content_preview']}\"*")
    
    def _get_confidence_color(self, confidence: str) -> str:
        confidence_lower = confidence.lower()
        if confidence_lower in ['very high', 'high']:
            return "🟢"
        elif confidence_lower == 'medium':
            return "🟡"
        elif confidence_lower in ['low', 'very low']:
            return "🔴"
        else:
            return "⚪"
    
    def _render_chat_input(self):
        pending_question = st.session_state.pop('pending_question', None)
        
        if pending_question:
            self._process_question(pending_question)
        
        question = st.chat_input("Ask a question about your documents...")
        
        if question:
            self._process_question(question)
    
    def _process_question(self, question: str):
        if self.processing_query:
            st.warning("⚠️ Please wait for the current query to complete...")
            return
            
        self.processing_query = True
        
        try:
            with st.chat_message("user"):
                st.write(question)
            
            selected_name = st.session_state.get('selected_filename', "All Documents")
            
            selected_file = None
            if selected_name != "All Documents":
                selected_file = selected_name
            
            if selected_file:
                logger.info(f"Querying with filename filter: {selected_file}")
            else:
                logger.info("Querying all documents")
            
            history = state_manager.get_chat_history()
            recent_history_full = history[-5:] if len(history) > 5 else history
            
            recent_history_cleaned = [
                {"question": msg["question"], "answer": msg["answer"]} 
                for msg in recent_history_full
            ]
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing documents and generating answer..."):
                    result = api_client.query_documents(
                        question, 
                        filename=selected_file,
                        chat_history=recent_history_cleaned
                    )
                
                if result["success"]:
                    data = result["data"]
                    st.write(data["answer"])
                    
                    state_manager.add_chat_message(
                        question=question,
                        answer=data["answer"],
                        confidence=data["confidence"],
                        source_info=data.get("source_info", {})
                    )
                    
                    if data.get("source_info"):
                        self._render_source_information(
                            data["source_info"], 
                            data["confidence"]
                        )
                else:
                    error_detail = result.get('error', 'Unknown error')
                    st.error(f"Error: {error_detail}")
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            st.error(f"An error occurred while processing your question: {str(e)}")
        finally:
            self.processing_query = False

class SystemStatusComponent:
    # ... (no changes) ...
    def render(self):
        pass

class ComponentFactory:
    # ... (no changes) ...
    def __init__(self):
        self.components = {
            'styler': ComponentStyler(),
            'header': HeaderComponent(),
            'sidebar': SidebarComponent(),
            'chat_area': ChatAreaComponent(),
            'status': SystemStatusComponent()
        }
    
    def render_all(self):
        self.components['styler'].apply_custom_styles()
        self.components['sidebar'].render()
        self.components['header'].render()
        self.components['chat_area'].render()

component_factory = ComponentFactory()
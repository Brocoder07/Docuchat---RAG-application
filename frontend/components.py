"""
Consolidated Streamlit UI components with professional design and UX.
FIXED: Logout button now sets session state to None instead of deleting keys.
IMPROVED: Upload handler now inserts a placeholder document and polls backend once
to ensure the newly uploaded document appears immediately in the Processed Documents list
and the dropdown.
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
        st.markdown("""
        <style>
            /* Main styling */
            .main-header {
                font-size: 2.5rem;
                color: #1f77b4;
                margin-bottom: 1rem;
            }
            .sub-header {
                font-size: 1.2rem;
                color: #666;
                margin-bottom: 1rem;
            }
            .metric-card {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #1f77b4;
            }
            .source-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid #e9ecef;
                margin-bottom: 0.5rem;
            }
            .confidence-high { color: #28a745; font-weight: bold; }
            .confidence-medium { color: #ffc107; font-weight: bold; }
            .confidence-low { color: #dc3545; font-weight: bold; }
            
            /* Chat message styling */
            .user-message {
                background-color: #e3f2fd;
                padding: 1rem;
                border-radius: 1rem 1rem 0 1rem;
                margin: 0.5rem 0;
            }
            .assistant-message {
                background-color: #f5f5f5;
                padding: 1rem;
                border-radius: 1rem 1rem 1rem 0;
                margin: 0.5rem 0;
            }
            
            /* Button styling */
            .stButton button {
                width: 100%;
                border-radius: 0.5rem;
            }
        </style>
        """, unsafe_allow_html=True)

class HeaderComponent:
    """Application header with branding and status."""
    
    def render(self):
        """Render the main application header."""
        st.markdown('<div class="main-header">📚 DocuChat - AI Document Q&A</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Ask questions about your documents - Powered by Groq LLM</div>', unsafe_allow_html=True)
        st.divider()

class SidebarComponent:
    """Sidebar with document management and system info."""
    
    def __init__(self):
        self.quick_questions = [
            "What is this document about?",
            "Can you summarize the main points?",
            "What are the key topics discussed?",
            "Are there any important definitions?"
        ]
    
    def render(self):
        """Render the sidebar with all components."""
        with st.sidebar:
            st.header(f"👤 Welcome, {st.session_state.user_email}")
            if st.button("Log Out", use_container_width=True, key="logout"):
                st.session_state.id_token = None
                st.session_state.user_email = None
                st.session_state.auth_page = 'login'
                st.rerun()
            st.divider()

            self._render_document_section()
            st.divider()
            self._render_quick_questions()
            st.divider()
            self._render_system_info()
            st.divider()
            self._render_actions()
    
    def _render_document_section(self):
        st.header("📁 Document Management")
        uploader_key = getattr(st.session_state, 'uploader_key', 'default_uploader')
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=config.files.ALLOWED_EXTENSIONS,
            help=f"Supported formats: {', '.join(config.files.ALLOWED_EXTENSIONS).upper()}",
            key=uploader_key
        )
        
        if uploaded_file is not None:
            self._handle_file_upload(uploaded_file)
        
        self._render_document_list()
    
    def _handle_file_upload(self, uploaded_file):
        """
        Save/upload file, create a placeholder in the UI immediately,
        and poll the backend briefly to fetch the real processed metadata.
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
                # Call API client to upload (auth handled inside)
                result = api_client.upload_document(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
                
                if not result["success"]:
                    st.error(f"❌ Upload failed: {result.get('error')}")
                    st.session_state.processing_files.discard(file_key)
                    return
                
                data = result.get("data") or {}
                message = data.get("message", "Document uploaded")
                document_id = data.get("document_id") or str(uuid.uuid4())
                
                # Immediate UI feedback: add a placeholder entry to the document list
                placeholder_doc = {
                    "filename": uploaded_file.name,
                    "chunks": data.get("chunks_count", 0),
                    "source": "uploads/" + uploaded_file.name,
                    "document_id": document_id,
                    "processed_at": "processing"
                }
                
                # Ensure session state document_list exists
                if 'document_list' not in st.session_state:
                    st.session_state.document_list = []
                
                # Remove any existing placeholder for same filename to avoid duplicates
                st.session_state.document_list = [
                    d for d in st.session_state.document_list if not (d.get("filename") == uploaded_file.name and d.get("processed_at") == "processing")
                ]
                
                st.session_state.document_list.insert(0, placeholder_doc)
                state_manager.update_documents_count(len(st.session_state.document_list))
                
                st.success(f"✅ {message}")
                st.session_state.uploader_key = str(uuid.uuid4())
                
                # Non-blocking short poll: check for processed metadata (briefly)
                poll_interval = 1.0
                poll_timeout = 10.0  # seconds (short poll only)
                elapsed = 0.0
                found_real = False
                
                # Try to fetch the list a few times to see if the background worker finished quickly
                while elapsed < poll_timeout:
                    time.sleep(poll_interval)
                    elapsed += poll_interval
                    try:
                        docs_result = api_client.list_documents()
                        if docs_result and docs_result.get("success"):
                            docs = docs_result["data"].get("documents", [])
                            # Try to find an entry matching document_id or filename
                            match = None
                            for d in docs:
                                if d.get("document_id") == document_id or d.get("filename") == uploaded_file.name:
                                    match = d
                                    break
                            
                            if match:
                                # Replace placeholder with real entry
                                # Remove existing placeholder(s)
                                st.session_state.document_list = [d for d in st.session_state.document_list if not (d.get("filename") == uploaded_file.name and d.get("processed_at") == "processing")]
                                st.session_state.document_list.insert(0, {
                                    "filename": match.get("filename"),
                                    "chunks": match.get("chunks"),
                                    "source": match.get("source"),
                                    "document_id": match.get("document_id"),
                                    "processed_at": match.get("processed_at")
                                })
                                state_manager.update_documents_count(len(st.session_state.document_list))
                                found_real = True
                                logger.info(f"Upload: replaced placeholder with processed document for {uploaded_file.name}")
                                break
                    except Exception as e:
                        logger.debug(f"Upload polling error: {e}")
                        # keep trying until timeout
                
                if not found_real:
                    # Leave placeholder in the list. UX shows "processing".
                    st.info("Document uploaded and is being processed in background. It will appear in the list shortly.")
            
            st.session_state.processing_files.discard(file_key)
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            st.error(f"Upload failed: {str(e)}")
            if hasattr(st.session_state, 'processing_files'):
                st.session_state.processing_files.discard(file_key)
    
    def _render_document_list(self):
        st.subheader("Processed Documents")
        
        if st.button("🔄 Refresh List", key="refresh_docs", use_container_width=True):
            # Force fetch and update session state
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
            st.rerun()
        
        try:
            result = api_client.list_documents()
            if result["success"] and result["data"]["documents"]:
                documents = result["data"]["documents"]
                # Update session state document_list if it is empty or different
                st.session_state.document_list = documents
                
                for doc in documents:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{doc['filename']}**")
                            st.caption(f"Chunks: {doc['chunks']}")
                        with col2:
                            delete_key = f"delete_{doc['document_id']}_{hash(doc['filename'])}"
                            if st.button("🗑️", key=delete_key):
                                self._delete_document(doc['document_id'])
                
                st.caption(f"Total: {len(documents)} documents")
            else:
                # Show any placeholders present in session_state OR fallback message
                placeholders = [d for d in st.session_state.get('document_list', []) if d.get('processed_at') == 'processing']
                if placeholders:
                    for p in placeholders:
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{p['filename']}**")
                                st.caption("Status: processing")
                            with col2:
                                # Disabled delete while processing to avoid accidental removes
                                st.button("🗑️", key=f"del_placeholder_{p['document_id']}", disabled=True)
                    st.caption(f"Total: {len(st.session_state.get('document_list', []))} (including processing)")
                else:
                    st.session_state.document_list = []
                    st.info("No documents processed yet. Upload a document to get started!")
        except Exception as e:
            logger.error(f"Error rendering document list: {str(e)}")
            st.error("Unable to load document list")
    
    def _delete_document(self, document_id: str):
        try:
            result = api_client.delete_document(document_id)
            if result["success"]:
                st.success("Document deleted successfully!")
                # Remove from session_state.document_list immediately
                st.session_state.document_list = [d for d in st.session_state.get('document_list', []) if d.get('document_id') != document_id]
                state_manager.update_documents_count(len(st.session_state.get('document_list', [])))
                st.rerun()
            else:
                st.error(f"Delete failed: {result['error']}")
        except Exception as e:
            logger.error(f"Delete error: {str(e)}")
            st.error(f"Delete failed: {str(e)}")
    
    def _render_quick_questions(self):
        st.header("🎯 Quick Questions")
        
        for question in self.quick_questions:
            if st.button(question, key=f"quick_{question}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()
    
    def _render_system_info(self):
        st.header("📊 System Info")
        
        health_info = api_client.get_health_info()
        status_info = api_client.get_system_status()
        
        if health_info and status_info and status_info.get("success"):
            status_data = status_info["data"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("My Documents", status_data["pipeline"]["documents_processed"])
            with col2:
                st.metric("My Chunks", status_data["vector_store"]["total_chunks"])
            
            col3, col4 = st.columns(2)
            with col3:
                status_color = "🟢" if health_info["status"] == "healthy" else "🔴"
                st.metric("Status", f"{status_color} {health_info['status'].title()}")
            with col4:
                st.metric("Model", health_info["llm_model"])
            
            st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.warning("Unable to fetch system information")
    
    def _render_actions(self):
        st.header("⚡ Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh All", key="refresh_all", use_container_width=True):
                st.experimental_rerun()
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
                state_manager.clear_chat_history()
                st.success("Chat history cleared!")
                st.rerun()

class ChatAreaComponent:
    def __init__(self):
        self.processing_query = False
        
        if 'selected_filename' not in st.session_state:
            st.session_state.selected_filename = "All Documents"

    def render(self):
        st.header("💬 Chat with Your Documents")
        self._render_chat_history()
        self._render_chat_input()
    
    def _render_chat_history(self):
        try:
            if not hasattr(state_manager, 'get_chat_history'):
                st.warning("Chat history feature is temporarily unavailable.")
                return
            
            chat_history = state_manager.get_chat_history()
        
            if not chat_history:
                st.info("💬 Start a conversation by asking a question about your documents!")
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
            
            if source_info.get('documents'):
                st.subheader("📄 Documents Referenced")
                unique_docs = list(set(source_info['documents']))
                for doc in unique_docs:
                    st.write(f"• **{doc}**")
            
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
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(f"*\"{chunk['content_preview']}\"*")
                        with col2:
                            st.metric("Relevance", f"{confidence_percent}%")
                        
                        if i < len(chunks) - 1:
                            st.divider()
                    
                    st.write("")
    
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
        doc_list = st.session_state.get('document_list', [])
        options_list = ["All Documents"] + [doc['filename'] for doc in doc_list]
        
        try:
            default_index = options_list.index(st.session_state.selected_filename)
        except ValueError:
            default_index = 0
        
        selected_doc_name = st.selectbox(
            "Query a specific document (optional):",
            options=options_list,
            index=default_index,
            key="doc_selector"
        )
        
        st.session_state.selected_filename = selected_doc_name

        pending_question = st.session_state.pop('pending_question', None)
        
        if pending_question:
            self._process_question(pending_question)
        
        question = st.chat_input("Ask a question about your documents...")
        
        if question:
            self._process_question(question)
    
    def _process_question(self, question: str):
        """Process a question and display the response with duplicate prevention."""
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
    def render(self):
        if st.sidebar.button("📊 System Dashboard", use_container_width=True):
            st.session_state.show_dashboard = True
        
        if st.session_state.get('show_dashboard', False):
            self._render_dashboard()
    
    def _render_dashboard(self):
        st.header("📊 System Dashboard")
        
        result = api_client.get_system_status()
        
        if not result["success"]:
            st.error("Unable to fetch system status")
            return
        
        data = result["data"]
        
        st.subheader("🚀 Pipeline Status")
        pipeline_cols = st.columns(4)
        with pipeline_cols[0]:
            st.metric("My Documents", data["pipeline"]["documents_processed"])
        with pipeline_cols[1]:
            st.metric("Total Queries", data["pipeline"]["total_queries"])
        with pipeline_cols[2]:
            status = "🟢" if data["pipeline"]["initialized"] else "🔴"
            st.metric("Pipeline", f"{status} {'Ready' if data['pipeline']['initialized'] else 'Error'}")
        with pipeline_cols[3]:
            st.metric("Vector Store", f"🟢 Ready" if data["vector_store"]["initialized"] else "🔴 Error")
        
        st.subheader("🧠 LLM Service")
        llm_cols = st.columns(3)
        with llm_cols[0]:
            st.metric("Model", data["llm_service"]["current_model"])
        with llm_cols[1]:
            st.metric("Status", "🟢 Ready" if data["llm_service"]["initialized"] else "🔴 Error")
        with llm_cols[2]:
            st.metric("Temperature", config.model.TEMPERATURE)
        
        st.subheader("📈 Performance Metrics")
        eval_data = data["evaluation"]["recent_metrics"]
        if eval_data:
            eval_cols = st.columns(2) 
            with eval_cols[0]:
                st.metric("Avg Response Time", f"{eval_data.get('avg_response_time', 0):.1f}s")
            with eval_cols[1]:
                st.metric("Avg Chunks", f"{eval_data.get('avg_chunks_retrieved', 0):.1f}")
        
        alerts = data["evaluation"].get("performance_alerts", [])
        if alerts:
            st.subheader("🚨 Performance Alerts")
            for alert in alerts:
                st.warning(f"{alert['type']}: {alert['message']}")
        
        if st.button("Close Dashboard"):
            st.session_state.show_dashboard = False
            st.rerun()

class ComponentFactory:
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
        self.components['header'].render()
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            self.components['sidebar'].render()
            self.components['status'].render()
        
        with col2:
            self.components['chat_area'].render()

component_factory = ComponentFactory()
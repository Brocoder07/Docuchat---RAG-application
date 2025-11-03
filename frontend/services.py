"""
Frontend API client with robust error handling and retry logic.
"""
import logging
import requests
from typing import Dict, Any, Optional, List
import time
from datetime import datetime

from core.config import config

logger = logging.getLogger(__name__)

class APIClient:
    """
    Robust API client with retry mechanism and comprehensive error handling.
    """
    
    def __init__(self):
        self.base_url = f"http://{config.api.HOST}:{config.api.PORT}/api/v1"
        self.default_timeout = 30
        self.query_timeout = 60   # 🚨 Specific timeout for queries
        self.max_retries = 3      # 🚨 Reduced from 3 to 2
        self.retry_delay = 2      # 🚨 Increased delay
    
    def _make_request(self, method: str, endpoint: str, timeout: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and comprehensive error handling.
        """
        if timeout is None:
            timeout = self.default_timeout
            
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    timeout=timeout,
                    **kwargs
                )
                
                # Handle successful response
                if 200 <= response.status_code < 300:
                    try:
                        return {
                            "success": True,
                            "data": response.json(),
                            "status_code": response.status_code
                        }
                    except ValueError as e:
                        logger.error(f"JSON parsing error for {url}: {str(e)}")
                        return {
                            "success": False,
                            "error": "Invalid JSON response from server",
                            "status_code": response.status_code
                        }
                
                # Handle client errors (no retry)
                if 400 <= response.status_code < 500:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', 'Client error occurred')
                        return {
                            "success": False,
                            "error": error_msg,
                            "status_code": response.status_code
                        }
                    except ValueError:
                        return {
                            "success": False,
                            "error": f"Client error: {response.status_code}",
                            "status_code": response.status_code
                        }
                
                # Handle server errors (retry)
                if attempt < self.max_retries - 1:
                    logger.warning(f"Server error {response.status_code} for {url}, retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"Server error: {response.status_code}",
                        "status_code": response.status_code
                    }
                    
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Timeout for {url}, retrying... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"Request timed out after {timeout} seconds"
                    }
                    
            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Connection error for {url}, retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return {
                        "success": False,
                        "error": "Cannot connect to backend API. Please ensure the server is running."
                    }
                    
            except requests.exceptions.RequestException as e:
                return {
                    "success": False,
                    "error": f"Network error: {str(e)}"
                }
        
        return {
            "success": False,
            "error": "Max retries exceeded"
        }
    
    def check_health(self) -> bool:
        """Check if API is healthy and responsive."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            return False
    
    def get_health_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed health information."""
        result = self._make_request("GET", "/health")
        return result.get("data") if result["success"] else None
    
    def upload_document(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Upload document with progress tracking."""
        try:
            files = {"file": (filename, file_data)}
            return self._make_request("POST", "/upload", files=files)
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return {
                "success": False,
                "error": f"Upload failed: {str(e)}"
            }
    
    # -----------------------------------------------------------------
    # 🚨 MODIFIED: Updated `query_documents` to send `filename`
    # -----------------------------------------------------------------
    def query_documents(self, question: str, top_k: int = 5, filename: Optional[str] = None) -> Dict[str, Any]:
        """Query documents with comprehensive error handling and longer timeout."""
        return self._make_request(
            "POST",
            "/query",
            timeout=self.query_timeout,  # 🚨 Use query-specific timeout
            json={"question": question, "top_k": top_k, "filename": filename} # 🚨 Send filename
        )
    # -----------------------------------------------------------------
    
    def list_documents(self) -> Dict[str, Any]:
        """Get list of processed documents."""
        return self._make_request("GET", "/documents")
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a specific document."""
        return self._make_request("DELETE", f"/documents/{document_id}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return self._make_request("GET", "/status")
    
    def get_evaluation_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get evaluation metrics."""
        return self._make_request("GET", f"/evaluation/metrics?hours={hours}")

class SessionStateManager:
    """Manage frontend session state with persistence."""
    
    def __init__(self):
        self.default_state = {
            'chat_history': [],
            'documents_loaded': 0,
            'api_healthy': False,
            'last_update': None,
            'user_preferences': {}
        }
    
    def initialize_session(self):
        """Initialize or reset session state."""
        import streamlit as st
        
        for key, value in self.default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Update last update timestamp
        st.session_state['last_update'] = datetime.now().isoformat()
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get chat history from session state."""
        import streamlit as st
        return st.session_state.get('chat_history', [])
    
    def get_documents_count(self) -> int:
        """Get documents count from session state."""
        import streamlit as st
        return st.session_state.get('documents_loaded', 0)
    
    def add_chat_message(self, question: str, answer: str, confidence: str, source_info: Dict):
        """Add message to chat history with source information."""
        import streamlit as st
        
        chat_entry = {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'source_info': source_info,
            'timestamp': datetime.now().isoformat()
        }
        
        st.session_state.chat_history.append(chat_entry)
        
        # Keep only last 50 messages to prevent memory issues
        if len(st.session_state.chat_history) > 50:
            st.session_state.chat_history.pop(0)
    
    def clear_chat_history(self):
        """Clear chat history."""
        import streamlit as st
        st.session_state.chat_history = []
    
    def update_documents_count(self, count: int):
        """Update documents count in session state."""
        import streamlit as st
        st.session_state.documents_loaded = count
    
    def set_api_health(self, healthy: bool):
        """Update API health status."""
        import streamlit as st
        st.session_state.api_healthy = healthy

# Global instances
api_client = APIClient()
state_manager = SessionStateManager()
"""
Main Streamlit application with full Firebase authentication routing.
FIXED: Initialized st.session_state.selected_filename.
"""
import streamlit as st
import logging
import sys
import os
import time
from datetime import datetime

# CRITICAL FIX: Robust path resolution
try:
    possible_paths = [
        os.path.dirname(os.path.abspath(__file__)),
        os.getcwd(),
        os.path.join(os.getcwd(), 'frontend'),
        os.path.dirname(os.getcwd()),
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'core', 'config.py')):
            sys.path.insert(0, path)
            break
    else:
        raise ImportError("Could not find project root directory")
    
    from frontend.components import component_factory
    from frontend.services import api_client, state_manager
    from frontend.auth_components import auth_component_factory
    
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

from core.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ApplicationInitializer:
    
    def __init__(self):
        self.initialized = False
        self.startup_time = None
    
    def initialize(self) -> bool:
        """Initialize the application with comprehensive checks."""
        try:
            if hasattr(st.session_state, 'app_initialized') and st.session_state.app_initialized:
                return True
                
            logger.info("🚀 Initializing DocuChat Frontend...")
            self.startup_time = datetime.now()
            
            self._configure_page()
            
            # Initialize session state (basic keys)
            state_manager.initialize_session()
            
            # -----------------------------------------------------------------
            # 🚨 MODIFIED: Initialize all session state variables here
            # -----------------------------------------------------------------
            if 'app_initialized' not in st.session_state:
                st.session_state.app_initialized = False
            if 'id_token' not in st.session_state:
                st.session_state.id_token = None
            if 'user_email' not in st.session_state:
                st.session_state.user_email = None
            if 'auth_page' not in st.session_state:
                st.session_state.auth_page = 'login'
            if 'session_restored' not in st.session_state:
                st.session_state.session_restored = False
            
            # -----------------------------------------------------------------
            # 🚨 FIXED: Add the missing variable initialization
            # -----------------------------------------------------------------
            if 'selected_filename' not in st.session_state:
                st.session_state.selected_filename = "All Documents"
            # -----------------------------------------------------------------

            if not self._check_backend_connectivity():
                logger.error("Backend connectivity check failed")
                return False
            
            self.initialized = True
            st.session_state.app_initialized = True
            logger.info("✅ Frontend initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Frontend initialization failed: {str(e)}")
            return False
    
    def _configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="DocuChat - AI Document Q&A",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/brocoder07/docuchat---rag-application',
                'Report a bug': 'https://github.com/brocoder07/docuchat---rag-application/issues',
                'About': "DocuChat - AI-powered Document Q&A System"
            }
        )
    
    def _check_backend_connectivity(self) -> bool:
        """Check backend API connectivity (public /health endpoint)."""
        max_retries = 8
        retry_delay = 3
        
        for attempt in range(max_retries):
            try:
                if api_client.check_health():
                    logger.info("✅ Backend API is healthy")
                    state_manager.set_api_health(True)
                    return True
                else:
                    logger.warning(f"Backend not ready (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    
            except Exception as e:
                logger.warning(f"Backend check failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        logger.error("❌ Backend API is not available after all retries")
        state_manager.set_api_health(False)
        return False

class ErrorHandler:
    @staticmethod
    def render_error_page(error_message: str, error_details: str = None):
        st.error("🚨 Application Error")
        
        st.markdown(f"""
        ### We encountered an issue
        
        **Error:** {error_message}
        
        {f"*Details:* {error_details}" if error_details else ""}
        
        ### What you can do:
        1. **Refresh the page** - This often resolves temporary issues
        2. **Check the backend** - Ensure the API server is running
        3. **Check the logs** - Look for detailed error information
        4. **Contact support** - If the issue persists
        
        If you're running this locally, make sure to start the backend server:
        ```bash
        python -m api.main
        ```
        """)
        
        with st.expander("Technical Details (for debugging)"):
            st.code(f"""
            Error: {error_message}
            Details: {error_details}
            Time: {datetime.now()}
            Python: {sys.version}
            Streamlit: {st.__version__}
            """)
    
    @staticmethod
    def render_backend_unavailable():
        st.error("🔌 Backend Unavailable")
        
        st.markdown("""
        ### Backend API Server is Not Running
        
        The DocuChat frontend cannot connect to the backend API server.
        
        ### To fix this:
        
        1. **Start the backend server** (in a separate terminal):
        ```bash
        # Navigate to project root
        cd docuchat---rag-application
        
        # Start the backend API
        python -m api.main
        ```
        
        2. **Wait for the server to start** - You should see:
           ```
           🚀 Starting DocuChat API...
           ✅ RAG Pipeline initialized successfully
           INFO:     Uvicorn running on [http://127.0.0.1:8000](http://127.0.0.1:8000)
           ```
        
        3. **Refresh this page** once the backend is running
        
        ### Troubleshooting:
        - Check if port 8000 is available
        - Verify all dependencies are installed
        - Check the backend logs for errors
        - Ensure you're in the correct Python environment
        """)
        
        if st.button("🔄 Check Again"):
            st.experimental_rerun()

def main():
    """
    Main application entry point with authentication routing.
    """
    try:
        initializer = ApplicationInitializer()
        
        if not initializer.initialize():
            ErrorHandler.render_backend_unavailable()
            return
        
        component_factory.components['styler'].apply_custom_styles()
        
        # Try to restore the session from browser storage if it's not in st.session_state
        if not st.session_state.id_token and not st.session_state.session_restored:
            auth_component_factory.try_restore_session()
        
        # Now, check for the token again
        if not st.session_state.id_token:
            auth_component_factory.render_auth_page()
        else:
            component_factory.render_all()
            _render_footer(initializer.startup_time)

    except Exception as e:
        logger.error(f"🚨 Critical application error: {str(e)}", exc_info=True)
        ErrorHandler.render_error_page(
            "A critical error occurred in the application",
            str(e)
        )

def _render_footer(startup_time):
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if startup_time:
            st.caption(f"🕒 Started: {startup_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.caption("🕒 Started: Unknown")
    
    with col2:
        st.caption(f"🐍 Python: {sys.version.split()[0]}")
    
    with col3:
        st.caption("💡 DocuChat v2.0.0")
    
    with st.expander("🔍 Debug Information"):
        st.write("**Session State:**")
        safe_session_state = {}
        for k, v in st.session_state.items():
            if k != 'file_uploader' and not callable(v):
                if isinstance(v, (str, int, float, bool, type(None))):
                    if k == 'id_token' and v is not None:
                        safe_session_state[k] = f"firebase_id_token_******"
                    else:
                        safe_session_state[k] = v
                else:
                    safe_session_state[k] = f"<{type(v).__name__} object>"
        
        st.json(safe_session_state)
        
        st.write("**System Information:**")
        st.code(f"""
        Python: {sys.version}
        Streamlit: {st.__version__}
        Platform: {sys.platform}
        Working Directory: {os.getcwd()}
        """)

# Application entry point
if __name__ == "__main__":
    main()
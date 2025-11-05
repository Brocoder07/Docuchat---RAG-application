"""
NEW FILE: Handles Streamlit Login/Register UI using Pyrebase.
FIXED: Added try_restore_session to handle page refreshes.
"""
import streamlit as st
import logging
import pyrebase
from core.config import config

logger = logging.getLogger(__name__)

try:
    firebase = pyrebase.initialize_app(config.firebase.FIREBASE_WEB_CONFIG)
    auth = firebase.auth()
    logger.info("✅ Pyrebase initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize Pyrebase: {e}")
    st.error(f"Failed to initialize authentication service: {e}")


class AuthComponentFactory:
    
    # -----------------------------------------------------------------
    # 🚨 ADDED: New function to restore session from browser
    # -----------------------------------------------------------------
    def try_restore_session(self):
        """
        Checks browser local storage for a valid Firebase session.
        If found, populates st.session_state and reruns.
        """
        # This flag prevents an infinite rerun loop
        st.session_state.session_restored = True
        
        try:
            # `auth.current_user` checks local storage for a user
            # This is the key to persistence
            user = auth.current_user
            
            if user:
                logger.info(f"Restoring session for user: {user['email']}")
                # We have a user. Get their fresh ID token.
                # Pyrebase handles refreshing the token automatically.
                fresh_token = user['idToken']
                
                # Populate session state
                st.session_state.id_token = fresh_token
                st.session_state.user_email = user['email']
                
                # Rerun the app. app.py will now see the token and load the main app.
                st.rerun()
            else:
                logger.info("No active session found in browser storage.")
                
        except Exception as e:
            # This can happen if local storage is corrupted or token is invalid
            logger.warning(f"Failed to restore session: {e}")
            st.session_state.id_token = None
            st.session_state.user_email = None
    # -----------------------------------------------------------------

    def render_auth_page(self):
        """Renders the login or register page based on session state."""
        st.markdown('<div class="main-header">📚 DocuChat - AI Document Q&A</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Please log in or register to continue</div>', unsafe_allow_html=True)
        st.divider()
        
        auth_page = st.session_state.get('auth_page', 'login')
        
        if auth_page == 'login':
            self._render_login_form()
        else:
            self._render_register_form()

    def _render_login_form(self):
        """Renders the login form."""
        with st.form("login_form"):
            st.subheader("Login")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_pass")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                login_button = st.form_submit_button("Login")
            with col2:
                go_to_register = st.form_submit_button("Go to Register")
            
            if login_button:
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    with st.spinner("Logging in..."):
                        try:
                            user = auth.sign_in_with_email_and_password(email, password)
                            
                            st.session_state.id_token = user['idToken']
                            st.session_state.user_email = user['email']
                            logger.info(f"User {email} logged in successfully.")
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Firebase login failed: {e}")
                            st.error(f"Login failed: Invalid email or password.")
                            
            if go_to_register:
                st.session_state.auth_page = 'register'
                st.rerun()

    def _render_register_form(self):
        """Renders the registration form."""
        with st.form("register_form"):
            st.subheader("Register")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password (min 8 characters)", type="password", key="reg_pass")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                register_button = st.form_submit_button("Register")
            with col2:
                go_to_login = st.form_submit_button("Go to Login")
            
            if register_button:
                if not email or not password or not confirm_password:
                    st.error("Please fill out all fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters.")
                else:
                    with st.spinner("Registering..."):
                        try:
                            user = auth.create_user_with_email_and_password(email, password)
                            logger.info(f"User {email} registered successfully.")
                            st.success("Registration successful! Please log in.")
                            st.session_state.auth_page = 'login'
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Firebase registration failed: {e}")
                            st.error(f"Registration failed. The email may already be in use.")
            
            if go_to_login:
                st.session_state.auth_page = 'login'
                st.rerun()

# Global instance
auth_component_factory = AuthComponentFactory()
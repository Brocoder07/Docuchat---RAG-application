"""
Handles Streamlit Login/Register UI using Pyrebase.
Restored: added try_restore_session() to support session restoration on page reload.
"""
import streamlit as st
import logging
import pyrebase
from core.config import config
from typing import Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Initialize Pyrebase
# -----------------------------------------------------------------
try:
    firebase = pyrebase.initialize_app(config.firebase.FIREBASE_WEB_CONFIG)
    auth = firebase.auth()
    logger.info("✅ Pyrebase initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize Pyrebase: {e}")
    # When Streamlit loads this module during testing, st may not be available.
    try:
        st.error(f"Failed to initialize authentication service: {e}")
    except Exception:
        pass
# -----------------------------------------------------------------


class AuthComponentFactory:
    """
    Streamlit auth component factory.

    Methods:
    - try_restore_session(): attempt to restore an existing browser-stored session
    - render_auth_page(): render login or register views
    """

    # -----------------------------------------------------------------
    # Restores session from Pyrebase's client-side state (if present).
    # This prevents forcing the user to re-login after a page refresh.
    # -----------------------------------------------------------------
    def try_restore_session(self):
        """
        Check browser/local storage for a valid Firebase session (via Pyrebase).
        If found, populate st.session_state and trigger a rerun so the app shows the main UI.
        """
        # Prevent multiple restores in a single page load
        st.session_state.session_restored = True

        try:
            # auth.current_user is Pyrebase's way to access the current user saved in browser storage
            user = getattr(auth, "current_user", None)

            if user:
                # Pyrebase stores user info including idToken and email
                token = user.get("idToken") or user.get("refreshToken") or None
                email = user.get("email") or user.get("userEmail") or None

                if token:
                    st.session_state.id_token = token
                if email:
                    st.session_state.user_email = email

                logger.info(f"Restored session for user: {email}")
                # Rerun to let the app load authenticated UI
                st.rerun()
            else:
                logger.debug("No active session found in browser storage.")
                # Make sure session flags are consistent
                st.session_state.id_token = None
                st.session_state.user_email = None

        except Exception as e:
            # If anything goes wrong while restoring, clear session keys and continue
            logger.warning(f"Failed to restore session: {e}")
            st.session_state.id_token = None
            st.session_state.user_email = None

    # -----------------------------------------------------------------
    # Existing renderers (login/register)
    # -----------------------------------------------------------------
    def render_auth_page(self):
        """Renders the login or register page based on session state."""
        st.markdown('<div class="main-header">📚 DocuChat - AI Document Q&A</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Please log in or register to continue</div>', unsafe_allow_html=True)
        st.divider()

        auth_page = st.session_state.get("auth_page", "login")

        if auth_page == "login":
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
                            # Use Pyrebase to log in
                            user = auth.sign_in_with_email_and_password(email, password)

                            # Save the ID Token and email in session state
                            st.session_state.id_token = user.get("idToken")
                            st.session_state.user_email = user.get("email") or email
                            logger.info(f"User {email} logged in successfully.")
                            # Rerun to transition into authenticated UI
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Firebase login failed: {e}")
                            st.error("Login failed: Invalid email or password.")

            if go_to_register:
                st.session_state.auth_page = "register"
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
                            # Use Pyrebase to register
                            user = auth.create_user_with_email_and_password(email, password)
                            logger.info(f"User {email} registered successfully.")
                            st.success("Registration successful! Please log in.")
                            st.session_state.auth_page = "login"
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Firebase registration failed: {e}")
                            st.error("Registration failed. The email may already be in use.")

            if go_to_login:
                st.session_state.auth_page = "login"
                st.rerun()


# Global instance
auth_component_factory = AuthComponentFactory()
"""
NEW FILE: Handles Firebase Admin initialization and user authentication dependency.
"""
import logging
import firebase_admin
from firebase_admin import credentials, auth
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.config import config

logger = logging.getLogger(__name__)

# Initialize Firebase Admin
try:
    cred = credentials.Certificate(config.firebase.SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred)
    logger.info("✅ Firebase Admin SDK initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize Firebase Admin SDK: {e}")
    # Application should not start if this fails
    raise e

# Create the security dependency
http_bearer = HTTPBearer()

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(http_bearer)) -> str:
    """
    FastAPI dependency to verify Firebase ID token and return the user's UID.
    This will be used to protect all secure endpoints.
    """
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No credentials provided"
        )
    
    try:
        # Verify the token with Firebase
        id_token = creds.credentials
        decoded_token = auth.verify_id_token(id_token)
        
        # Get the User ID (uid) from the token
        uid = decoded_token.get("uid")
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid token: UID not found")
        
        return uid
    
    except auth.InvalidIdTokenError:
        logger.warning("Invalid Firebase ID token received.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    except Exception as e:
        logger.error(f"Error during token verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error verifying token"
        )
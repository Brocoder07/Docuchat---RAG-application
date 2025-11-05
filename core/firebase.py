"""
Firebase Admin initialization and FastAPI auth dependency.

- Supports real service-account JSON initialization and emulator scenarios.
- Reads REQUIRE_FIREBASE from environment (so core.config does not need to contain it).
- Lazy initialization on import; `init_firebase()` can be called explicitly if desired.
"""
import os
import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Lazy holder variables
_firebase_admin = None
_firebase_auth = None
_initialized = False

http_bearer = HTTPBearer()

def init_firebase() -> bool:
    """
    Initialize Firebase Admin SDK with the following order:
      1) If SERVICE_ACCOUNT_KEY_PATH exists -> use it.
      2) Else -> try Application Default Credentials.
      3) If FIREBASE_AUTH_EMULATOR_HOST is set, allow initialization for emulator usage.

    Returns True on success, False on failure.
    """
    global _firebase_admin, _firebase_auth, _initialized

    if _initialized:
        return True

    try:
        import firebase_admin
        from firebase_admin import credentials, auth as fb_auth

        emulator = os.getenv("FIREBASE_AUTH_EMULATOR_HOST", "").strip()
        if emulator:
            os.environ["FIREBASE_AUTH_EMULATOR_HOST"] = emulator
            logger.info(f"Using Firebase Auth emulator at {emulator}")

        cred_path = os.getenv("SERVICE_ACCOUNT_KEY_PATH", "").strip()
        project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()

        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, options={"projectId": project_id} if project_id else None)
            logger.info("✅ Firebase Admin SDK initialized successfully (service account)")
        else:
            try:
                # Try ADC (useful on GCP or environments with metadata)
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred, options={"projectId": project_id} if project_id else None)
                logger.info("✅ Firebase Admin SDK initialized with Application Default Credentials")
            except Exception as e_adc:
                # If emulator is configured, try minimal initialization
                if emulator:
                    try:
                        firebase_admin.initialize_app(options={"projectId": project_id} if project_id else None)
                        logger.info("✅ Firebase Admin SDK initialized for emulator (minimal init)")
                    except Exception as e2:
                        logger.warning(f"Firebase emulator init failed: {e2}")
                        _initialized = False
                        return False
                else:
                    logger.warning(f"Application Default Credentials unavailable: {e_adc}")
                    _initialized = False
                    return False

        _firebase_admin = firebase_admin
        _firebase_auth = fb_auth
        _initialized = True
        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize Firebase Admin SDK: {str(e)}", exc_info=True)
        _initialized = False
        return False

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(http_bearer)) -> str:
    """
    FastAPI dependency to verify Firebase ID token and return the user's UID.

    Behavior:
    - If Firebase isn't initialized, try lazy init.
    - If still not initialized:
        - If REQUIRE_FIREBASE=true -> respond with 503 Service Unavailable
        - Else -> respond with 401 Unauthorized (auth not configured)
    """
    global _firebase_auth, _initialized

    require_firebase_env = os.getenv("REQUIRE_FIREBASE", "true").lower() in ("1", "true", "yes")

    if not _initialized:
        ok = init_firebase()
        if not ok:
            if require_firebase_env:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication service is not configured")
            else:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication not configured in this environment")

    if not creds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No credentials provided")

    id_token = creds.credentials
    try:
        decoded_token = _firebase_auth.verify_id_token(id_token)
        uid = decoded_token.get("uid")
        if not uid:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token: UID not found")
        return uid
    except _firebase_auth.InvalidIdTokenError:
        logger.warning("Invalid Firebase ID token received")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")
    except _firebase_auth.ExpiredIdTokenError:
        logger.warning("Expired Firebase ID token")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Expired authentication token")
    except Exception as e:
        logger.error(f"Error during token verification: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Failed to verify authentication token")
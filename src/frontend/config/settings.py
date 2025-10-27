"""
Frontend configuration settings.
"""

class FrontendConfig:
    """Frontend configuration."""
    
    # API settings
    API_BASE_URL = "http://localhost:8000"
    
    # Page configuration
    PAGE_TITLE = "DocuChat - FREE Document Q&A"
    PAGE_ICON = "📚"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    # File upload settings
    ALLOWED_FILE_TYPES = ['pdf', 'txt', 'docx', 'md']
    MAX_FILE_SIZE_MB = 50
    
    # UI settings
    QUICK_QUESTIONS = [
        "What is this document about?",
        "Can you summarize the main points?",
        "What are the key topics discussed?",
        "Are there any important definitions?"
    ]

config = FrontendConfig()
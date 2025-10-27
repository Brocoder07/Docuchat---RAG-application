"""
API client for communicating with the backend.
"""
import requests
from typing import Optional, Dict, Any
from src.frontend.config.settings import config

class APIClient:
    """Client for communicating with the DocuChat API."""
    
    def __init__(self):
        self.base_url = config.API_BASE_URL
        self.timeout = 60  # Increased timeout to 60 seconds
    
    def check_health(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            return False
    
    def get_health_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed health information."""
        try:
            # Use the direct health endpoint without redirects
            response = requests.get(f"{self.base_url}/health/", timeout=5)
            if response.status_code == 200:
                return response.json()
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            pass
        return None
    
    def upload_document(self, file_data, filename: str) -> Dict[str, Any]:
        """
        Upload a document to the API.
        
        Args:
            file_data: File data to upload
            filename: Name of the file
            
        Returns:
            API response
        """
        try:
            files = {"file": (filename, file_data)}
            response = requests.post(
                f"{self.base_url}/documents/upload", 
                files=files,
                timeout=30
            )
            return {
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None,
                "error": response.json().get('detail') if response.status_code != 200 else None
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Connection error: {str(e)}"
            }
    
    def query_documents(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Send a query to the API.
        
        Args:
            question: The question to ask
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            API response
        """
        try:
            response = requests.post(
                f"{self.base_url}/chat/query",
                json={"question": question, "top_k": top_k},
                timeout=self.timeout  # Use the increased timeout
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": response.json().get('detail', 'Unknown error')
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"Request timed out after {self.timeout} seconds. The AI is taking longer than expected to process your question."
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Connection error: {str(e)}"
            }
    
    def list_documents(self) -> Dict[str, Any]:
        """Get list of processed documents."""
        try:
            response = requests.get(f"{self.base_url}/documents", timeout=10)
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": response.json().get('detail', 'Unknown error')
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Connection error: {str(e)}"
            }

# Global API client instance
api_client = APIClient()
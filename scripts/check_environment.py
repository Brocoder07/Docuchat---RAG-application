import sys
import pkg_resources

def check_environment():
    print(f"Python executable: {sys.executable}")
    print(f"Virtual environment active: {'venv' in sys.executable}")
    
    # Check key packages
    required_packages = ['fastapi', 'sentence-transformers', 'faiss-cpu', 'streamlit']
    
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✓ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"✗ {package}: NOT FOUND")

if __name__ == "__main__":
    check_environment()
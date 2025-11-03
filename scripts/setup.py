"""
Environment setup and dependency validation script.
Senior Engineer Principle: Automated setup with comprehensive checks.
"""
import sys
import subprocess
import importlib
import logging
from typing import List, Dict, Tuple

def setup_logging():
    """Configure logging for setup script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        logging.info(f"✅ Python {current_version[0]}.{current_version[1]} (>= {required_version[0]}.{required_version[1]})")
        return True
    else:
        logging.error(f"❌ Python {current_version[0]}.{current_version[1]} is not supported. Required: {required_version[0]}.{required_version[1]}+")
        return False

def get_required_packages() -> List[Tuple[str, str]]:
    """Return list of required packages with versions."""
    return [
        ("streamlit", "1.32.0"),
        ("fastapi", "0.109.2"),
        ("uvicorn", "0.27.1"),
        ("langchain", "0.1.14"),
        ("langchain-community", "0.0.34"),
        ("chromadb", "0.4.22"),
        ("sentence-transformers", "2.7.0"),
        ("pypdf2", "3.0.1"),
        ("python-docx", "1.1.0"),
        ("pdfplumber", "0.10.0"),
        ("pymupdf", "1.24.4"),
        ("python-dotenv", "1.0.1"),
        ("requests", "2.31.0"),
        ("groq", "0.9.0"),
    ]

def check_package(package_name: str, required_version: str = None) -> bool:
    """Check if a package is installed and meets version requirements."""
    try:
        module = importlib.import_module(package_name.replace('-', '_'))
        if required_version:
            actual_version = getattr(module, '__version__', 'unknown')
            if actual_version != 'unknown':
                # Simple version comparison (for demo purposes)
                from packaging import version
                if version.parse(actual_version) >= version.parse(required_version):
                    logging.info(f"✅ {package_name}=={actual_version} (>= {required_version})")
                    return True
                else:
                    logging.error(f"❌ {package_name}=={actual_version} (required: >= {required_version})")
                    return False
            else:
                logging.info(f"✅ {package_name} (version unknown)")
                return True
        else:
            logging.info(f"✅ {package_name}")
            return True
    except ImportError:
        logging.error(f"❌ {package_name} not installed")
        return False

def install_package(package_spec: str) -> bool:
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        logging.info(f"✅ Installed {package_spec}")
        return True
    except subprocess.CalledProcessError:
        logging.error(f"❌ Failed to install {package_spec}")
        return False

def check_groq_api_key() -> bool:
    """Check if Groq API key is configured."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logging.warning("⚠️  GROQ_API_KEY not found in .env file")
        logging.info("   Get your API key from: https://console.groq.com")
        logging.info("   Then add it to your .env file:")
        logging.info("   GROQ_API_KEY=your_actual_key_here")
        return False
    
    logging.info("✅ GROQ_API_KEY found in environment")
    return True

def create_directories() -> bool:
    """Create necessary directories for the application."""
    import os
    
    directories = [
        "data/uploads",
        "data/vector_store",
        "logs"
    ]
    
    try:
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"✅ Created directory: {directory}")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to create directories: {str(e)}")
        return False

def main():
    """Main setup function with comprehensive checks."""
    setup_logging()
    
    logging.info("🔧 DocuChat Environment Setup")
    logging.info("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    logging.info("\n📦 Checking required packages...")
    
    # Check and install packages
    required_packages = get_required_packages()
    missing_packages = []
    
    for package_name, required_version in required_packages:
        if not check_package(package_name, required_version):
            missing_packages.append(f"{package_name}>={required_version}")
    
    if missing_packages:
        logging.info(f"\n🔧 Installing {len(missing_packages)} missing packages...")
        for package_spec in missing_packages:
            if not install_package(package_spec):
                logging.error("❌ Setup failed due to package installation errors")
                sys.exit(1)
    
    # 🚨 FIXED: Use the correct function
    logging.info("\n🔑 Checking Groq API key...")
    if not check_groq_api_key():
        logging.warning("⚠️  GROQ_API_KEY not configured")
        logging.info("   Get your API key from: https://console.groq.com")
        logging.info("   Then add it to your .env file as GROQ_API_KEY=your_key")
    
    logging.info("\n📁 Creating directories...")
    if not create_directories():
        logging.error("❌ Failed to create necessary directories")
        sys.exit(1)
    
    logging.info("\n" + "=" * 50)
    logging.info("✅ Setup completed successfully!")
    logging.info("\n🎉 You're ready to run DocuChat!")
    logging.info("\nTo start the application:")
    logging.info("1. Start the backend API:")
    logging.info("   python -m api.main")
    logging.info("2. Start the frontend (in a new terminal):")
    logging.info("   streamlit run frontend/app.py")
    logging.info("\nThe application will be available at:")
    logging.info("   Frontend: http://localhost:8501")
    logging.info("   Backend:  http://localhost:8000")

if __name__ == "__main__":
    main()
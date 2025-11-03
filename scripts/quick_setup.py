"""
Quick setup script to fix dependency issues.
Senior Engineer Principle: Automated dependency resolution.
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command with error handling."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Quick Setup - Fixing Dependency Issues")
    print("=" * 50)
    
    # Fix sentence-transformers compatibility
    commands = [
        ("pip uninstall sentence-transformers huggingface-hub -y", "Uninstalling incompatible versions"),
        ("pip install sentence-transformers==2.7.0", "Installing sentence-transformers"),
        ("pip install huggingface-hub==0.23.2", "Installing huggingface-hub"),
        ("pip install chromadb==0.4.22", "Installing ChromaDB"),
        ("pip install langchain==0.1.14", "Installing LangChain"),
        ("pip install groq==0.9.0", "Installing Groq"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"❌ Setup failed at: {description}")
            return False
    
    print("\n" + "=" * 50)
    print("✅ Quick setup completed successfully!")
    print("\n🎯 Now try starting the backend:")
    print("   python -m api.main")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
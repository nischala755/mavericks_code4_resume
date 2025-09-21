# Resume Relevance Check System - Launch Script

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_files():
    """Check if required files exist"""
    required_files = ["app.py", "requirements.txt"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def create_data_directory():
    """Create data directory if it doesn't exist"""
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
        print("✅ Created data directory")
    else:
        print("✅ Data directory exists")

def main():
    """Main setup and launch function"""
    print("🚀 Resume Relevance Check System - Setup & Launch")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check required files
    if not check_files():
        return
    
    # Create data directory
    create_data_directory()
    
    # Install dependencies
    if not install_dependencies():
        return
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Get your Gemini API key from: https://makersuite.google.com/app/apikey")
    print("2. The application will open in your browser")
    print("3. Enter your API key in the sidebar")
    print("4. Start by adding a job description")
    print("\n🚀 Launching application...")
    
    # Launch Streamlit app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "false"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

if __name__ == "__main__":
    main()
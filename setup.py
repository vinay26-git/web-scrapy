#!/usr/bin/env python3
"""
Setup script for Web Scraping Pipeline
Handles installation and environment setup
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True


def install_spacy_model():
    """Install spaCy model for NLP processing."""
    print("\n🧠 Installing spaCy model...")
    
    try:
        import spacy
        # Try to load the model
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model already installed")
        return True
    except OSError:
        # Model not found, install it
        if run_command("python -m spacy download en_core_web_sm", "Installing spaCy model"):
            print("✅ spaCy model installed successfully")
            return True
        else:
            print("⚠️  spaCy model installation failed, but pipeline will still work with basic chunking")
            return False


def check_gpu():
    """Check for GPU availability."""
    print("\n🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU detected: {gpu_name}")
            print(f"   GPU count: {gpu_count}")
            return True
        else:
            print("⚠️  No CUDA GPU detected, will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, GPU check skipped")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = ["output", "logs", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def test_installation():
    """Test the installation by running a simple example."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        from web_scraper import WebScraper
        from chunking import HybridChunker
        from embeddings import EmbeddingGenerator
        from pipeline import WebScrapingPipeline
        
        print("✅ All modules imported successfully")
        
        # Test basic functionality
        chunker = HybridChunker(max_chunk_size=100, min_chunk_size=50)
        test_text = "This is a test text for chunking. It should be processed correctly."
        chunks = chunker.chunk_text(test_text, domain='generic')
        
        if chunks:
            print(f"✅ Chunking test passed: {len(chunks)} chunks created")
        else:
            print("⚠️  Chunking test failed: no chunks created")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Web Scraping Pipeline Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Install spaCy model
    install_spacy_model()
    
    # Check GPU
    check_gpu()
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📖 Next steps:")
    print("1. Run 'python example.py' to see the pipeline in action")
    print("2. Check the README.md for detailed usage instructions")
    print("3. Modify the configuration in example.py for your needs")
    
    print("\n🔧 Configuration tips:")
    print("- For GPU environments: Use larger batch sizes")
    print("- For memory-constrained environments: Use smaller chunk sizes")
    print("- For high-quality results: Use 'all-mpnet-base-v2' model")
    print("- For fast processing: Use 'all-MiniLM-L6-v2' model")


if __name__ == "__main__":
    main()
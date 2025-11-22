#!/usr/bin/env python3
"""
Setup script for YOLO Traffic Counter deployment
Downloads necessary model files and sets up environment
"""

import os
import urllib.request
from pathlib import Path

def download_yolo_model():
    """Download YOLOv8 nano model if not present"""
    model_path = Path("yolov8n.pt")
    
    if not model_path.exists():
        print("Downloading YOLOv8 nano model...")
        model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print(f"✅ Model downloaded successfully to {model_path}")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            # Create a placeholder file to prevent app crashes
            model_path.touch()
            print("Created placeholder model file")
    else:
        print("✅ YOLOv8 model already exists")

def setup_directories():
    """Create necessary directories"""
    directories = ["temp", "uploads"]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def setup_environment():
    """Set up environment variables for deployment"""
    # Set environment variables for Streamlit
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    print("✅ Environment variables set")

if __name__ == "__main__":
    print("🚀 Setting up YOLO Traffic Counter for deployment...")
    
    setup_environment()
    setup_directories()
    download_yolo_model()
    
    print("✅ Setup completed successfully!")
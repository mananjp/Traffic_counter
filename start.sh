#!/bin/bash

# Netlify startup script for YOLO Traffic Counter
echo "🚀 Starting YOLO Traffic Counter..."

# Set environment variables
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Run setup
echo "📦 Running setup..."
python setup.py

# Start Streamlit
echo "🌟 Starting Streamlit app..."
streamlit run traffic_counter.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
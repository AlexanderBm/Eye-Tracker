#!/bin/bash
# Setup script for Mac Client

echo "=== Setting up Eye Tracker Client on Mac ==="

# 1. Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed."
    echo "Please install Homebrew first: https://brew.sh/"
    exit 1
fi

# 2. Install GStreamer dependencies
echo "--- Installing GStreamer dependencies via Homebrew ---"
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav

# 3. Set up Python Virtual Environment
echo "--- Setting up Python Virtual Environment ---"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created 'venv'"
else
    echo "'venv' already exists"
fi

# 4. Install Python requirements
echo "--- Installing Python dependencies ---"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup Complete! ==="
echo "To run the client:"
echo "1. Activate venv: source venv/bin/activate"
echo "2. Run: ./scripts/run_inference.sh"

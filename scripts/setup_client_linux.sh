#!/bin/bash
# Setup script for Linux Client

echo "=== Setting up Eye Tracker Client on Linux ==="

# 1. Check for apt
if ! command -v apt-get &> /dev/null; then
    echo "Error: apt-get not found. This script is designed for Debian/Ubuntu-based distributions."
    echo "For other distros, install GStreamer and its plugins via your package manager, then re-run with --skip-gstreamer."
    exit 1
fi

SKIP_GSTREAMER=false
for arg in "$@"; do
    if [ "$arg" = "--skip-gstreamer" ]; then
        SKIP_GSTREAMER=true
    fi
done

# 2. Install GStreamer dependencies
if [ "$SKIP_GSTREAMER" = false ]; then
    echo "--- Installing GStreamer dependencies via apt ---"
    sudo apt-get update
    sudo apt-get install -y \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        python3-venv \
        python3-dev
else
    echo "--- Skipping GStreamer installation (--skip-gstreamer) ---"
    sudo apt-get install -y python3-venv python3-dev 2>/dev/null || true
fi

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

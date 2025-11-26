# Setup script for Windows Client
Write-Host "=== Setting up Eye Tracker Client on Windows ==="

# 1. Check for Python
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH."
    Write-Host "Please install Python from https://www.python.org/downloads/"
    exit 1
}

# 2. GStreamer Check (Informational)
Write-Host "--- Checking GStreamer ---"
if (-not (Get-Command "gst-launch-1.0" -ErrorAction SilentlyContinue)) {
    Write-Warning "GStreamer does not appear to be in your PATH."
    Write-Host "Please ensure GStreamer is installed and added to PATH."
    Write-Host "Download: https://gstreamer.freedesktop.org/download/"
    Write-Host "Recommended: Install 'MSVC 64-bit' (Runtime & Development)"
    # We don't exit here because user might have it in a non-standard location or just wants to setup python
} else {
    Write-Host "GStreamer found."
}

# 3. Set up Python Virtual Environment
Write-Host "--- Setting up Python Virtual Environment ---"
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "Created 'venv'"
} else {
    Write-Host "'venv' already exists"
}

# 4. Install Python requirements
Write-Host "--- Installing Python dependencies ---"
# Activate venv in current scope
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "=== Setup Complete! ==="
Write-Host "To run the client:"
Write-Host "1. Activate venv: .\venv\Scripts\Activate.ps1"
Write-Host "2. Run: python src/inference_pipe.py --eye 0"

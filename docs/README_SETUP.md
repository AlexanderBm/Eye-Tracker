# Eye Tracker Setup Guide

This guide explains how to set up the Eye Tracker streaming system from scratch.

## Prerequisites
*   **Mac (Client)**: macOS with Homebrew installed.
*   **Raspberry Pi (Server)**: Running Raspberry Pi OS (Debian based), connected to the same network.
*   **Cameras**: Two Pupil Labs cameras connected to the RPi.

## 1. Client Setup (Mac)
Run the setup script to install GStreamer and Python dependencies:

```bash
```bash
chmod +x scripts/setup_client.sh
./scripts/setup_client.sh
```

This will:
1.  Install GStreamer via Homebrew.
2.  Create a Python virtual environment (`venv`).
3.  Install required Python packages (`opencv-python`, `torch`, etc.).

## 2. Server Setup (Raspberry Pi)
You need to deploy the scripts to the Raspberry Pi and install dependencies.

### Option A: Automatic Deployment
If you know the Pi's IP, user, and password:

```bash
# Usage: ./scripts/deploy_server.sh <RPI_IP> <USER> <PASSWORD>
chmod +x scripts/deploy_server.sh
./scripts/deploy_server.sh 192.168.1.2 kimchi kimchi
```

Then SSH into the Pi and run the setup:
```bash
ssh kimchi@192.168.1.2
./setup_server.sh
```

### Option B: Manual Setup
Copy `start_stream_sw.sh` and `setup_server.sh` to the Pi manually, then run `./setup_server.sh` on the Pi.

## 3. Running the System

### Step 1: Start the Stream (on RPi)
SSH into the Pi and start the stream, pointing it to your Mac's IP address:

```bash
# On Raspberry Pi
./start_stream_sw.sh <YOUR_MAC_IP>
# Example: ./start_stream_sw.sh 192.168.1.1
```

### Step 2: Start the Viewer (on Mac)
Run the inference script on your Mac:

```bash
```bash
# On Mac
./scripts/run_inference.sh
```

This will open two windows showing the camera feeds with inference overlays.
Data (video and CSV) will be saved to `data/collected_data/session_<TIMESTAMP>/`.

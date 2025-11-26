# Eye Tracker Setup Guide

This guide explains how to set up the Eye Tracker streaming system from scratch.

## Prerequisites
*   **Mac (Client)**: macOS with Homebrew installed.
*   **Windows (Client)**: Windows 10/11 with PowerShell.
*   **Raspberry Pi (Server)**: Running Raspberry Pi OS (Debian based), connected to the same network.
*   **Cameras**: Two Pupil Labs cameras connected to the RPi.

## 1. Client Setup

### Option A: Mac Setup
1.  **Run Setup Script**:
    ```bash
    chmod +x scripts/setup_client.sh
    ./scripts/setup_client.sh
    ```
    This installs GStreamer (via Homebrew), creates a Python `venv`, and installs dependencies.

### Option B: Windows Setup
1.  **Install GStreamer**:
    *   Download the **MSVC 64-bit** installer (Runtime & Development) from [gstreamer.freedesktop.org](https://gstreamer.freedesktop.org/download/).
    *   **Important**: Choose "Complete" installation to ensure `gst-launch-1.0` is in your PATH.

2.  **Run Setup Script**:
    Open PowerShell and run:
    ```powershell
    .\scripts\setup_client.ps1
    ```

---

## 2. Server Setup (Raspberry Pi)

You need to deploy the scripts to the Raspberry Pi.

### From Mac
```bash
# Usage: ./scripts/deploy_server.sh <RPI_IP> <USER> <PASSWORD>
chmod +x scripts/deploy_server.sh
./scripts/deploy_server.sh 192.168.1.2 kimchi kimchi
```

### From Windows
```powershell
# Usage: .\scripts\deploy_server.ps1 <RPI_IP> <USER>
.\scripts\deploy_server.ps1 192.168.1.2 kimchi
```
*Note: You will be prompted for the Pi's password twice.*

### Finalize on Server
SSH into the Pi (from either OS) and run:
```bash
./setup_server.sh
```

---

## 3. Running the System

### Step 1: Start Stream (on RPi)
SSH into the Pi and start the stream, pointing it to your computer's IP:
```bash
./start_stream_sw.sh <YOUR_COMPUTER_IP>
```

### Step 2: Start Viewer (Client)

**On Mac**:
```bash
./scripts/run_inference.sh
```

**On Windows**:
```powershell
.\scripts\run_inference.ps1
```

This will open two windows showing the camera feeds with inference overlays.
Data is saved to `data/collected_data/session_<TIMESTAMP>/`.

## 5. Usage Reference

The core inference script is `src/inference_pipe.py`. It can be run manually for advanced usage or debugging.

### Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--eye` | int | **Required** | Eye ID (0 for Left, 1 for Right). |
| `--input_video` | str | None | Path to an MP4 file for offline processing. If omitted, reads from stdin (live stream). |
| `--model` | str | `models/human_conf0.pt` | Path to the model weights file. |
| `--data_dir` | str | `data/collected_data` | Directory where output data (CSV, video) will be saved. |
| `--width` | int | 400 | Width of the input video frame. |
| `--height` | int | 400 | Height of the input video frame. |
| `--flip` | int | None | Flip mode: `0` = vertical, `1` = horizontal. |
| `--title` | str | "Stream" | Window title for the display. |
| `--headless` | flag | False | Run without GUI display. Useful for offline processing or server-side execution. |

### Examples

**1. Live Stream (Standard Usage)**
Usually run via `scripts/run_inference.sh`, but can be run manually if piping from GStreamer:
```bash
# (GStreamer pipeline) | python src/inference_pipe.py --eye 0
```

**2. Offline Video Processing**
Process a pre-recorded video file:
```bash
python src/inference_pipe.py --input_video path/to/video --data_dir path/to/processsed/output --eye 0/1 --headless
```

**3. Custom Model**
Use a different model file:
```bash
python src/inference_pipe.py --model models/new_model.pt
```


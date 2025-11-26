# Run GStreamer streams and pipe to Python inference script on Windows

param (
    [string]$ModelPath = "models/human_conf0.pt"
)

if (-not (Test-Path $ModelPath)) {
    Write-Warning "Model not found at $ModelPath"
}

# Use venv python if available
if (Test-Path "venv\Scripts\python.exe") {
    $PythonCmd = "venv\Scripts\python.exe"
    Write-Host "Using venv python: $PythonCmd"
}
else {
    $PythonCmd = "python"
    Write-Host "Using system python: $PythonCmd"
}

# Check for GStreamer
if (-not (Get-Command "gst-launch-1.0" -ErrorAction SilentlyContinue)) {
    $GstPath = "C:\Program Files\gstreamer\1.0\msvc_x86_64\bin"
    if (Test-Path $GstPath) {
        Write-Host "GStreamer not in PATH. Adding $GstPath"
        $env:PATH = "$GstPath;$env:PATH"
    }
    else {
        Write-Error "GStreamer not found in PATH or at $GstPath. Please install GStreamer."
        exit 1
    }
}

# Create session directory
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SessionDir = "data/collected_data/session_$Timestamp"
New-Item -ItemType Directory -Force -Path $SessionDir | Out-Null
Write-Host "Saving data to: $SessionDir"

# Construct commands
# Note: We use cmd.exe /c to handle the pipe correctly because PowerShell pipes can corrupt binary data
$Cmd1 = "gst-launch-1.0 -q udpsrc port=5000 ! application/x-rtp, payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR,width=400,height=400 ! fdsink | $PythonCmd src/inference_pipe.py --title ""Camera 1 (Inference)"" --model ""$ModelPath"" --eye 0 --data_dir ""$SessionDir"""
$Cmd2 = "gst-launch-1.0 -q udpsrc port=5001 ! application/x-rtp, payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR,width=400,height=400 ! fdsink | $PythonCmd src/inference_pipe.py --title ""Camera 2 (Inference)"" --model ""$ModelPath"" --eye 1 --data_dir ""$SessionDir"""

Write-Host "Starting Inference Stream for Camera 1 (Port 5000)..."
Start-Process cmd.exe -ArgumentList "/c $Cmd1"

Write-Host "Starting Inference Stream for Camera 2 (Port 5001)..."
Start-Process cmd.exe -ArgumentList "/c $Cmd2"

Write-Host "Inference streams started in new windows."

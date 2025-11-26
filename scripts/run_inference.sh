#!/bin/bash
# Run GStreamer streams and pipe to Python inference script

# Default model path
DEFAULT_MODEL="models/human_conf0.pt"
MODEL_PATH="${1:-$DEFAULT_MODEL}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model not found at $MODEL_PATH"
fi

# Use venv python if available
if [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
    echo "Using venv python: $PYTHON_CMD"
else
    PYTHON_CMD="python"
    echo "Using system python: $PYTHON_CMD"
fi

# Create session directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SESSION_DIR="data/collected_data/session_${TIMESTAMP}"
mkdir -p "$SESSION_DIR"
echo "Saving data to: $SESSION_DIR"

echo "Starting Inference Stream for Camera 1 (Port 5000)..."
gst-launch-1.0 -q udpsrc port=5000 ! application/x-rtp, payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR,width=400,height=400 ! fdsink | $PYTHON_CMD src/inference_pipe.py --title "Camera 1 (Inference)" --model "$MODEL_PATH" --eye 0 --data_dir "$SESSION_DIR" &

echo "Starting Inference Stream for Camera 2 (Port 5001)..."
gst-launch-1.0 -q udpsrc port=5001 ! application/x-rtp, payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR,width=400,height=400 ! fdsink | $PYTHON_CMD src/inference_pipe.py --title "Camera 2 (Inference)" --model "$MODEL_PATH" --eye 1 --data_dir "$SESSION_DIR" &

echo "Inference streams started. Press Ctrl+C to stop."
wait

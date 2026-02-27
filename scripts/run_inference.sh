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
gst-launch-1.0 -v v4l2src device=/dev/video2 ! \
  image/jpeg,width=192,height=192 ! \
  jpegdec ! \
  videoconvert ! \
  v4l2h264enc extra-controls="controls,repeat_sequence_header=1" ! \
  'video/x-h264,level=(string)4' ! \
  rtph264pay config-interval=1 pt=96 ! \
  udpsink host=$CLIENT_IP port=5000 sync=false &
  
echo "Starting Inference Stream for Camera 2 (Port 5001)..."
gst-launch-1.0 -v v4l2src device=/dev/video4 ! \
  image/jpeg,width=192,height=192 ! \
  jpegdec ! \
  videoconvert ! \
  v4l2h264enc extra-controls="controls,repeat_sequence_header=1" ! \
  'video/x-h264,level=(string)4' ! \
  rtph264pay config-interval=1 pt=96 ! \
  udpsink host=$CLIENT_IP port=5001 sync=false &
  
echo "Inference streams started. Press Ctrl+C to stop."
wait

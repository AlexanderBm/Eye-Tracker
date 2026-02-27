#!/bin/bash
# Start GStreamer pipelines for two Eye Cameras using Software Encoding
# Usage: ./start_stream_sw.sh <CLIENT_IP>

if [ -z "$1" ]; then
    echo "Usage: $0 <CLIENT_IP>"
    exit 1
fi

CLIENT_IP=$1

# Kill existing gst-launch-1.0 processes
killall gst-launch-1.0 2>/dev/null

# Camera 1 (Eye Left): /dev/video2
echo "Starting Eye Camera 1 (video2) -> $CLIENT_IP:5000"
gst-launch-1.0 -v v4l2src device=/dev/video2 ! \
  image/jpeg,width=192,height=192 ! \
  jpegdec ! \
  videoconvert ! \
  v4l2h264enc extra-controls="controls,repeat_sequence_header=1" ! \
  'video/x-h264,level=(string)4' ! \
  rtph264pay config-interval=1 pt=96 ! \
  udpsink host=$CLIENT_IP port=5000 sync=false &

# Camera 2 (Eye Right): /dev/video4
echo "Starting Eye Camera 2 (video4) -> $CLIENT_IP:5001"
gst-launch-1.0 -v v4l2src device=/dev/video4 ! \
  image/jpeg,width=192,height=192 ! \
  jpegdec ! \
  videoconvert ! \
  v4l2h264enc extra-controls="controls,repeat_sequence_header=1" ! \
  'video/x-h264,level=(string)4' ! \
  rtph264pay config-interval=1 pt=96 ! \
  udpsink host=$CLIENT_IP port=5001 sync=false &

echo "Streams started (Software Encoding). Press Ctrl+C to stop."
wait

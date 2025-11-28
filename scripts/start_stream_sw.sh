#!/bin/bash
# Start GStreamer pipelines for two Eye Cameras
# Streams MJPEG directly (no re-encoding) for maximum performance
# Usage: ./start_stream_sw.sh <CLIENT_IP>

if [ -z "$1" ]; then
    echo "Usage: $0 <CLIENT_IP>"
    exit 1
fi

CLIENT_IP=$1

# Kill existing gst-launch-1.0 processes
killall gst-launch-1.0 2>/dev/null
sleep 1

# Pre-configure cameras to 200fps MJPG mode
echo "Pre-configuring cameras to 200fps mode..."
v4l2-ctl -d /dev/video2 --set-fmt-video=width=192,height=192,pixelformat=MJPG --set-parm=200
v4l2-ctl -d /dev/video4 --set-fmt-video=width=192,height=192,pixelformat=MJPG --set-parm=200

sleep 0.5

# Camera 1 (Eye Left): /dev/video2
# Stream MJPEG directly via RTP - no decode/encode overhead
echo "Starting Eye Camera 1 (video2) -> $CLIENT_IP:5000 @ 192x192 200fps (MJPEG)"
gst-launch-1.0 -v v4l2src device=/dev/video2 ! \
  "image/jpeg,width=192,height=192" ! \
  rtpjpegpay ! \
  udpsink host=$CLIENT_IP port=5000 sync=false &

sleep 1

# Camera 2 (Eye Right): /dev/video4
echo "Starting Eye Camera 2 (video4) -> $CLIENT_IP:5001 @ 192x192 200fps (MJPEG)"
gst-launch-1.0 -v v4l2src device=/dev/video4 ! \
  "image/jpeg,width=192,height=192" ! \
  rtpjpegpay ! \
  udpsink host=$CLIENT_IP port=5001 sync=false &

echo "Streams started (MJPEG Direct @ 200fps). Press Ctrl+C to stop."
wait
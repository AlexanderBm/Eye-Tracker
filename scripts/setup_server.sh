#!/bin/bash
# Setup script for Raspberry Pi Server
# Run this ON the Raspberry Pi

echo "=== Setting up Eye Tracker Server on Raspberry Pi ==="

# 1. Update package list
echo "--- Updating apt ---"
sudo apt-get update

# 2. Install GStreamer
echo "--- Installing GStreamer ---"
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools \
    gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 gstreamer1.0-pulseaudio

echo ""
echo "=== Setup Complete! ==="
echo "To start streaming:"
echo "./start_stream_sw.sh <MAC_IP_ADDRESS>"

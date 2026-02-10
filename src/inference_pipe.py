import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import argparse
import zmq
import msgpack

# Define the model class (copied from video_client.py)
class AzElRegressor128x128(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.backbone = nn.Sequential(
            block(in_ch, 32),
            block(32, 64),
            block(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def preprocess(frame, img_size=128):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    f = gray.astype(np.float32) / 255.0
    m, s = f.mean(), f.std() + 1e-6
    f = (f - m) / s
    x = torch.from_numpy(f).unsqueeze(0)
    return x

import os
import csv
import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--model", type=str, default="models/human_conf0.pt")
    parser.add_argument("--title", type=str, default="Stream")
    parser.add_argument("--flip", type=int, default=None, help="Flip: 0=vertical, 1=horizontal")
    parser.add_argument("--eye", type=int, required=True, help="Eye ID: 0 or 1")
    parser.add_argument("--data_dir", type=str, default="data/collected_data", help="Directory to save data")
    parser.add_argument("--input_video", type=str, default=None, help="Path to input video file (offline mode)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    parser.add_argument("--zmq_pub_port", type=int, default=5555, help="ZMQ PUB port for forwarding eye data (0 to disable)")
    args = parser.parse_args()

    # Setup data collection
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)
    
    # Setup data collection
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)
    
    # Filenames are simple now as they are inside a timestamped directory
    csv_path = os.path.join(args.data_dir, f"data_eye{args.eye}.csv")
    csv_header = ["frame_idx", "timestamp", "az", "el"]
    
    # Create CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    # Initialize VideoWriter
    # Note: MJPG is widely supported. For MP4, 'mp4v' or 'avc1' might be needed.
    # Using 'mp4v' for compatibility.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Assuming 200fps as per GStreamer pipeline
    
    # Raw video writer
    video_path_raw = os.path.join(args.data_dir, f"video_raw_eye{args.eye}.mp4")
    out_video_raw = cv2.VideoWriter(video_path_raw, fourcc, 200.0, (args.width, args.height))
    
    # Overlay video writer
    video_path_overlay = os.path.join(args.data_dir, f"video_overlay_eye{args.eye}.mp4")
    out_video_overlay = cv2.VideoWriter(video_path_overlay, fourcc, 200.0, (args.width, args.height))

    width = args.width
    height = args.height
    frame_size = width * height * 3

    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = AzElRegressor128x128(in_ch=1).to(device)
        # Load weights - handle different saving formats
        try:
            checkpoint = torch.load(args.model, map_location=device)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Proceeding without model...")
            model = None
            
        if model:
            model.eval()
            print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        model = None

    # Setup ZMQ publisher
    zmq_pub = None
    zmq_topic = f"eye.{args.eye}"
    if args.zmq_pub_port > 0:
        zmq_port = args.zmq_pub_port + args.eye  # eye 0 -> 5555, eye 1 -> 5556
        zmq_ctx = zmq.Context()
        zmq_pub = zmq_ctx.socket(zmq.PUB)
        zmq_pub.bind(f"tcp://*:{zmq_port}")
        # JPEG encode params for image compression (quality 80 for speed/size balance)
        zmq_jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        print(f"ZMQ PUB bound on tcp://*:{zmq_port} (topic: {zmq_topic})")

    if not args.input_video:
        print(f"Reading raw video {width}x{height} from stdin...")
    
    # Threaded frame reader to prevent blocking
    import queue
    import threading
    
    frame_queue = queue.Queue(maxsize=1)
    running = True
    
    # Video capture for offline mode
    cap = None
    if args.input_video:
        if not os.path.exists(args.input_video):
            print(f"Error: Input video not found at {args.input_video}")
            return
        cap = cv2.VideoCapture(args.input_video)
        print(f"Processing video file: {args.input_video}")
        
        # Update width/height from video if not specified (optional, but good practice)
        # For now, we stick to args or assume video matches
    
    def read_frames():
        while running:
            try:
                if args.input_video:
                    ret, frame_read = cap.read()
                    if not ret:
                        break
                    # Resize if needed to match expected dimensions
                    if frame_read.shape[1] != width or frame_read.shape[0] != height:
                        frame_read = cv2.resize(frame_read, (width, height))
                    
                    # Simulate real-time if needed, or just process as fast as possible
                    # For offline, we might want to process every frame, so we use a larger queue or blocking put
                    # But to keep logic similar, we'll just put it in the queue
                    
                    # For offline processing, we want to ensure we process EVERY frame, 
                    # so we should probably not drop frames like in live mode.
                    # However, the main loop logic drops frames if queue is full.
                    # Let's modify the queue logic slightly for offline.
                    
                    frame_queue.put(frame_read) # Blocking put for offline to ensure no frame drop? 
                    # If we block here, the main loop must consume fast enough.
                    
                else:
                    raw_data = sys.stdin.buffer.read(frame_size)
                    if not raw_data:
                        break
                    # Only keep the latest frame
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    frame_queue.put(raw_data)
            except Exception:
                break
    
    reader_thread = threading.Thread(target=read_frames, daemon=True)
    reader_thread.start()
    
    # FPS calculation variables
    start_time = time.time()
    fps_frame_count = 0
    global_frame_idx = 0
    fps = 0.0
    
    while True:
        try:
            # Get latest frame with short timeout to check for exit
            # For offline, we might want to wait longer
            timeout = 1.0 if args.input_video else 0.1
            frame_data = frame_queue.get(timeout=timeout)
        except queue.Empty:
            if not reader_thread.is_alive():
                break
            continue
            
        # Convert to numpy array
        if args.input_video:
            frame = frame_data # It's already an image
        else:
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3)).copy()
        
        # Flip if requested
        if args.flip is not None:
            frame = cv2.flip(frame, args.flip)
        
        # Run inference
        if model:
            try:
                x = preprocess(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(x).cpu().numpy()[0]
                
                phi, theta = pred[0], pred[1]
                
                # Save data (Clean frame before overlay)
                if out_video_raw.isOpened():
                    out_video_raw.write(frame)
                
                # Publish via ZMQ (before overlay, so subscriber gets clean image)
                if zmq_pub is not None:
                    _, jpeg_buf = cv2.imencode('.jpg', frame, zmq_jpeg_params)
                    payload = msgpack.dumps({
                        'topic': zmq_topic,
                        'frame_idx': global_frame_idx,
                        'timestamp': time.time(),
                        'az': float(phi),
                        'el': float(theta),
                        'image': jpeg_buf.tobytes(),
                        'width': width,
                        'height': height,
                    }, use_bin_type=True)
                    zmq_pub.send_string(zmq_topic, flags=zmq.SNDMORE)
                    zmq_pub.send(payload)
                
                # Append to CSV
                current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([global_frame_idx, current_time, phi, theta])

                text = f"Az:{phi:+5.1f} El:{theta:+5.1f}"
                # Draw text with a black outline for better visibility
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save overlay frame
                if out_video_overlay.isOpened():
                    out_video_overlay.write(frame)
            except Exception as e:
                pass

        # Calculate and display FPS
        fps_frame_count += 1
        global_frame_idx += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = fps_frame_count / elapsed_time
            fps_frame_count = 0
            start_time = time.time()
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, fps_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Position windows side-by-side
        if not args.headless:
            window_name = args.title
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Position based on camera number
            if "Camera 1" in args.title:
                cv2.moveWindow(window_name, 50, 50)
            elif "Camera 2" in args.title:
                cv2.moveWindow(window_name, 500, 50)
                
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

    running = False
    reader_thread.join(timeout=1.0)
    if cap:
        cap.release()
    if out_video_raw.isOpened():
        out_video_raw.release()
    if out_video_overlay.isOpened():
        out_video_overlay.release()
    if zmq_pub is not None:
        zmq_pub.close()
        zmq_ctx.term()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

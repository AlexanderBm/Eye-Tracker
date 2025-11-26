#!/usr/bin/env python3
"""
Ultra Low-Latency Video Client with Az/El Regression Model
Receives video stream and runs real-time azimuth/elevation prediction
"""

import asyncio
import websockets
import json
import logging
import time
import threading
import queue
import base64
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import argparse
from pathlib import Path
import signal
import sys

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. ML inference disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzElRegressor128x128(nn.Module):
    """Az/El regression model matching the original pipeline.py architecture."""
    
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
            block(in_ch, 32),   # 128 -> 64
            block(32, 64),      # 64  -> 32
            block(64, 128),     # 32  -> 16
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),   # [phi, theta]
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

class MLInferenceEngine:
    """Base class for ML inference engines"""
    
    def __init__(self):
        self.model_loaded = False
        self.processing_time_ms = 0
        
    def load_model(self, model_path: str) -> bool:
        """Load ML model"""
        raise NotImplementedError
        
    def predict(self, frame: np.ndarray) -> dict:
        """Run inference on frame"""
        raise NotImplementedError
        
    def get_info(self) -> dict:
        """Get model information"""
        return {
            "model_loaded": self.model_loaded,
            "processing_time_ms": self.processing_time_ms
        }

class AzElInferenceEngine(MLInferenceEngine):
    """Az/El regression inference engine"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        # Select best available device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Mac GPU acceleration
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")  # NVIDIA GPU
        else:
            self.device = torch.device("cpu")  # CPU fallback
        self.img_size = 128
        
    def load_model(self, model_path: str) -> bool:
        """Load Az/El regression model"""
        if not HAS_TORCH:
            logger.error("PyTorch not available for Az/El inference")
            return False
            
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model
            self.model = AzElRegressor128x128(in_ch=1).to(self.device)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
                
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            self.model_loaded = True
            logger.info(f"Az/El model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Az/El model: {e}")
            return False
    
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for Az/El model (matches pipeline.py)"""
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Resize to 128x128
        gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        
        # Normalize per-image (as in pipeline.py)
        f = gray.astype(np.float32) / 255.0
        m, s = f.mean(), f.std() + 1e-6
        f = (f - m) / s
        
        # Convert to tensor (1, H, W)
        x = torch.from_numpy(f).unsqueeze(0)
        return x
    
    def predict(self, frame: np.ndarray) -> dict:
        """Run Az/El inference"""
        if not self.model_loaded or self.model is None:
            return {"phi_deg": 0.0, "theta_deg": 0.0, "processing_time_ms": 0}
        
        start_time = time.time()
        
        try:
            # Preprocess
            x = self.preprocess(frame)
            
            # Run inference
            with torch.no_grad():
                x = x.unsqueeze(0).to(self.device)  # (1, 1, H, W)
                pred = self.model(x).cpu().numpy()[0]  # [phi, theta]
            
            self.processing_time_ms = (time.time() - start_time) * 1000
            
            return {
                "phi_deg": float(pred[0]),
                "theta_deg": float(pred[1]),
                "processing_time_ms": self.processing_time_ms,
                "model_type": "azel_regressor"
            }
            
        except Exception as e:
            logger.error(f"Az/El inference error: {e}")
            return {"phi_deg": 0.0, "theta_deg": 0.0, "processing_time_ms": 0}

class DummyInferenceEngine(MLInferenceEngine):
    """Dummy inference engine for testing"""
    
    def __init__(self):
        super().__init__()
        self.model_loaded = True
        
    def load_model(self, model_path: str) -> bool:
        """Dummy model loading"""
        self.model_loaded = True
        logger.info("Dummy Az/El inference engine loaded")
        return True
        
    def predict(self, frame: np.ndarray) -> dict:
        """Dummy prediction"""
        start_time = time.time()
        
        # Simulate processing time
        time.sleep(0.005)  # 5ms simulation
        
        # Generate dummy Az/El values
        phi = np.random.uniform(-45, 45)  # Azimuth: -45 to +45 degrees
        theta = np.random.uniform(-30, 30)  # Elevation: -30 to +30 degrees
        
        self.processing_time_ms = (time.time() - start_time) * 1000
        
        return {
            "phi_deg": phi,
            "theta_deg": theta,
            "processing_time_ms": self.processing_time_ms,
            "model_type": "dummy_azel"
        }

class UltraLowLatencyClient:
    """Ultra-optimized client for minimal latency with Az/El inference"""
    
    def __init__(self, server_url: str, scale_factor: int = 3, inference_engine=None):
        self.server_url = server_url
        self.scale_factor = scale_factor
        self.websocket = None
        self.running = False
        
        # ML inference
        self.inference_engine = inference_engine
        self.enable_inference = inference_engine is not None
        self.latest_results = None
        
        # Minimal buffering for ultra-low latency
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.inference_queue = queue.Queue(maxsize=2)  # Minimal queue for inference
        
        # Performance tracking
        self.stats = {
            "frames_received": 0,
            "frames_displayed": 0,
            "frames_processed": 0,
            "connection_time": None,
            "last_frame_time": None,
            "avg_latency_ms": 0,
            "frame_times": [],
            "avg_inference_time_ms": 0
        }
        
        # GUI components
        self.root = None
        self.canvas = None
        self.info_label = None
        self.status_label = None
        self.azel_label = None
        self.display_size = (192 * scale_factor, 192 * scale_factor)
        
    async def connect(self):
        """Connect with minimal timeout"""
        try:
            logger.info(f"Connecting to {self.server_url}")
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=None,  # Disable ping
                ping_timeout=None,
                close_timeout=1,
                max_size=2**20,  # 1MB
                compression=None
            )
            self.stats["connection_time"] = time.time()
            logger.info("Connected to ultra-low latency server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Quick disconnect"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
    
    async def receive_loop(self):
        """Ultra-fast receive loop"""
        logger.info("Starting ultra-low latency receive loop")
        
        while self.running and self.websocket:
            try:
                # Minimal timeout for responsiveness
                message = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
                await self.handle_message(message)
                
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosedError:
                logger.info("Connection closed")
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")
                break
    
    async def handle_message(self, message: str):
        """Handle message with minimal processing"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "config":
                logger.info(f"Config: {data['width']}x{data['height']}@{data['fps']}fps")
                
            elif msg_type == "frame":
                await self.handle_frame(data)
                
        except Exception as e:
            logger.error(f"Message handling error: {e}")
    
    async def handle_frame(self, frame_data: dict):
        """Handle frame with minimal latency"""
        try:
            receive_time = time.time()
            frame_timestamp = frame_data.get("timestamp", receive_time)
            
            # Calculate latency
            latency_ms = (receive_time - frame_timestamp) * 1000
            
            # Update latency tracking
            self.stats["frame_times"].append(latency_ms)
            if len(self.stats["frame_times"]) > 100:  # Keep last 100 frames
                self.stats["frame_times"].pop(0)
            
            self.stats["avg_latency_ms"] = sum(self.stats["frame_times"]) / len(self.stats["frame_times"])
            
            # Decode frame immediately
            encoded_data = frame_data["data"]
            frame_bytes = base64.b64decode(encoded_data)
            
            # Convert to OpenCV format
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return
            
            # Update latest frame with minimal locking
            with self.frame_lock:
                self.latest_frame = frame
                self.stats["frames_received"] += 1
                self.stats["last_frame_time"] = receive_time
            
            # Add to inference queue if enabled
            if self.enable_inference and not self.inference_queue.full():
                try:
                    self.inference_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass  # Skip if queue full
                
        except Exception as e:
            logger.error(f"Frame handling error: {e}")
    
    def inference_loop(self):
        """Az/El inference processing loop"""
        if not self.enable_inference:
            return
            
        logger.info("Starting Az/El inference loop")
        
        while self.running:
            try:
                # Get frame from queue
                try:
                    frame = self.inference_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Run Az/El inference
                results = self.inference_engine.predict(frame)
                self.latest_results = results
                self.stats["frames_processed"] += 1
                
                # Update inference timing stats
                if "processing_time_ms" in results:
                    self.stats["avg_inference_time_ms"] = results["processing_time_ms"]
                
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                time.sleep(0.1)
        
        logger.info("Inference loop ended")
    
    def create_gui(self):
        """Create minimal GUI for ultra-low latency display with Az/El info"""
        self.root = tk.Tk()
        self.root.title("Ultra Low-Latency Stream - 192x192 with Az/El Tracking")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.geometry(f"{self.display_size[0] + 40}x{self.display_size[1] + 160}")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video display - scaled up for visibility
        self.canvas = tk.Canvas(
            main_frame, 
            width=self.display_size[0], 
            height=self.display_size[1], 
            bg="black"
        )
        self.canvas.grid(row=0, column=0, columnspan=2, pady=(0, 5))
        
        # Az/El display
        self.azel_label = ttk.Label(main_frame, text="Az/El: --- / ---", font=("Arial", 12, "bold"))
        self.azel_label.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        # Status
        self.status_label = ttk.Label(main_frame, text="Status: Disconnected")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W)
        
        # Performance info
        self.info_label = ttk.Label(main_frame, text="")
        self.info_label.grid(row=3, column=0, columnspan=2, sticky=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(5, 0))
        
        self.connect_button = ttk.Button(button_frame, text="Connect", command=self.on_connect)
        self.connect_button.grid(row=0, column=0, padx=(0, 5))
        
        self.disconnect_button = ttk.Button(button_frame, text="Disconnect", command=self.on_disconnect, state="disabled")
        self.disconnect_button.grid(row=0, column=1)
    
    def update_display(self):
        """Ultra-fast display update with Az/El overlay"""
        if not self.root:
            return
        
        try:
            # Get latest frame
            with self.frame_lock:
                current_frame = self.latest_frame
            
            if current_frame is not None:
                frame_display = current_frame.copy()
                
                # Draw Az/El crosshair if we have results
                if self.latest_results and self.enable_inference:
                    self.draw_azel_overlay(frame_display, self.latest_results)
                
                # Scale up frame for visibility (nearest neighbor for speed)
                scaled_frame = cv2.resize(
                    frame_display, 
                    self.display_size, 
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL and display
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo  # Keep reference
                
                self.stats["frames_displayed"] += 1
            
            # Update Az/El display
            if self.latest_results and self.enable_inference:
                phi = self.latest_results.get("phi_deg", 0)
                theta = self.latest_results.get("theta_deg", 0)
                self.azel_label.config(text=f"Az: {phi:+6.2f}°   El: {theta:+6.2f}°")
            else:
                self.azel_label.config(text="Az/El: No Model")
            
            # Update status
            if self.websocket:
                self.status_label.config(text="Status: Connected")
                self.connect_button.config(state="disabled")
                self.disconnect_button.config(state="normal")
            else:
                self.status_label.config(text="Status: Disconnected")
                self.connect_button.config(state="normal")
                self.disconnect_button.config(state="disabled")
            
            # Update performance info
            info_lines = [
                f"Received: {self.stats['frames_received']} frames",
                f"Displayed: {self.stats['frames_displayed']} frames",
                f"Latency: {self.stats['avg_latency_ms']:.1f}ms"
            ]
            
            if self.enable_inference:
                info_lines.append(f"Processed: {self.stats['frames_processed']} frames")
                info_lines.append(f"Inference: {self.stats['avg_inference_time_ms']:.1f}ms")
            
            if self.stats["last_frame_time"]:
                time_since_last = time.time() - self.stats["last_frame_time"]
                info_lines.append(f"Last frame: {time_since_last:.2f}s ago")
            
            self.info_label.config(text="\n".join(info_lines))
            
        except Exception as e:
            logger.error(f"Display update error: {e}")
        
        # High-frequency updates for minimal latency
        if self.root:
            self.root.after(16, self.update_display)  # ~60 FPS
    
    def draw_azel_overlay(self, frame: np.ndarray, results: dict):
        """Draw Az/El visualization overlay on frame"""
        if not results or "phi_deg" not in results or "theta_deg" not in results:
            return
        
        phi = results["phi_deg"]  # Azimuth
        theta = results["theta_deg"]  # Elevation
        
        # Draw only the text overlay with phi, theta values
        text = f"Az:{phi:+5.1f} El:{theta:+5.1f}"
        cv2.putText(frame, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def on_connect(self):
        """Handle connect button"""
        def run_connect():
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.connect_and_start())
                else:
                    loop.run_until_complete(self.connect_and_start())
            except RuntimeError:
                # No event loop running, create one
                asyncio.run(self.connect_and_start())
        
        # Run in separate thread to avoid blocking Tkinter
        threading.Thread(target=run_connect, daemon=True).start()
    
    def on_disconnect(self):
        """Handle disconnect button"""
        def run_disconnect():
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.stop_and_disconnect())
                else:
                    loop.run_until_complete(self.stop_and_disconnect())
            except RuntimeError:
                # No event loop running, create one
                asyncio.run(self.stop_and_disconnect())
        
        # Run in separate thread to avoid blocking Tkinter
        threading.Thread(target=run_disconnect, daemon=True).start()
    
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        if self.root:
            self.root.quit()
    
    async def connect_and_start(self):
        """Connect and start receiving with inference"""
        if await self.connect():
            self.running = True
            
            # Start inference thread if enabled
            if self.enable_inference:
                inference_thread = threading.Thread(target=self.inference_loop)
                inference_thread.daemon = True
                inference_thread.start()
            
            # Start receive loop
            await self.receive_loop()
    
    async def stop_and_disconnect(self):
        """Stop streaming and disconnect"""
        self.running = False
        await self.disconnect()
    
    def run_gui(self):
        """Run the GUI main loop"""
        self.create_gui()
        self.update_display()
        self.root.mainloop()

def create_inference_engine(engine_type: str, model_path: str = None) -> MLInferenceEngine:
    """Create inference engine based on type"""
    if engine_type == "azel" and HAS_TORCH:
        engine = AzElInferenceEngine()
        if model_path and engine.load_model(model_path):
            return engine
        else:
            logger.warning("Failed to load Az/El model, using dummy engine")
            return DummyInferenceEngine()
    elif engine_type == "dummy":
        return DummyInferenceEngine()
    else:
        logger.warning(f"Unknown engine type: {engine_type}, using dummy")
        return DummyInferenceEngine()
    
    def on_connect(self):
        """Handle connect button"""
        asyncio.create_task(self.connect_and_start())
    
    def on_disconnect(self):
        """Handle disconnect button"""
        asyncio.create_task(self.stop_and_disconnect())
    
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        if self.root:
            self.root.quit()
    
    async def connect_and_start(self):
        """Connect and start receiving"""
        if await self.connect():
            self.running = True
            await self.receive_loop()
    
    async def stop_and_disconnect(self):
        """Stop and disconnect"""
        self.running = False
        await self.disconnect()
    
    def run_gui(self):
        """Run GUI with minimal overhead"""
        self.create_gui()
        self.update_display()
        self.root.mainloop()

async def run_headless(client):
    """Run client in headless mode for performance testing"""
    logger.info("Running in headless mode with Az/El inference")
    
    if not await client.connect():
        return
    
    client.running = True
    
    # Start inference thread if enabled
    if client.enable_inference:
        inference_thread = threading.Thread(target=client.inference_loop)
        inference_thread.daemon = True
        inference_thread.start()
    
    # Performance monitoring
    start_time = time.time()
    last_report = start_time
    
    try:
        while client.running:
            await client.receive_loop()
            
            # Report performance every 10 seconds
            current_time = time.time()
            if current_time - last_report >= 10.0:
                uptime = current_time - start_time
                fps = client.stats["frames_received"] / uptime if uptime > 0 else 0
                
                info_msg = f"Performance: {fps:.1f} FPS avg, {client.stats['avg_latency_ms']:.1f}ms avg latency"
                if client.enable_inference:
                    inf_fps = client.stats["frames_processed"] / uptime if uptime > 0 else 0
                    info_msg += f", {inf_fps:.1f} inference FPS"
                    if client.latest_results:
                        phi = client.latest_results.get("phi_deg", 0)
                        theta = client.latest_results.get("theta_deg", 0)
                        info_msg += f", Az/El: {phi:+6.2f}°/{theta:+6.2f}°"
                
                logger.info(info_msg)
                last_report = current_time
                
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        await client.disconnect()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Ultra Low-Latency Video Client with Az/El Inference")
    parser.add_argument("--server-url", default="ws://192.168.1.172:8765", help="Server URL")
    parser.add_argument("--scale", type=int, default=3, help="Display scale factor")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--enable-inference", action="store_true", default=True, help="Enable Az/El inference")
    parser.add_argument("--inference-engine", choices=["azel", "dummy"], default="azel", help="Inference engine type")
    parser.add_argument("--model-path", type=str, default="models/human_conf0.pt", help="Path to Az/El model file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create inference engine
    inference_engine = None
    if args.enable_inference:
        inference_engine = create_inference_engine(args.inference_engine, args.model_path)
    
    # Create client
    client = UltraLowLatencyClient(args.server_url, args.scale, inference_engine)
    
    # Signal handling
    def signal_handler(sig, frame):
        logger.info("Interrupted")
        client.running = False
        if client.root:
            client.root.quit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if args.headless:
            asyncio.run(run_headless(client))
        else:
            # Set up event loop for GUI mode
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run GUI
            client.run_gui()
            
    except KeyboardInterrupt:
        logger.info("Application interrupted")
    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
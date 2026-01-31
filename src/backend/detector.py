"""
Real-time Object Detection using YOLOv11 and MSS Screen Capture
For Rust player detection
"""
import sys
import json
import time
import mss
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import io
import base64

class RustPlayerDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize the detector with a YOLO model
        
        Args:
            model_path: Path to the .pt model file
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.sct = mss.mss()
        self.is_running = False
        self.current_detections = []
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            # Check if file is valid PyTorch model
            import os
            file_size = os.path.getsize(self.model_path)
            
            # Real PyTorch models are typically > 1MB, JSON files are much smaller
            if file_size < 1000000:  # Less than 1MB
                # Check if it's JSON
                with open(self.model_path, 'rb') as f:
                    first_bytes = f.read(10)
                    if first_bytes.startswith(b'{'):
                        return {
                            "status": "error", 
                            "message": f"File appears to be JSON metadata (size: {file_size} bytes). Please provide the actual PyTorch .pt model file from your YOLO training (typically 10-50MB+)."
                        }
            
            self.model = YOLO(self.model_path)
            return {"status": "success", "message": "Model loaded successfully"}
        except Exception as e:
            error_msg = str(e)
            if "invalid load key" in error_msg and "'{'" in error_msg:
                return {
                    "status": "error", 
                    "message": "File is JSON, not a PyTorch model. Please use the actual .pt file from your YOLO training (typically 10-50MB+)."
                }
            return {"status": "error", "message": f"Failed to load model: {error_msg}"}
    
    def capture_screen(self, monitor=None):
        """
        Capture screen using MSS
        
        Args:
            monitor: Monitor region dict or None for primary monitor
        Returns:
            PIL Image
        """
        if monitor is None:
            monitor = self.sct.monitors[1]  # Primary monitor
        
        screenshot = self.sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return img
    
    def detect(self, image=None, monitor=None):
        """
        Run detection on screen capture or provided image
        
        Args:
            image: PIL Image (optional, if None captures screen)
            monitor: Monitor region dict (optional)
        Returns:
            List of detections with format:
            [{"class": class_name, "confidence": float, "bbox": [x1, y1, x2, y2]}, ...]
        """
        if self.model is None:
            return {"status": "error", "message": "Model not loaded"}
        
        try:
            # Capture screen if no image provided
            if image is None:
                image = self.capture_screen(monitor)
            
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    detections.append({
                        "class": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class_id": class_id
                    })
            
            self.current_detections = detections
            return {
                "status": "success",
                "detections": detections,
                "count": len(detections)
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Detection failed: {str(e)}"}
    
    def start_continuous_detection(self, monitor=None, fps=30):
        """
        Start continuous detection loop
        
        Args:
            monitor: Monitor region dict (optional)
            fps: Target frames per second
        """
        self.is_running = True
        frame_time = 1.0 / fps
        
        while self.is_running:
            start_time = time.time()
            
            result = self.detect(monitor=monitor)
            
            # Output result as JSON for Electron to read
            print(json.dumps(result))
            sys.stdout.flush()
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def stop_detection(self):
        """Stop continuous detection"""
        self.is_running = False
    
    def set_confidence(self, threshold):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        return {"status": "success", "confidence": self.confidence_threshold}

def main():
    """Main function for command-line interface"""
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Model path required"}))
        sys.exit(1)
    
    model_path = sys.argv[1]
    detector = RustPlayerDetector(model_path)
    
    # Load model
    load_result = detector.load_model()
    print(json.dumps(load_result))
    sys.stdout.flush()
    
    if load_result["status"] != "success":
        sys.exit(1)
    
    # Read commands from stdin
    try:
        for line in sys.stdin:
            command = json.loads(line.strip())
            cmd = command.get("command")
            
            if cmd == "detect":
                result = detector.detect()
                print(json.dumps(result))
                sys.stdout.flush()
            
            elif cmd == "start":
                monitor = command.get("monitor")
                fps = command.get("fps", 30)
                detector.start_continuous_detection(monitor=monitor, fps=fps)
            
            elif cmd == "stop":
                detector.stop_detection()
                print(json.dumps({"status": "success", "message": "Detection stopped"}))
                sys.stdout.flush()
            
            elif cmd == "set_confidence":
                threshold = command.get("threshold", 0.5)
                result = detector.set_confidence(threshold)
                print(json.dumps(result))
                sys.stdout.flush()
            
            elif cmd == "exit":
                detector.stop_detection()
                break
                
    except KeyboardInterrupt:
        detector.stop_detection()
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.stdout.flush()

if __name__ == "__main__":
    main()

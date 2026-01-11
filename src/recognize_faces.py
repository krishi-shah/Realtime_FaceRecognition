"""
Real-Time Face Recognition Module (OpenCV LBPH Version)
Performs live face recognition using webcam feed
"""

import cv2
import time
import os
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime

from . import config
from .encode_faces import FaceTrainer
from .capture_training_images import FaceDetector


class FaceRecognizer:
    """Real-time face recognition from webcam feed"""
    
    def __init__(self):
        self.trainer = FaceTrainer()
        self.detector = None
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
        
        # Current detection results
        self.current_face_data = []
        
    def initialize(self) -> bool:
        """Initialize the recognizer by loading the model"""
        # Initialize face detector
        print("\n  Initializing face detector...")
        self.detector = FaceDetector()
        
        # Load trained model
        return self.trainer.load_model()
    
    def start_recognition(self) -> None:
        """Start the real-time face recognition loop"""
        
        if not self.trainer.label_to_name:
            print("\n[X] No trained model loaded. Please train the model first.")
            return
            
        # Initialize webcam with robust backend selection
        print(f"\n  Opening camera (index {config.CAMERA_INDEX})...")
        
        # Try different backends in order of preference for Windows
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),  # More reliable on Windows
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto-detect")
        ]
        
        camera_opened = False
        for backend, backend_name in backends:
            print(f"  Trying {backend_name} backend...")
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
            
            if not self.cap.isOpened():
                print(f"  [X] {backend_name} failed to open camera")
                continue
            
            # Set camera properties before testing
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            
            # Test multiple frame reads to ensure camera is actually working
            test_successful = False
            try:
                for i in range(5):  # Try reading 5 frames
                    ret, test_frame = self.cap.read()
                    if not ret or test_frame is None:
                        print(f"  [!] {backend_name}: Failed to read frame {i+1}/5")
                        break
                    
                    # Validate frame has valid dimensions
                    if len(test_frame.shape) != 3 or test_frame.shape[0] == 0 or test_frame.shape[1] == 0:
                        print(f"  [!] {backend_name}: Invalid frame dimensions")
                        break
                    
                    # If we got here on the last iteration, camera is working
                    if i == 4:
                        test_successful = True
                        print(f"  [OK] Camera opened with {backend_name}")
                        print(f"  [OK] Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
                        camera_opened = True
                        break
                
                if test_successful:
                    break
                else:
                    print(f"  [!] {backend_name} opened camera but cannot reliably read frames")
                    self.cap.release()
                    
            except cv2.error as e:
                print(f"  [X] {backend_name} error: {str(e)[:50]}...")
                self.cap.release()
                continue
        
        if not camera_opened:
            print("\n  [X] Could not open camera with any backend")
            print("\n  Please check:")
            print("      1. Close the Windows Camera app if it's open")
            print("      2. Close any video call apps (Teams, Zoom, Discord, etc.)")
            print("      3. Check Windows Settings > Privacy > Camera")
            print("         - Enable 'Camera access'")
            print("         - Enable 'Let desktop apps access your camera'")
            print("      4. Restart your computer if the issue persists")
            return
        
        print("\n" + "="*60)
        print("  REAL-TIME FACE RECOGNITION")
        print("="*60)
        print(f"\n  Model: LBPH Face Recognizer")
        print(f"  Threshold: {config.RECOGNITION_THRESHOLD}")
        print(f"  Known persons: {', '.join(self.trainer.label_to_name.values())}")
        print(f"\n  Controls:")
        print(f"    [Q] or [ESC] - Quit")
        print(f"    [+] - Increase threshold (more lenient)")
        print(f"    [-] - Decrease threshold (stricter)")
        print(f"    [S] - Take screenshot")
        print("="*60 + "\n")
        
        self.start_time = time.time()
        self.frame_count = 0
        threshold = config.RECOGNITION_THRESHOLD
        frame_error_count = 0
        max_consecutive_errors = 10
        
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    frame_error_count += 1
                    print(f"Warning: Failed to read frame ({frame_error_count}/{max_consecutive_errors})")
                    if frame_error_count >= max_consecutive_errors:
                        print("\n[X] Too many consecutive frame read errors. Camera may have disconnected.")
                        break
                    time.sleep(0.1)
                    continue
                
                # Reset error count on successful read
                frame_error_count = 0
                
            except cv2.error as e:
                print(f"\n[X] Camera error: {str(e)[:100]}")
                break
            
            self.frame_count += 1
            
            # Process face detection and recognition
            self._process_frame(frame, threshold)
            
            # Draw results on frame
            display_frame = self._draw_results(frame, threshold)
            
            # Show frame
            cv2.imshow(config.WINDOW_NAME, display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                break
            elif key == ord('+') or key == ord('='):
                threshold = min(200, threshold + 5)
                print(f"  Threshold increased to: {threshold}")
            elif key == ord('-') or key == ord('_'):
                threshold = max(10, threshold - 5)
                print(f"  Threshold decreased to: {threshold}")
            elif key == ord('s') or key == ord('S'):
                self._save_screenshot(display_frame)
        
        self._cleanup()
        
    def _process_frame(self, frame: np.ndarray, threshold: float) -> None:
        """Process a frame for face detection and recognition"""
        self.current_face_data = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector.detect(frame)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_gray = gray[y:y+h, x:x+w]
            
            try:
                # Resize face for recognition
                face_resized = cv2.resize(face_gray, config.FACE_SIZE)
                
                # Recognize face
                label, confidence = self.trainer.recognizer.predict(face_resized)
                
                # In LBPH, lower confidence = better match
                if confidence <= threshold:
                    name = self.trainer.label_to_name.get(label, "Unknown")
                    # Convert confidence to percentage (inverse relationship)
                    conf_percent = max(0, (threshold - confidence) / threshold)
                else:
                    name = None
                    conf_percent = 0
                
                self.current_face_data.append({
                    'bbox': (x, y, w, h),
                    'name': name,
                    'confidence': conf_percent,
                    'raw_score': confidence
                })
                
            except Exception as e:
                self.current_face_data.append({
                    'bbox': (x, y, w, h),
                    'name': None,
                    'confidence': 0,
                    'raw_score': 999
                })
    
    def _draw_results(self, frame: np.ndarray, threshold: float) -> np.ndarray:
        """Draw recognition results on frame"""
        display_frame = frame.copy()
        
        # Draw face boxes and names
        for face_data in self.current_face_data:
            x, y, w, h = face_data['bbox']
            name = face_data['name']
            confidence = face_data['confidence']
            raw_score = face_data['raw_score']
            
            if name:
                # Recognized face - Green
                color = config.BOX_COLOR
                label = f"{name} ({confidence:.0%})"
            else:
                # Unknown face - Gray
                color = config.BOX_COLOR_UNKNOWN
                label = None
            
            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label if recognized
            if label:
                # Calculate text size
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           config.FONT_SCALE, config.FONT_THICKNESS)[0]
                
                # Draw background rectangle for text
                cv2.rectangle(display_frame, 
                             (x, y + h), 
                             (x + text_size[0] + 10, y + h + text_size[1] + 15),
                             color, -1)
                
                # Draw name label
                cv2.putText(display_frame, label, 
                           (x + 5, y + h + text_size[1] + 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           config.FONT_SCALE, config.TEXT_COLOR, 
                           config.FONT_THICKNESS)
        
        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time
        
        # Status bar at top
        status_height = 70
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], status_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
        
        # Display stats
        cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Threshold: {threshold:.0f}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Recognition status
        recognized_count = sum(1 for f in self.current_face_data if f['name'])
        total_faces = len(self.current_face_data)
        
        status_text = f"Faces: {total_faces} | Recognized: {recognized_count}"
        cv2.putText(display_frame, status_text, 
                   (display_frame.shape[1] - 280, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Model indicator
        cv2.putText(display_frame, "Model: LBPH", 
                   (display_frame.shape[1] - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Instructions
        inst_text = "[Q] Quit | [+/-] Threshold | [S] Screenshot"
        cv2.putText(display_frame, inst_text,
                   (display_frame.shape[1]//2 - 180, display_frame.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return display_frame
    
    def _save_screenshot(self, frame: np.ndarray) -> None:
        """Save current frame as screenshot"""
        screenshots_dir = os.path.join(config.BASE_DIR, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        filepath = os.path.join(screenshots_dir, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"  [OK] Screenshot saved: {filename}")
    
    def _cleanup(self) -> None:
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n[OK] Face recognition stopped")


def start_recognition():
    """Start the face recognition system"""
    recognizer = FaceRecognizer()
    
    if recognizer.initialize():
        recognizer.start_recognition()
    else:
        print("\n[X] Failed to initialize face recognizer")
        print("  Please ensure you have captured training images and trained the model.")


if __name__ == "__main__":
    start_recognition()

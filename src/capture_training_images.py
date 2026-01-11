"""
Training Image Capture Module (OpenCV Version)
Captures face images from webcam for training the recognition system
"""

import cv2
import os
import time
from datetime import datetime

from . import config


class FaceDetector:
    """Face detection using OpenCV DNN or Haar Cascade"""
    
    def __init__(self):
        self.use_dnn = config.USE_DNN_DETECTOR
        self.net = None
        self.face_cascade = None
        self._init_detector()
        
    def _init_detector(self):
        """Initialize the face detector"""
        if self.use_dnn:
            try:
                # Try to use OpenCV's DNN face detector
                model_file = cv2.data.haarcascades + "../dnn/opencv_face_detector_uint8.pb"
                config_file = cv2.data.haarcascades + "../dnn/opencv_face_detector.pbtxt"
                
                if os.path.exists(model_file) and os.path.exists(config_file):
                    self.net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                    print("  Using DNN face detector")
                else:
                    self.use_dnn = False
            except Exception:
                self.use_dnn = False
                
        if not self.use_dnn:
            # Fallback to Haar Cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("  Using Haar Cascade face detector")
    
    def detect(self, frame):
        """
        Detect faces in frame
        
        Returns:
            List of (x, y, w, h) tuples for each detected face
        """
        if self.use_dnn and self.net is not None:
            return self._detect_dnn(frame)
        else:
            return self._detect_haar(frame)
    
    def _detect_dnn(self, frame):
        """Detect faces using DNN"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > config.DNN_CONFIDENCE_THRESHOLD:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Convert to (x, y, w, h)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces
    
    def _detect_haar(self, frame):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=config.MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(x, y, w, h) for (x, y, w, h) in faces]


class TrainingImageCapture:
    """Capture training images from webcam with face detection feedback"""
    
    def __init__(self):
        self.cap = None
        self.person_name = None
        self.person_dir = None
        self.captured_count = 0
        self.detector = None
        
    def start_capture(self, person_name: str) -> bool:
        """
        Start the webcam capture session for a specific person
        
        Args:
            person_name: Name of the person (will be used as label)
            
        Returns:
            True if capture was successful, False otherwise
        """
        self.person_name = person_name.strip().replace(" ", "_")
        self.person_dir = os.path.join(config.TRAINING_DIR, self.person_name)
        os.makedirs(self.person_dir, exist_ok=True)
        
        # Initialize face detector
        print("\n  Initializing face detector...")
        self.detector = FaceDetector()
        
        # Count existing images
        existing_images = [f for f in os.listdir(self.person_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.captured_count = len(existing_images)
        
        # Initialize webcam
        print(f"  Opening camera (index {config.CAMERA_INDEX})...")
        
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
            print("      4. Try running: opencv-python-headless if opencv-python doesn't work")
            print("      5. Restart your computer if the issue persists")
            return False
        
        # Create window explicitly to ensure proper focus handling
        print("  Creating display window...")
        try:
            cv2.namedWindow("Training Image Capture", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Training Image Capture", 1280, 720)
            print("  [OK] Display window created")
        except Exception as e:
            print(f"  [X] Error creating window: {e}")
            self.cap.release()
            return False
        
        print(f"\n{'='*60}")
        print(f"  TRAINING IMAGE CAPTURE - {person_name}")
        print(f"{'='*60}")
        print(f"\n  Existing images: {self.captured_count}")
        print(f"  Target: {config.MIN_TRAINING_IMAGES} - {config.MAX_TRAINING_IMAGES} images")
        print(f"\n  Controls:")
        print(f"    [SPACE] - Capture image manually")
        print(f"    [A]     - Toggle auto-capture mode")
        print(f"    [Q]     - Quit and save")
        print(f"    [ESC]   - Cancel without saving")
        print(f"\n  IMPORTANT: Click on the camera window to ensure it has focus!")
        print(f"{'='*60}")
        print(f"\n  Starting capture loop...")
        print(f"  >> Look for the 'Training Image Capture' window on your screen <<\n")
        
        return self._capture_loop()
        
    def _capture_loop(self) -> bool:
        """Main capture loop with face detection feedback"""
        auto_capture = False
        last_capture_time = 0
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
                
            # Create a copy for display
            display_frame = frame.copy()
            
            # Detect faces
            faces = self.detector.detect(frame)
            face_detected = len(faces) > 0
            
            # Draw face rectangles and status
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                color = (0, 255, 0) if len(faces) == 1 else (0, 165, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                
                # Draw face quality indicator
                quality = "Good" if w > 150 and h > 150 else "Move closer"
                cv2.putText(display_frame, quality, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Status bar at top
            status_bg = (40, 40, 40)
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 80), status_bg, -1)
            
            # Display info
            cv2.putText(display_frame, f"Person: {self.person_name}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Images: {self.captured_count}/{config.MAX_TRAINING_IMAGES}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Auto-capture indicator
            auto_text = "AUTO: ON" if auto_capture else "AUTO: OFF"
            auto_color = (0, 255, 0) if auto_capture else (128, 128, 128)
            cv2.putText(display_frame, auto_text, (display_frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, auto_color, 2)
            
            # Face detection status message
            msg_y = display_frame.shape[0] - 30
            if not face_detected:
                cv2.putText(display_frame, "No face detected - Please look at camera", 
                           (display_frame.shape[1]//2 - 250, msg_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif len(faces) > 1:
                cv2.putText(display_frame, "Multiple faces detected - Only one person please",
                           (display_frame.shape[1]//2 - 280, msg_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(display_frame, "Face detected! Press SPACE to capture",
                           (display_frame.shape[1]//2 - 220, msg_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Instructions bar
            inst_y = display_frame.shape[0] - 60
            cv2.rectangle(display_frame, (0, inst_y - 10), (display_frame.shape[1], inst_y + 25), status_bg, -1)
            cv2.putText(display_frame, "[SPACE] Capture  |  [A] Auto-capture  |  [Q] Save & Quit  |  [ESC] Cancel",
                       (20, inst_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            
            # Auto-capture logic
            current_time = time.time()
            if auto_capture and face_detected and len(faces) == 1:
                if current_time - last_capture_time >= config.CAPTURE_INTERVAL:
                    if self.captured_count < config.MAX_TRAINING_IMAGES:
                        self._save_face_image(frame, faces[0])
                        last_capture_time = current_time
            
            # Show frame
            cv2.imshow("Training Image Capture", display_frame)
            
            # Handle key presses - use longer delay for better key capture on Windows
            # waitKey must be called after imshow, and delay should be > 0 for GUI
            key_code = cv2.waitKey(30)
            
            # Check if window is closed (X button)
            try:
                if cv2.getWindowProperty("Training Image Capture", cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[X] Window closed - Capture cancelled")
                    self._cleanup()
                    return False
            except cv2.error:
                # Window might have been destroyed
                print("\n[X] Window closed - Capture cancelled")
                self._cleanup()
                return False
            
            # Extract key code (handle both normal and extended keys)
            if key_code == -1:
                continue  # No key pressed, continue loop
            
            key = key_code & 0xFF
            
            if key == ord(' '):  # Space - manual capture
                if face_detected and len(faces) == 1:
                    self._save_face_image(frame, faces[0])
                else:
                    print("Cannot capture: Need exactly one face in frame")
                    
            elif key == ord('a') or key == ord('A'):  # Toggle auto-capture
                auto_capture = not auto_capture
                print(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")
                
            elif key == ord('q') or key == ord('Q'):  # Quit and save
                if self.captured_count >= config.MIN_TRAINING_IMAGES:
                    print(f"\n[OK] Captured {self.captured_count} images for '{self.person_name}'")
                    break
                else:
                    print(f"\n[!] Need at least {config.MIN_TRAINING_IMAGES} images. Currently have {self.captured_count}.")
                    print("Press Q again to quit anyway, or continue capturing.")
                    # Wait a bit to avoid immediate re-trigger
                    time.sleep(0.3)
                    
            elif key == 27:  # ESC - cancel
                print("\n[X] Capture cancelled")
                self._cleanup()
                return False
                
            # Check if we have enough images
            if self.captured_count >= config.MAX_TRAINING_IMAGES:
                print(f"\n[OK] Reached maximum images ({config.MAX_TRAINING_IMAGES})")
                break
        
        self._cleanup()
        return self.captured_count >= config.MIN_TRAINING_IMAGES
    
    def _save_face_image(self, frame, face_rect):
        """Save a captured face image"""
        x, y, w, h = face_rect
        
        # Add padding around the face
        height, width = frame.shape[:2]
        padding = 50
        y1 = max(0, y - padding)
        x1 = max(0, x - padding)
        y2 = min(height, y + h + padding)
        x2 = min(width, x + w + padding)
        
        # Crop face region with padding
        face_image = frame[y1:y2, x1:x2]
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.person_name}_{timestamp}.jpg"
        filepath = os.path.join(self.person_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.captured_count += 1
        print(f"  [OK] Captured image {self.captured_count}: {filename}")
        
    def _cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def capture_training_images():
    """Interactive function to capture training images"""
    print("\n" + "="*60)
    print("  FACE RECOGNITION - TRAINING IMAGE CAPTURE")
    print("="*60)
    
    # Get person name
    person_name = input("\nEnter the name of the person to train: ").strip()
    
    if not person_name:
        print("Error: Name cannot be empty")
        return False
        
    # Start capture
    capturer = TrainingImageCapture()
    success = capturer.start_capture(person_name)
    
    if success:
        print(f"\n{'='*60}")
        print(f"  Training images saved successfully!")
        print(f"  Location: {os.path.join(config.TRAINING_DIR, person_name.replace(' ', '_'))}")
        print(f"  Next step: Run 'python main.py --train' to train the model")
        print(f"{'='*60}\n")
    
    return success


if __name__ == "__main__":
    capture_training_images()

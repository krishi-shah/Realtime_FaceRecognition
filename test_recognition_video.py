#!/usr/bin/env python3
"""
Test face recognition using a video file or image (camera workaround)
"""

import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.encode_faces import FaceTrainer
from src.capture_training_images import FaceDetector
from src import config

def test_recognition_with_image():
    """Test recognition using training images themselves"""
    
    print("\n" + "="*60)
    print("  TESTING FACE RECOGNITION (Image Mode)")
    print("="*60)
    
    # Load the trained model
    print("\n1. Loading trained model...")
    trainer = FaceTrainer()
    if not trainer.load_model():
        print("   [X] No trained model found!")
        print("   Run step 2 (Train Model) first")
        return False
    
    stats = trainer.get_stats()
    print(f"   [OK] Model loaded - Trained for: {', '.join(stats['persons'])}")
    
    # Initialize face detector
    print("\n2. Initializing face detector...")
    detector = FaceDetector()
    
    # Get recognizer
    recognizer = trainer.recognizer
    label_to_name = trainer.label_to_name
    
    # Test on training images
    print("\n3. Testing recognition on training images...")
    print("="*60)
    
    training_dir = config.TRAINING_DIR
    test_count = 0
    correct_count = 0
    
    for person_name in os.listdir(training_dir):
        person_path = os.path.join(training_dir, person_name)
        if not os.path.isdir(person_path):
            continue
            
        images = [f for f in os.listdir(person_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\nTesting images for: {person_name}")
        print("-" * 40)
        
        # Test first 3 images for each person
        for img_file in images[:3]:
            img_path = os.path.join(person_path, img_file)
            frame = cv2.imread(img_path)
            
            if frame is None:
                continue
            
            # Detect faces
            faces = detector.detect(frame)
            
            if len(faces) == 0:
                print(f"  {img_file}: No face detected")
                continue
            
            # Test recognition on first face
            (x, y, w, h) = faces[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, config.FACE_SIZE)
            
            # Recognize
            label, confidence = recognizer.predict(face_resized)
            predicted_name = label_to_name.get(label, "Unknown")
            
            test_count += 1
            is_correct = predicted_name.upper() == person_name.upper()
            if is_correct:
                correct_count += 1
            
            status = "[OK]" if is_correct else "[X]"
            print(f"  {status} {img_file}")
            print(f"      Expected: {person_name}, Got: {predicted_name}, Confidence: {confidence:.1f}")
            
            # Show the image
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_name} ({confidence:.0f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Try to display (might not work if camera doesn't work)
            try:
                cv2.imshow("Recognition Test", frame)
                cv2.waitKey(800)  # Show for 800ms
            except:
                pass  # Silently fail if window doesn't work
    
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*60)
    print("  RECOGNITION TEST RESULTS")
    print("="*60)
    print(f"  Total tests: {test_count}")
    print(f"  Correct: {correct_count}")
    print(f"  Accuracy: {(correct_count/test_count*100):.1f}%" if test_count > 0 else "  No tests run")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_recognition_with_image()

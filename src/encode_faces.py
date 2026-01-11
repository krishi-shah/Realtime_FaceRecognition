"""
Face Training Module (OpenCV LBPH Version)
Trains the face recognition model using captured images
"""

import os
import pickle
import cv2
import numpy as np
from typing import Dict, List, Tuple

from . import config


class FaceTrainer:
    """Train the LBPH face recognizer with captured images"""
    
    def __init__(self):
        # Create LBPH Face Recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.label_to_name: Dict[int, str] = {}
        self.name_to_label: Dict[str, int] = {}
        
    def train(self) -> Tuple[int, int]:
        """
        Train the face recognizer with all training images
        
        Returns:
            Tuple of (successful_images, failed_images)
        """
        print("\n" + "="*60)
        print("  TRAINING FACE RECOGNITION MODEL")
        print("="*60)
        print("\n  Using: LBPH (Local Binary Pattern Histogram)")
        
        faces = []
        labels = []
        success_count = 0
        fail_count = 0
        
        # Check training directory
        if not os.path.exists(config.TRAINING_DIR):
            print(f"\n[X] Training directory not found: {config.TRAINING_DIR}")
            print("  Please run training image capture first.")
            return 0, 0
            
        # Get all person directories
        person_dirs = [d for d in os.listdir(config.TRAINING_DIR) 
                      if os.path.isdir(os.path.join(config.TRAINING_DIR, d))]
        
        if not person_dirs:
            print(f"\n[X] No training data found in: {config.TRAINING_DIR}")
            print("  Please run training image capture first.")
            return 0, 0
            
        print(f"\n  Found {len(person_dirs)} person(s) to train:")
        
        # Assign numeric labels to each person
        for idx, person_name in enumerate(sorted(person_dirs)):
            self.label_to_name[idx] = person_name.replace("_", " ")
            self.name_to_label[person_name] = idx
            
        # Process each person's images
        for person_name in person_dirs:
            person_path = os.path.join(config.TRAINING_DIR, person_name)
            label = self.name_to_label[person_name]
            
            # Get all images
            image_files = [f for f in os.listdir(person_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"\n  Processing '{person_name}' ({len(image_files)} images)...")
            
            person_success = 0
            person_fail = 0
            
            for image_file in image_files:
                image_path = os.path.join(person_path, image_file)
                
                try:
                    # Load image in grayscale
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"    [!] Could not load: {image_file}")
                        person_fail += 1
                        continue
                        
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Detect face in image
                    detected_faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(50, 50)
                    )
                    
                    if len(detected_faces) == 0:
                        # If no face detected, use the whole image (it's already cropped)
                        face_gray = cv2.resize(gray, config.FACE_SIZE)
                        faces.append(face_gray)
                        labels.append(label)
                        person_success += 1
                        print(f"    [OK] Processed: {image_file}")
                    else:
                        # Use the first detected face
                        (x, y, w, h) = detected_faces[0]
                        face_gray = gray[y:y+h, x:x+w]
                        face_gray = cv2.resize(face_gray, config.FACE_SIZE)
                        faces.append(face_gray)
                        labels.append(label)
                        person_success += 1
                        print(f"    [OK] Processed: {image_file}")
                        
                except Exception as e:
                    print(f"    [X] Error processing {image_file}: {str(e)[:50]}")
                    person_fail += 1
            
            success_count += person_success
            fail_count += person_fail
            print(f"    Summary: {person_success} successful, {person_fail} failed")
        
        # Train the recognizer
        if len(faces) > 0:
            print(f"\n  Training model with {len(faces)} face images...")
            self.recognizer.train(faces, np.array(labels))
            print("  [OK] Model trained successfully!")
        else:
            print("\n  [X] No faces to train with")
            
        return success_count, fail_count
    
    def save_model(self) -> bool:
        """Save the trained model and labels to disk"""
        try:
            # Save the recognizer model
            self.recognizer.save(config.MODEL_FILE)
            
            # Save the label mappings
            labels_data = {
                "label_to_name": self.label_to_name,
                "name_to_label": self.name_to_label
            }
            with open(config.LABELS_FILE, "wb") as f:
                pickle.dump(labels_data, f)
                
            print(f"\n[OK] Model saved to: {config.MODEL_FILE}")
            print(f"[OK] Labels saved to: {config.LABELS_FILE}")
            return True
            
        except Exception as e:
            print(f"\n[X] Error saving model: {str(e)}")
            return False
    
    def load_model(self) -> bool:
        """Load the trained model and labels from disk"""
        try:
            if not os.path.exists(config.MODEL_FILE):
                print(f"\n[X] Model file not found: {config.MODEL_FILE}")
                return False
                
            if not os.path.exists(config.LABELS_FILE):
                print(f"\n[X] Labels file not found: {config.LABELS_FILE}")
                return False
            
            # Load the recognizer model
            self.recognizer.read(config.MODEL_FILE)
            
            # Load the label mappings
            with open(config.LABELS_FILE, "rb") as f:
                labels_data = pickle.load(f)
                
            self.label_to_name = labels_data["label_to_name"]
            self.name_to_label = labels_data["name_to_label"]
            
            print(f"\n[OK] Loaded model with {len(self.label_to_name)} person(s)")
            print(f"  Known persons: {', '.join(self.label_to_name.values())}")
            return True
            
        except Exception as e:
            print(f"\n[X] Error loading model: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """Get statistics about the model"""
        return {
            "num_persons": len(self.label_to_name),
            "persons": list(self.label_to_name.values())
        }


# Keep the old function name for compatibility
class FaceEncoder(FaceTrainer):
    """Alias for FaceTrainer for backward compatibility"""
    
    def encode_training_images(self) -> Tuple[int, int]:
        """Alias for train()"""
        return self.train()
    
    def save_encodings(self) -> bool:
        """Alias for save_model()"""
        return self.save_model()
    
    def load_encodings(self) -> bool:
        """Alias for load_model()"""
        return self.load_model()
    
    def get_encoding_stats(self) -> Dict:
        """Alias for get_stats()"""
        stats = self.get_stats()
        # Convert to old format
        return {
            "total": stats["num_persons"],
            "persons": {name: 1 for name in stats["persons"]}
        }


def train_model():
    """Train the face recognition model"""
    trainer = FaceTrainer()
    
    success, fail = trainer.train()
    
    if success > 0:
        trainer.save_model()
        
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Total successful: {success}")
        print(f"  Total failed: {fail}")
        
        stats = trainer.get_stats()
        print(f"\n  Trained persons:")
        for name in stats["persons"]:
            print(f"    - {name}")
            
        print(f"\n  Next step: Run 'python main.py --recognize' to start recognition")
        print(f"{'='*60}\n")
        return True
    else:
        print("\n[X] No faces were trained successfully")
        return False


# Alias for backward compatibility
encode_faces = train_model


if __name__ == "__main__":
    train_model()

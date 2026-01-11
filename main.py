#!/usr/bin/env python3
"""
Real-Time Face Recognition System (OpenCV Version)
===================================================

A Python application that identifies specific people in a live webcam feed
using OpenCV's LBPH face recognizer. Compatible with Python 3.14+!

Usage:
    python main.py                  # Interactive menu
    python main.py --capture        # Capture training images
    python main.py --train          # Train the face recognition model
    python main.py --recognize      # Start face recognition
    python main.py --all            # Run complete pipeline

"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.capture_training_images import capture_training_images
from src.encode_faces import train_model, FaceTrainer
from src.recognize_faces import start_recognition
from src import config


def print_banner():
    """Print application banner"""
    banner = """
    ================================================================
    |                                                              |
    |       REAL-TIME FACE RECOGNITION SYSTEM                      |
    |                                                              |
    |       Powered by OpenCV + LBPH Algorithm                     |
    |       Compatible with Python 3.14+                           |
    |                                                              |
    ================================================================
    """
    print(banner)


def print_menu():
    """Print interactive menu"""
    print("\n" + "="*60)
    print("  MAIN MENU")
    print("="*60)
    print("""
    [1] Capture Training Images
        -> Take photos of a person for training
        
    [2] Train Face Recognition Model  
        -> Process training images into a model
        
    [3] Start Face Recognition
        -> Begin real-time recognition from webcam
        
    [4] Run Complete Pipeline
        -> Capture -> Train -> Recognize
        
    [5] View System Status
        -> Check training data and model
        
    [Q] Quit
    """)
    print("="*60)


def check_system_status():
    """Check and display system status"""
    print("\n" + "="*60)
    print("  SYSTEM STATUS")
    print("="*60)
    
    # Check training directory
    print(f"\n  Training Directory: {config.TRAINING_DIR}")
    
    if os.path.exists(config.TRAINING_DIR):
        persons = [d for d in os.listdir(config.TRAINING_DIR) 
                  if os.path.isdir(os.path.join(config.TRAINING_DIR, d))]
        
        if persons:
            print(f"  [OK] Found {len(persons)} person(s):")
            for person in persons:
                person_path = os.path.join(config.TRAINING_DIR, person)
                images = [f for f in os.listdir(person_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                status = "[OK]" if len(images) >= config.MIN_TRAINING_IMAGES else "[!]"
                print(f"      {status} {person}: {len(images)} images")
        else:
            print("  [!] No training data found")
    else:
        print("  [X] Training directory does not exist")
    
    # Check model files
    print(f"\n  Model File: {config.MODEL_FILE}")
    
    if os.path.exists(config.MODEL_FILE) and os.path.exists(config.LABELS_FILE):
        trainer = FaceTrainer()
        if trainer.load_model():
            stats = trainer.get_stats()
            print(f"  [OK] Model trained for {stats['num_persons']} person(s)")
            for name in stats['persons']:
                print(f"      - {name}")
    else:
        print("  [!] No trained model found (run training first)")
    
    # Configuration
    print(f"\n  Configuration:")
    print(f"      Recognition Threshold: {config.RECOGNITION_THRESHOLD}")
    print(f"      Camera Index: {config.CAMERA_INDEX}")
    print(f"      Face Size: {config.FACE_SIZE}")
    
    print("\n" + "="*60)


def run_complete_pipeline():
    """Run the complete pipeline: capture → train → recognize"""
    print("\n" + "="*60)
    print("  RUNNING COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Capture
    print("\n>>> Step 1/3: Capturing Training Images")
    if not capture_training_images():
        print("\n[X] Training image capture failed or cancelled")
        return
    
    # Step 2: Train
    print("\n>>> Step 2/3: Training Face Recognition Model")
    if not train_model():
        print("\n[X] Training failed")
        return
    
    # Step 3: Recognize
    print("\n>>> Step 3/3: Starting Face Recognition")
    start_recognition()
    
    print("\n[OK] Pipeline complete!")


def interactive_mode():
    """Run in interactive menu mode"""
    print_banner()
    
    while True:
        print_menu()
        choice = input("  Enter your choice: ").strip().upper()
        
        if choice == '1':
            capture_training_images()
        elif choice == '2':
            train_model()
        elif choice == '3':
            start_recognition()
        elif choice == '4':
            run_complete_pipeline()
        elif choice == '5':
            check_system_status()
        elif choice == 'Q':
            print("\n  Goodbye!\n")
            break
        else:
            print("\n  [!] Invalid choice. Please try again.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Real-Time Face Recognition System (OpenCV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                  # Interactive menu
    python main.py --capture        # Capture training images
    python main.py --train          # Train the model
    python main.py --recognize      # Start face recognition
    python main.py --all            # Run complete pipeline
    python main.py --status         # View system status
        """
    )
    
    parser.add_argument('--capture', action='store_true',
                       help='Capture training images from webcam')
    parser.add_argument('--train', '--encode', action='store_true',
                       help='Train the face recognition model')
    parser.add_argument('--recognize', action='store_true',
                       help='Start real-time face recognition')
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline (capture -> train -> recognize)')
    parser.add_argument('--status', action='store_true',
                       help='View system status')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.capture:
        print_banner()
        capture_training_images()
    elif args.train:
        print_banner()
        train_model()
    elif args.recognize:
        print_banner()
        start_recognition()
    elif args.all:
        print_banner()
        run_complete_pipeline()
    elif args.status:
        print_banner()
        check_system_status()
    else:
        # No arguments - run interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()

"""
Configuration settings for the Face Recognition System (OpenCV Version)
Compatible with Python 3.14+
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_DIR = os.path.join(DATA_DIR, "training")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "face_recognizer.yml")
LABELS_FILE = os.path.join(MODELS_DIR, "labels.pkl")

# Ensure directories exist
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Face Detection Settings
# Using OpenCV's DNN face detector (more accurate than Haar)
USE_DNN_DETECTOR = True  # True for DNN (more accurate), False for Haar (faster)
DNN_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for DNN detection

# Recognition Settings
# Using LBPH (Local Binary Pattern Histogram) - fast and effective
RECOGNITION_THRESHOLD = 80  # Lower = stricter (typical range: 50-100)
# Scores below this are considered a match

# Video Capture Settings
CAMERA_INDEX = 0  # Default camera (change if you have multiple cameras)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Training Settings
MIN_TRAINING_IMAGES = 4  # Minimum images required for training
MAX_TRAINING_IMAGES = 20  # Maximum images to capture
CAPTURE_INTERVAL = 0.6  # Seconds between auto-captures

# Face Processing
FACE_SIZE = (200, 200)  # Resize faces to this size for recognition
MIN_FACE_SIZE = (100, 100)  # Minimum face size to detect

# UI Settings
WINDOW_NAME = "Real-Time Face Recognition"
BOX_COLOR = (0, 255, 0)  # Green BGR for recognized
BOX_COLOR_UNKNOWN = (128, 128, 128)  # Gray for unknown
TEXT_COLOR = (255, 255, 255)  # White BGR
FONT_SCALE = 0.7
FONT_THICKNESS = 2

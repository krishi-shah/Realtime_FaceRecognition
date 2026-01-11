#!/usr/bin/env python3
"""
Real-Time Face Recognition System - Streamlit Web App
======================================================
"""

import streamlit as st
import cv2
import numpy as np
import os
import pickle
from datetime import datetime
from PIL import Image
import tempfile
import time

# Add src to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.encode_faces import FaceTrainer

# Page config
st.set_page_config(
    page_title="Real-Time Face Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'label_ids' not in st.session_state:
        st.session_state.label_ids = {}
    if 'capture_count' not in st.session_state:
        st.session_state.capture_count = 0
    if 'recognition_threshold' not in st.session_state:
        st.session_state.recognition_threshold = config.RECOGNITION_THRESHOLD


def load_face_detector():
    """Load Haar Cascade face detector"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)


def detect_faces(image, face_cascade):
    """Detect faces in an image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=config.MIN_FACE_SIZE
    )
    return faces, gray


def save_training_image(image, person_name):
    """Save a training image"""
    person_dir = os.path.join(config.TRAINING_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{person_name.lower()}_{timestamp}.jpg"
    filepath = os.path.join(person_dir, filename)
    
    cv2.imwrite(filepath, image)
    return filepath


def get_training_stats():
    """Get statistics about training data"""
    if not os.path.exists(config.TRAINING_DIR):
        return {}
    
    stats = {}
    for person in os.listdir(config.TRAINING_DIR):
        person_path = os.path.join(config.TRAINING_DIR, person)
        if os.path.isdir(person_path):
            images = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            stats[person] = len(images)
    return stats


def train_recognition_model():
    """Train the face recognition model"""
    trainer = FaceTrainer()
    
    if not os.path.exists(config.TRAINING_DIR):
        return False, "Training directory not found"
    
    persons = [d for d in os.listdir(config.TRAINING_DIR) 
              if os.path.isdir(os.path.join(config.TRAINING_DIR, d))]
    
    if not persons:
        return False, "No training data found"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        trainer.prepare_training_data(progress_callback=lambda p, m: (
            progress_bar.progress(p), status_text.text(m)
        ))
        trainer.train()
        trainer.save_model()
        
        # Load into session state
        st.session_state.trained_model = trainer.recognizer
        st.session_state.label_ids = trainer.label_ids
        
        progress_bar.progress(100)
        status_text.text("✓ Training complete!")
        return True, f"Model trained successfully with {len(persons)} person(s)"
    except Exception as e:
        return False, f"Training failed: {str(e)}"


def load_trained_model():
    """Load trained model if exists"""
    if os.path.exists(config.MODEL_FILE) and os.path.exists(config.LABELS_FILE):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(config.MODEL_FILE)
            
            with open(config.LABELS_FILE, 'rb') as f:
                labels = pickle.load(f)
            
            st.session_state.trained_model = recognizer
            st.session_state.label_ids = labels
            return True
        except:
            return False
    return False


def recognize_face(face_gray, recognizer, label_ids, threshold):
    """Recognize a face"""
    if recognizer is None:
        return "No Model", 100.0
    
    face_resized = cv2.resize(face_gray, config.FACE_SIZE)
    label, confidence = recognizer.predict(face_resized)
    
    if confidence < threshold:
        name = list(label_ids.keys())[list(label_ids.values()).index(label)]
        return name, confidence
    else:
        return "Unknown", confidence


def main():
    """Main Streamlit app"""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Real-Time Face Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by OpenCV + LBPH Algorithm</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        mode = st.radio(
            "Select Mode:",
            ["📊 Dashboard", "📸 Capture Training Images", "🎓 Train Model", "🔍 Recognize Faces"],
            label_visibility="visible"
        )
        
        st.divider()
        
        # Threshold adjustment
        st.session_state.recognition_threshold = st.slider(
            "Recognition Threshold",
            min_value=30,
            max_value=150,
            value=st.session_state.recognition_threshold,
            help="Lower = stricter matching"
        )
        
        st.divider()
        
        # Model status
        st.subheader("📦 Model Status")
        if st.session_state.trained_model is not None:
            st.success("✓ Model Loaded")
            st.write(f"**Persons:** {len(st.session_state.label_ids)}")
            for name in st.session_state.label_ids.keys():
                st.write(f"- {name}")
        else:
            if load_trained_model():
                st.success("✓ Model Loaded")
            else:
                st.warning("⚠ No trained model")
                st.info("Train a model first!")
    
    # Main content
    if mode == "📊 Dashboard":
        show_dashboard()
    elif mode == "📸 Capture Training Images":
        show_capture_page()
    elif mode == "🎓 Train Model":
        show_training_page()
    elif mode == "🔍 Recognize Faces":
        show_recognition_page()


def show_dashboard():
    """Show dashboard with system status"""
    st.header("📊 System Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    stats = get_training_stats()
    total_images = sum(stats.values()) if stats else 0
    num_persons = len(stats)
    
    with col1:
        st.metric("Persons Trained", num_persons)
    with col2:
        st.metric("Training Images", total_images)
    with col3:
        has_model = os.path.exists(config.MODEL_FILE)
        st.metric("Model Status", "✓ Ready" if has_model else "✗ Not Trained")
    
    st.divider()
    
    if stats:
        st.subheader("👥 Training Data")
        for person, count in stats.items():
            status = "✓" if count >= config.MIN_TRAINING_IMAGES else "⚠"
            st.write(f"{status} **{person}**: {count} images")
    else:
        st.info("No training data yet. Start by capturing training images!")
    
    st.divider()
    
    st.subheader("🚀 Quick Start Guide")
    st.markdown("""
    1. **Capture Training Images**: Take 10-15 photos of each person
    2. **Train Model**: Process the images to create recognition model
    3. **Recognize Faces**: Start real-time face recognition
    """)


def show_capture_page():
    """Show training image capture page"""
    st.header("📸 Capture Training Images")
    
    person_name = st.text_input(
        "Enter person's name:",
        placeholder="e.g., John Doe",
        help="Name to identify this person"
    )
    
    if not person_name:
        st.warning("Please enter a name to begin capturing.")
        return
    
    st.info(f"Capturing images for: **{person_name}**")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Instructions")
        st.markdown("""
        - Look at the camera
        - Click 'Capture' button
        - Vary expressions slightly
        - Capture 10-15 images
        - Keep face centered
        """)
        
        stats = get_training_stats()
        if person_name in stats:
            st.success(f"Current images: {stats[person_name]}")
    
    with col1:
        # Camera input
        img_file = st.camera_input("Take a picture", key=f"camera_{st.session_state.capture_count}")
        
        if img_file is not None:
            # Convert to OpenCV format
            image = Image.open(img_file)
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Detect face
            face_cascade = load_face_detector()
            faces, gray = detect_faces(image_bgr, face_cascade)
            
            if len(faces) > 0:
                # Draw rectangle on first face
                x, y, w, h = faces[0]
                cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
                st.image(image_np, caption="Face Detected!", use_container_width=True)
                
                if st.button("💾 Save This Image", type="primary"):
                    # Save face crop
                    face_crop = image_bgr[y:y+h, x:x+w]
                    filepath = save_training_image(face_crop, person_name)
                    st.success(f"✓ Image saved! Total: {get_training_stats().get(person_name, 0)}")
                    st.session_state.capture_count += 1
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.warning("⚠ No face detected. Please try again with better lighting.")


def show_training_page():
    """Show model training page"""
    st.header("🎓 Train Face Recognition Model")
    
    stats = get_training_stats()
    
    if not stats:
        st.error("❌ No training data found. Please capture training images first.")
        return
    
    st.subheader("Training Data Summary")
    for person, count in stats.items():
        status = "✓" if count >= config.MIN_TRAINING_IMAGES else "⚠"
        st.write(f"{status} **{person}**: {count} images")
    
    st.divider()
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training model... This may take a minute..."):
            success, message = train_recognition_model()
            
            if success:
                st.success(f"✓ {message}")
                st.balloons()
            else:
                st.error(f"✗ {message}")


def show_recognition_page():
    """Show face recognition page"""
    st.header("🔍 Face Recognition")
    
    if st.session_state.trained_model is None:
        st.error("❌ No trained model found. Please train a model first.")
        return
    
    st.info("📹 Enable your camera below to start recognition")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Settings")
        st.write(f"**Threshold:** {st.session_state.recognition_threshold}")
        st.write(f"**Persons:** {len(st.session_state.label_ids)}")
        
        st.divider()
        
        st.subheader("Detected")
        detected_placeholder = st.empty()
    
    with col1:
        img_file = st.camera_input("Camera Feed")
        
        if img_file is not None:
            # Convert to OpenCV format
            image = Image.open(img_file)
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Detect faces
            face_cascade = load_face_detector()
            faces, gray = detect_faces(image_bgr, face_cascade)
            
            detected_names = []
            
            # Process each face
            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                
                # Recognize
                name, confidence = recognize_face(
                    face_gray,
                    st.session_state.trained_model,
                    st.session_state.label_ids,
                    st.session_state.recognition_threshold
                )
                
                # Draw rectangle and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(image_np, (x, y), (x+w, y+h), color, 2)
                
                label = f"{name} ({confidence:.1f})"
                cv2.putText(image_np, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                detected_names.append(f"{name} (conf: {confidence:.1f})")
            
            # Display result
            st.image(image_np, caption="Recognition Result", use_container_width=True)
            
            # Update detected persons
            with detected_placeholder.container():
                if detected_names:
                    for name in detected_names:
                        st.write(f"👤 {name}")
                else:
                    st.write("No faces detected")


if __name__ == "__main__":
    main()

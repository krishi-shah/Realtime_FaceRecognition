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

# Check if cv2.face is available
FACE_RECOGNITION_AVAILABLE = hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create')

# Add src to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config

# Only import FaceTrainer if cv2.face is available
FaceTrainer = None
if FACE_RECOGNITION_AVAILABLE:
    try:
        from src.encode_faces import FaceTrainer
    except (AttributeError, ImportError, TypeError):
        FACE_RECOGNITION_AVAILABLE = False
        FaceTrainer = None

# Page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Sci-Fi Theme CSS with Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    /* Keyframe Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 10px #00f5ff, 0 0 20px #00f5ff, 0 0 30px #00f5ff; }
        50% { text-shadow: 0 0 20px #00f5ff, 0 0 30px #00f5ff, 0 0 40px #00f5ff, 0 0 50px #00f5ff; }
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 245, 255, 0.4); }
        50% { box-shadow: 0 0 30px rgba(0, 245, 255, 0.8), 0 0 40px rgba(0, 245, 255, 0.6); }
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Main App Styling */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
        background-size: 400% 400%;
        animation: gradient-shift 15s ease infinite;
        color: #e0e0e0;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Headers with Animations */
    h1 {
        color: #00f5ff !important;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 900 !important;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 10px #00f5ff, 0 0 20px #00f5ff, 0 0 30px #00f5ff;
        margin-bottom: 0.5rem !important;
        letter-spacing: 3px;
        animation: fadeIn 1s ease-out, glow 3s ease-in-out infinite;
        background: linear-gradient(90deg, #00f5ff 0%, #9d4edd 50%, #00f5ff 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeIn 1s ease-out, glow 3s ease-in-out infinite, gradient-shift 3s linear infinite;
    }
    
    h2 {
        color: #9d4edd !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 10px #9d4edd;
        margin-top: 1.5rem !important;
        border-bottom: 2px solid #9d4edd;
        padding-bottom: 0.5rem;
        animation: fadeIn 0.8s ease-out;
        position: relative;
    }
    
    h2::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 0;
        height: 2px;
        background: linear-gradient(90deg, #9d4edd, #00f5ff);
        animation: slideIn 1s ease-out 0.5s forwards;
    }
    
    h3 {
        color: #ff6b6b !important;
        font-size: 1.6rem !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Buttons - Advanced Neon Effects */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 200% 200%;
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        font-family: 'Rajdhani', sans-serif;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 20px rgba(118, 75, 162, 0.4), 0 0 0 0 rgba(118, 75, 162, 0.7);
        width: 100%;
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
        letter-spacing: 1px;
        animation: fadeIn 0.5s ease-out;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 30px rgba(118, 75, 162, 0.6), 0 0 0 8px rgba(118, 75, 162, 0.1);
        background-position: right center;
        border: 2px solid #9d4edd;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:active {
        transform: translateY(-2px) scale(0.98);
        transition: all 0.1s;
    }
    
    .stButton>button:focus {
        border: 2px solid #00f5ff;
        box-shadow: 0 0 25px rgba(0, 245, 255, 0.7), 0 8px 20px rgba(118, 75, 162, 0.4);
        outline: none;
    }
    
    /* Primary Button */
    button[kind="primary"] {
        background: linear-gradient(135deg, #00f5ff 0%, #0099cc 50%, #00f5ff 100%) !important;
        background-size: 200% 200% !important;
        box-shadow: 0 8px 25px rgba(0, 245, 255, 0.5), 0 0 0 0 rgba(0, 245, 255, 0.7) !important;
        animation: gradient-shift 3s ease infinite, pulse 2s ease-in-out infinite !important;
    }
    
    button[kind="primary"]:hover {
        box-shadow: 0 12px 35px rgba(0, 245, 255, 0.7), 0 0 0 8px rgba(0, 245, 255, 0.15) !important;
        background-position: right center !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: rgba(26, 26, 46, 0.95);
        backdrop-filter: blur(10px);
        border-right: 2px solid #00f5ff;
        box-shadow: 2px 0 20px rgba(0, 245, 255, 0.2);
    }
    
    /* Tabs - Enhanced Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: rgba(26, 26, 46, 0.5);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #9d4edd;
        font-weight: 600;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        letter-spacing: 1px;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(157, 78, 221, 0.1);
        color: #00f5ff;
        transform: translateY(-2px);
        border: 2px solid rgba(157, 78, 221, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(157, 78, 221, 0.3) 0%, rgba(0, 245, 255, 0.3) 100%);
        color: #00f5ff;
        border: 2px solid #00f5ff;
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.4);
    }
    
    /* Text Input */
    .stTextInput>div>div>input {
        background-color: rgba(15, 15, 30, 0.8);
        color: #00f5ff;
        border: 2px solid #00f5ff;
        border-radius: 8px;
        padding: 0.75rem;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #9d4edd;
        box-shadow: 0 0 20px rgba(157, 78, 221, 0.5);
        outline: none;
    }
    
    .stTextInput label {
        color: #e0e0e0 !important;
        font-weight: 600;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
    }
    
    /* Camera Input */
    .stCameraInput>div {
        border: 3px solid #00f5ff;
        border-radius: 15px;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.4), 0 0 60px rgba(0, 245, 255, 0.2);
        animation: pulse 3s ease-in-out infinite, fadeIn 0.8s ease-out;
        overflow: hidden;
    }
    
    .stCameraInput>div:hover {
        box-shadow: 0 0 40px rgba(0, 245, 255, 0.6), 0 0 80px rgba(0, 245, 255, 0.3);
    }
    
    /* Metrics with Animation */
    [data-testid="stMetricValue"] {
        color: #00f5ff !important;
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 15px #00f5ff;
        animation: fadeIn 0.8s ease-out, float 3s ease-in-out infinite;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9d4edd !important;
        font-weight: 600;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        letter-spacing: 1px;
        animation: fadeIn 1s ease-out;
    }
    
    /* Enhanced Info/Success/Warning Boxes */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.15) 0%, rgba(0, 245, 255, 0.05) 100%);
        border-left: 4px solid #00f5ff;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.2);
        animation: fadeIn 0.5s ease-out, slideIn 0.5s ease-out;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(255, 107, 107, 0.05) 100%);
        border-left: 4px solid #ff6b6b;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.2);
        animation: fadeIn 0.5s ease-out, slideIn 0.5s ease-out;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 206, 84, 0.15) 0%, rgba(255, 206, 84, 0.05) 100%);
        border-left: 4px solid #ffce54;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(255, 206, 84, 0.2);
        animation: fadeIn 0.5s ease-out, slideIn 0.5s ease-out;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(157, 78, 221, 0.15) 0%, rgba(157, 78, 221, 0.05) 100%);
        border-left: 4px solid #9d4edd;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(157, 78, 221, 0.2);
        animation: fadeIn 0.5s ease-out, slideIn 0.5s ease-out;
    }
    
    /* Markdown Text */
    p, li {
        color: #d0d0d0 !important;
        font-family: 'Rajdhani', sans-serif;
        line-height: 1.8;
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00f5ff, transparent);
        margin: 2rem 0;
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Columns with Enhanced Styling */
    [data-testid="column"] {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.6) 0%, rgba(22, 33, 62, 0.4) 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(0, 245, 255, 0.3);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3), inset 0 0 30px rgba(0, 245, 255, 0.05);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease-out;
    }
    
    [data-testid="column"]:hover {
        border-color: rgba(0, 245, 255, 0.6);
        box-shadow: 0 12px 35px rgba(0, 245, 255, 0.2), inset 0 0 40px rgba(0, 245, 255, 0.1);
        transform: translateY(-5px);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #00f5ff !important;
        border-width: 4px;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Images */
    img {
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
        animation: fadeIn 0.8s ease-out;
        transition: transform 0.3s ease;
    }
    
    img:hover {
        transform: scale(1.02);
    }
    
    /* Sidebar Elements */
    .css-1lcbmhc, .css-1cypcdb {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Slider */
    .stSlider label {
        color: #e0e0e0 !important;
        font-weight: 600;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
    }
    
    .stSlider [data-baseweb="slider"] {
        background-color: rgba(0, 245, 255, 0.2);
    }
    
    .stSlider [data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #00f5ff, #9d4edd);
        box-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
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
    if not FACE_RECOGNITION_AVAILABLE or FaceTrainer is None:
        return False, "OpenCV face module not available. Make sure opencv-contrib-python is installed."
    
    try:
        trainer = FaceTrainer()
    except (AttributeError, TypeError) as e:
        return False, f"OpenCV face module not available. Make sure opencv-contrib-python is installed. Error: {str(e)}"
    
    if not os.path.exists(config.TRAINING_DIR):
        return False, "Training directory not found"
    
    persons = [d for d in os.listdir(config.TRAINING_DIR) 
              if os.path.isdir(os.path.join(config.TRAINING_DIR, d))]
    
    if not persons:
        return False, "No training data found"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Starting training...")
        progress_bar.progress(10)
        
        # Train the model
        success_count, fail_count = trainer.train()
        
        progress_bar.progress(80)
        status_text.text("Saving model...")
        
        if success_count > 0:
            trainer.save_model()
            
            # Load into session state
            st.session_state.trained_model = trainer.recognizer
            st.session_state.label_ids = trainer.name_to_label
            
            progress_bar.progress(100)
            status_text.text("✓ Training complete!")
            return True, f"Model trained successfully with {len(persons)} person(s), {success_count} images processed"
        else:
            return False, "No images were successfully processed"
    except Exception as e:
        return False, f"Training failed: {str(e)}"


def load_trained_model():
    """Load trained model if exists"""
    if os.path.exists(config.MODEL_FILE) and os.path.exists(config.LABELS_FILE):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(config.MODEL_FILE)
            
            with open(config.LABELS_FILE, 'rb') as f:
                labels_data = pickle.load(f)
            
            st.session_state.trained_model = recognizer
            # Use name_to_label dict from saved labels
            st.session_state.label_ids = labels_data.get("name_to_label", labels_data)
            return True
        except AttributeError:
            # cv2.face not available
            return False
        except Exception:
            return False
    return False


def recognize_face(face_gray, recognizer, label_ids, threshold):
    """Recognize a face"""
    if recognizer is None:
        return "No Model", 100.0
    
    face_resized = cv2.resize(face_gray, config.FACE_SIZE)
    label, confidence = recognizer.predict(face_resized)
    
    if confidence < threshold:
        # label_ids is name_to_label dict: {name: label_id}
        # Find name by label_id
        for name, label_id in label_ids.items():
            if label_id == label:
                return name, confidence
        return "Unknown", confidence
    else:
        return "Unknown", confidence


def main():
    """Main Streamlit app"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>🔮 FACE RECOGNITION SYSTEM</h1>
        <p style='color: #9d4edd; font-size: 1.1rem; margin-top: -1rem;'>Neural Network Powered Identification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📊 DASHBOARD", "📸 CAPTURE", "🎓 TRAIN", "🔍 RECOGNIZE"])
    
    with tab1:
        show_dashboard()
    with tab2:
        show_capture_page()
    with tab3:
        show_training_page()
    with tab4:
        show_recognition_page()
    
    # Settings in sidebar (collapsed by default)
    with st.sidebar:
        st.markdown("### ⚙️ SETTINGS")
        st.session_state.recognition_threshold = st.slider(
            "Recognition Threshold",
            min_value=30,
            max_value=150,
            value=st.session_state.recognition_threshold,
            help="Lower = stricter matching"
        )
        
        st.divider()
        st.markdown("### 📦 MODEL STATUS")
        if st.session_state.trained_model is not None:
            st.success(f"✓ Active Model\n**{len(st.session_state.label_ids)} Person(s)**")
            for name in st.session_state.label_ids.keys():
                st.write(f"• {name}")
        else:
            if load_trained_model():
                st.success(f"✓ Model Loaded\n**{len(st.session_state.label_ids)} Person(s)**")
            else:
                st.warning("⚠ No Model")
                st.caption("Train a model first")


def show_dashboard():
    """Show dashboard with system status"""
    st.markdown("<br>", unsafe_allow_html=True)
    
    stats = get_training_stats()
    total_images = sum(stats.values()) if stats else 0
    num_persons = len(stats)
    has_model = os.path.exists(config.MODEL_FILE)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PERSONS TRAINED", num_persons, delta=None)
    with col2:
        st.metric("TRAINING IMAGES", total_images, delta=None)
    with col3:
        status_text = "READY" if has_model else "NOT TRAINED"
        st.metric("MODEL STATUS", status_text)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if stats:
        st.markdown("### 👥 Training Data")
        for person, count in stats.items():
            status = "✓" if count >= config.MIN_TRAINING_IMAGES else "⚠"
            st.markdown(f"**{status} {person}** - {count} images")
    else:
        st.info("👆 Start by capturing training images in the CAPTURE tab")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. **CAPTURE** - Take 10-15 photos of each person
    2. **TRAIN** - Process images to create recognition model  
    3. **RECOGNIZE** - Start real-time face recognition
    """)


def show_capture_page():
    """Show training image capture page"""
    st.markdown("<br>", unsafe_allow_html=True)
    
    person_name = st.text_input(
        "👤 Enter Person's Name",
        placeholder="e.g., John Doe",
        help="Name to identify this person"
    )
    
    if not person_name:
        st.info("👆 Enter a name above to begin capturing")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### 📸 Capturing: **{person_name}**")
        img_file = st.camera_input("Position your face in the camera", key=f"camera_{st.session_state.capture_count}")
        
        if img_file is not None:
            image = Image.open(img_file)
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            face_cascade = load_face_detector()
            faces, gray = detect_faces(image_bgr, face_cascade)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 3)
                st.image(image_np, use_container_width=True)
                
                if st.button("💾 SAVE IMAGE", type="primary", use_container_width=True):
                    face_crop = image_bgr[y:y+h, x:x+w]
                    save_training_image(face_crop, person_name)
                    current_count = get_training_stats().get(person_name, 0)
                    st.success(f"✓ Saved! Total: {current_count} images")
                    st.session_state.capture_count += 1
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.warning("⚠ No face detected. Ensure good lighting and face the camera.")
    
    with col2:
        st.markdown("### 📋 Instructions")
        st.markdown("""
        - Face the camera directly
        - Keep face centered
        - Vary expressions slightly
        - Capture **10-15 images**
        - Ensure good lighting
        """)
        st.markdown("---")
        stats = get_training_stats()
        if person_name in stats:
            st.metric("Images Saved", stats[person_name])


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
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not FACE_RECOGNITION_AVAILABLE:
        st.error("❌ Face recognition module not available")
        st.info("Install: `pip install opencv-contrib-python-headless`")
        return
    
    if st.session_state.trained_model is None:
        st.error("❌ No trained model found")
        st.info("👆 Train a model first in the TRAIN tab")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Live Recognition")
        img_file = st.camera_input("Enable camera to start", key="recognition_camera")
        
        if img_file is not None:
            image = Image.open(img_file)
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            face_cascade = load_face_detector()
            faces, gray = detect_faces(image_bgr, face_cascade)
            
            detected_names = []
            
            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                name, confidence = recognize_face(
                    face_gray,
                    st.session_state.trained_model,
                    st.session_state.label_ids,
                    st.session_state.recognition_threshold
                )
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(image_np, (x, y), (x+w, y+h), color, 3)
                label = f"{name} ({confidence:.1f})"
                cv2.putText(image_np, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                detected_names.append((name, confidence))
            
            st.image(image_np, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Status")
        st.metric("Threshold", st.session_state.recognition_threshold)
        st.metric("Persons", len(st.session_state.label_ids))
        st.markdown("---")
        st.markdown("### 👤 Detected")
        if img_file is not None and detected_names:
            for name, conf in detected_names:
                if name != "Unknown":
                    st.success(f"**{name}**\n*Confidence: {conf:.1f}*")
                else:
                    st.warning(f"**Unknown**\n*Confidence: {conf:.1f}*")
        else:
            st.caption("No faces detected")


if __name__ == "__main__":
    main()

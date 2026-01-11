# 🎭 Real-Time Face Recognition System

A Python application that uses **OpenCV's LBPH (Local Binary Pattern Histogram)** algorithm to identify specific people in a live webcam feed. The system is trained on a small set of images and can recognize the trained person(s) in real-time.

**✅ Compatible with Python 3.14+ and Windows!**

## 🌐 Live Demo

**Try it now:** [https://realtime-face-recognition.streamlit.app/](https://realtime-face-recognition.streamlit.app/)

The web app allows you to:
- 📸 Capture training images using your browser's camera
- 🎓 Train face recognition models
- 🔍 Recognize faces in real-time
- 📊 View system statistics and dashboard

## ✨ Features

- **LBPH Face Recognition**: Fast and effective algorithm built into OpenCV
- **Real-Time Recognition**: Smooth 30+ FPS performance
- **Webcam Training Capture**: Built-in tool to capture training images
- **Interactive UI**: Visual feedback with bounding boxes and name labels
- **Adjustable Threshold**: Fine-tune recognition strictness in real-time
- **Local Processing**: All computation runs locally, no cloud services
- **Easy Installation**: Just `pip install` - no C++ compiler needed!
- **Python 3.14 Compatible**: Works with the latest Python versions

## 📋 Requirements

- Python 3.8+ (tested up to Python 3.14)
- Webcam
- Windows 10/11, Linux, or macOS

### Dependencies

- `opencv-python` - Video capture and image processing
- `opencv-contrib-python` - LBPH Face Recognizer
- `numpy` - Numerical operations
- `streamlit` - Web app framework (for web deployment)

## 🚀 Installation

### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate it (Windows CMD)
venv\Scripts\activate.bat

# Activate it (Linux/macOS)
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install opencv-python opencv-contrib-python numpy pillow
```

## 📖 Usage

### Interactive Mode (Recommended)

Simply run the main script:

```bash
python main.py
```

This opens an interactive menu:

```
[1] Capture Training Images   → Take photos of a person
[2] Train Face Recognition    → Create the recognition model
[3] Start Face Recognition    → Begin real-time recognition
[4] Run Complete Pipeline     → All steps in sequence
[5] View System Status        → Check training data
[Q] Quit
```

### Command Line Mode

```bash
# Capture training images
python main.py --capture

# Train the model
python main.py --train

# Start recognition
python main.py --recognize

# Run complete pipeline
python main.py --all

# Check system status
python main.py --status
```

### Web App (Streamlit)

For the web-based interface:

```bash
# Install Streamlit
pip install streamlit

# Run the web app
streamlit run app.py
```

This opens a browser-based interface with:
- 📸 Camera-based training image capture
- 🎓 Interactive model training
- 🔍 Real-time face recognition
- 📊 System dashboard and statistics

## 🌐 Web Deployment (Streamlit Cloud)

This app is deployed and running at: **[https://realtime-face-recognition.streamlit.app/](https://realtime-face-recognition.streamlit.app/)**

### Deploy Your Own

1. Fork this repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app"
4. Select your repository and set main file to `app.py`
5. Click "Deploy!"

**Note:** The web app uses browser camera access - users must grant camera permissions.

## 🎯 Quick Start Guide

### 1. Capture Training Images

Run the capture tool and enter the person's name when prompted:

```bash
python main.py --capture
```

**Controls during capture:**
- `SPACE` - Capture image manually
- `A` - Toggle auto-capture mode (hands-free!)
- `Q` - Save and quit
- `ESC` - Cancel

**Tips for good training images:**
- Capture **10-15 images** for best results
- Vary your facial expressions slightly
- Include different angles (front, slight left/right)
- Ensure good, consistent lighting
- Keep your face centered and clearly visible

### 2. Train the Model

Process your training images:

```bash
python main.py --train
```

This creates an LBPH model trained on your face images.

### 3. Start Recognition

Begin real-time face recognition:

```bash
python main.py --recognize
```

**Controls during recognition:**
- `Q` or `ESC` - Quit
- `+` / `-` - Adjust threshold (matching strictness)
- `S` - Take screenshot

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Recognition Threshold (lower = stricter)
# Typical range: 50-100
# Lower values = fewer false positives but might miss real matches
RECOGNITION_THRESHOLD = 80

# Camera Settings
CAMERA_INDEX = 0  # Change if you have multiple cameras

# Face Size for processing
FACE_SIZE = (200, 200)

# Minimum face size to detect
MIN_FACE_SIZE = (100, 100)
```

## 📁 Project Structure

```
Realtime_faceRecognition/
├── main.py                 # Desktop app entry point
├── app.py                  # Streamlit web app entry point
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies (for Streamlit Cloud)
├── README.md               # This file
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── capture_training_images.py  # Training image capture
│   ├── encode_faces.py     # Model training (LBPH)
│   └── recognize_faces.py  # Real-time recognition
├── data/
│   └── training/           # Training images by person
│       └── Person_Name/    # Folder for each person
├── models/
│   ├── face_recognizer.yml # Trained LBPH model
│   └── labels.pkl          # Name-label mappings
└── screenshots/            # Saved screenshots
```

## 🔧 Troubleshooting

### "Could not open webcam"

- Check if another application is using the camera
- Try changing `CAMERA_INDEX` in config.py (try 0, 1, 2)
- On Windows, check camera privacy settings

### Recognition is not accurate

- **Add more training images** (10-15 minimum)
- Adjust `RECOGNITION_THRESHOLD`:
  - Lower (e.g., 60) = stricter matching
  - Higher (e.g., 100) = more lenient
- Ensure training images have good lighting
- Make sure your face is clearly visible during capture

### False positives (wrong person recognized)

- Lower the `RECOGNITION_THRESHOLD` (try 50-70)
- Capture more distinct training images
- Ensure different people in training look distinct

### Face not detected

- Ensure good lighting (avoid backlight)
- Face the camera directly
- Move closer to the camera
- Remove heavy obstructions (large glasses, masks)

### Module 'cv2.face' not found

Make sure you installed `opencv-contrib-python`:
```bash
pip install opencv-contrib-python
```

## 📊 How LBPH Works

**Local Binary Pattern Histogram (LBPH)** is a texture-based face recognition algorithm:

1. **Training**: 
   - Divides face into local regions
   - Computes texture patterns (LBP) for each region
   - Creates histograms of patterns per person

2. **Recognition**:
   - Processes new face the same way
   - Compares histogram to stored ones
   - Returns closest match with confidence score

**Advantages:**
- ✅ Fast and efficient
- ✅ Works well with different lighting
- ✅ No deep learning frameworks needed
- ✅ Trains quickly on small datasets

## 🎨 Tips for Best Results

1. **Good Lighting**: Even, front-facing light works best
2. **Consistent Distance**: Stay about arm's length from camera
3. **Multiple Images**: More training images = better accuracy
4. **Clean Background**: Plain backgrounds help detection
5. **Face Visibility**: No sunglasses, masks, or heavy shadows

## 🙏 Acknowledgments

- [OpenCV](https://opencv.org/) - Computer vision library
- [OpenCV Face Module](https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html) - Face recognition algorithms

## 📄 License

This project is for educational purposes.

#!/usr/bin/env python3
"""
Simple camera test to verify OpenCV is working
"""

import cv2
import sys

print("="*60)
print("  CAMERA TEST")
print("="*60)

# Try to open camera with different backends
print("\n1. Testing camera access...")

backends = [
    (cv2.CAP_MSMF, "Media Foundation (MSMF)"),
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_ANY, "Auto-detect"),
]

cap = None
working_backend = None

for backend_id, backend_name in backends:
    print(f"\n   Trying {backend_name}...")
    test_cap = cv2.VideoCapture(0, backend_id)
    
    if test_cap.isOpened():
        # Test if we can read frames
        ret, frame = test_cap.read()
        if ret and frame is not None:
            print(f"   [OK] {backend_name} works!")
            cap = test_cap
            working_backend = backend_name
            break
        else:
            print(f"   [!] {backend_name} opened but cannot read frames")
            test_cap.release()
    else:
        print(f"   [X] {backend_name} failed to open camera")

if cap is None or not cap.isOpened():
    print("\n   [X] FAILED - Could not open camera with any backend")
    print("\n   Possible issues:")
    print("      - Camera is being used by another application (close Windows Camera app!)")
    print("      - Camera permissions are not granted")
    print("      - Camera is not connected")
    print("\n   Try:")
    print("      1. Close Windows Camera app if it's open")
    print("      2. Go to Settings > Privacy > Camera and enable desktop app access")
    print("      3. Restart your computer")
    sys.exit(1)

print(f"\n   [OK] Camera opened successfully with {working_backend}")

# Test reading a fresh frame
print("\n2. Testing frame capture...")
ret, frame = cap.read()
if not ret or frame is None:
    print("   [X] FAILED - Could not read frames from camera")
    cap.release()
    sys.exit(1)

print(f"   [OK] Frame captured - Size: {frame.shape[1]}x{frame.shape[0]}")
print(f"       Channels: {frame.shape[2]}, Data type: {frame.dtype}")

# Test window creation
print("\n3. Testing OpenCV GUI window...")
try:
    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
    cv2.imshow("Camera Test", frame)
    print("   [OK] Window created")
    print("\n" + "="*60)
    print("  If you can see a window with your camera feed,")
    print("  OpenCV is working correctly!")
    print("\n  Press any key in the window to continue...")
    print("="*60)
    
    # Wait for keypress or 10 seconds
    key = cv2.waitKey(10000)
    
    if key == -1:
        print("\n   [!] No key pressed within 10 seconds")
        print("       This might indicate the window didn't appear")
    else:
        print(f"\n   [OK] Key pressed (code: {key})")
    
except Exception as e:
    print(f"   [X] FAILED - Error: {e}")
    cap.release()
    sys.exit(1)

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("  Camera test complete!")
print("="*60)

#!/usr/bin/env python3
"""
HP Camera-specific test with workarounds
"""

import cv2
import sys
import time

print("="*60)
print("  HP CAMERA FIX TEST")
print("="*60)

print("\nAttempting HP Wide Vision camera workarounds...")

# Method 1: Try CAP_MSMF with explicit settings
print("\n1. Trying Media Foundation with explicit format...")
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if cap.isOpened():
    # Set format before trying to read
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    
    time.sleep(0.5)  # Give camera time to initialize
    
    # Try to read multiple frames (first few might fail)
    print("   Warming up camera...")
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"   [OK] Frame {i+1} captured!")
            print(f"   Size: {frame.shape[1]}x{frame.shape[0]}")
            
            # Try to show the window
            try:
                cv2.namedWindow("HP Camera Test", cv2.WINDOW_NORMAL)
                cv2.imshow("HP Camera Test", frame)
                print("\n   [SUCCESS] If you can see the window, press any key...")
                key = cv2.waitKey(5000)
                
                if key != -1:
                    print(f"   [OK] Key pressed! OpenCV is working!")
                else:
                    print("   [!] No key pressed, but window should be visible")
                    print("       Check if a window appeared on your screen")
                
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
            except Exception as e:
                print(f"   [!] Window error: {e}")
                cap.release()
                break
        else:
            print(f"   [!] Frame {i+1} failed, retrying...")
            time.sleep(0.1)
    
    cap.release()

print("\n   [X] Method 1 failed")

# Method 2: Try without specifying backend
print("\n2. Trying default backend...")
cap = cv2.VideoCapture(0)

if cap.isOpened():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    time.sleep(0.5)
    
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"   [OK] Frame captured with default backend!")
            print(f"   Size: {frame.shape[1]}x{frame.shape[0]}")
            
            cv2.namedWindow("HP Camera Test", cv2.WINDOW_NORMAL)
            cv2.imshow("HP Camera Test", frame)
            print("\n   [SUCCESS] Press any key in the window...")
            cv2.waitKey(5000)
            
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        time.sleep(0.1)
    
    cap.release()

print("\n   [X] Method 2 failed")
print("\n" + "="*60)
print("  RECOMMENDATION:")
print("  Your HP camera may need a driver update or")
print("  there might be security software blocking OpenCV.")
print("\n  Try:")
print("  1. Update camera drivers from HP Support")
print("  2. Temporarily disable antivirus")
print("  3. Try a different Python/OpenCV version")
print("="*60)

import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

# --- SET UP DIRECTORIES ---
folder_choice = input("Enter folder name to save images to (e.g., 'badge' or 'no_badge'): ").strip()
save_dir = os.path.join("dataset", "raw", folder_choice)
os.makedirs(save_dir, exist_ok=True)

# ==============================================================================
#                         CAMERA CALIBRATION CONFIGURATION
# ==============================================================================
# This system uses Triangle Similarity Math to estimate distance:
# Distance (inches) = (Real Width * Focal Length Factor) / Pixel Width on Screen
# ==============================================================================

# 1. PHYSICAL CONSTANT: Your real-world shoulder width in inches.
# Measure straight across the front of your chest from shoulder joint to shoulder joint.
# An average adult is roughly 16.0 to 18.0 inches.
REAL_SHOULDER_WIDTH_INCHES = 17.0
# 2. OPTICAL FACTOR (Focal Length): Maps your specific camera lens distortion.
# Since your Microsoft Surface camera has a specific wide-angle field of view, 
# you must calibrate this number once so the software math lines up with reality.
#
# HOW TO CALIBRATE THIS NUMBER FOR YOUR DEVICE:
#   Step A: Set this number temporarily to 530.0.
#   Step B: Run the script, grab a physical tape measure, and stand EXACTLY 3.0 feet 
#           (which is 36 inches) away from your Surface camera lens.
#   Step C: Look at the terminal readout:
#           - If the screen says you are "2.5 feet" away (too low), INCREASE this factor.
#           - If the screen says you are "4.1 feet" away (too high), DECREASE this factor.
#   Step D: Tweak it up or down until the live readout reads exactly "3.0 feet".
#
# Note: Turning OFF "Automatic Framing" in Windows settings keeps this factor stable!
FOCAL_LENGTH_FACTOR = 318 
# ==============================================================================

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(1)
img_counter = 0
Offset = -60

# --- INITIALIZE SMART COUNTER ---
# Counts how many images already exist in the target directory 
# so we don't reset our numbering sequence when restarting the script.
existing_files = [f for f in os.listdir(save_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
img_counter = len(existing_files)

print(f"\nReady to collect data! Images will save to: {save_dir}")
print("Instructions:")
print("1. Stand closer than 3.5 feet (Box turns RED).")
print("2. Press SPACEBAR to capture a squared chest crop.")
print("3. Press 'q' to finish and exit.\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy()
    
    results = model(frame, stream=True, verbose=False)
    current_distance = None
    chest_crop_square = None

    for r in results:
        # Check if the model found any keypoints AT ALL
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            # SAFETY SECURED: It is now safe to grab index [0]
            keypoints = r.keypoints.xy[0].cpu().numpy()
            
            if len(keypoints) > 6:
                ls_x, ls_y = int(keypoints[5][0]), int(keypoints[5][1])
                rs_x, rs_y = int(keypoints[6][0]), int(keypoints[6][1])
                
                # ... the rest of your math logic stays identical down here ...
                
                if ls_x > 0 and rs_x > 0:
                    pixel_width = np.sqrt((ls_x - rs_x)**2 + (ls_y - rs_y)**2)
                    if pixel_width > 0:
                        current_distance = (REAL_SHOULDER_WIDTH_INCHES * FOCAL_LENGTH_FACTOR) / pixel_width
                    
                    # Compute Bounding Boxes
                    box_width = int(pixel_width)
                    # We make height EQUAL to width to enforce a perfect mathematical square crop!
                    box_height = box_width 
                    
                    x_min, x_max = min(ls_x, rs_x), max(ls_x, rs_x)
                    y_min = min(ls_y, rs_y) + Offset
                    y_max = y_min + box_height
                    
                    # Visual feedback colors
                    is_in_range = current_distance is not None and (current_distance / 12.0) <= 3.5
                    box_color = (0, 0, 255) if is_in_range else (255, 0, 0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                    
                    if is_in_range:
                        # Grab the square patch from the pristine clean frame
                        crop = clean_frame[y_min:y_max, x_min:x_max]
                        if crop.size > 0:
                            # Resize immediately to standard 224x224 so all data matches perfectly
                            chest_crop_square = cv2.resize(crop, (224, 224))
                            cv2.imshow("Live Target Crop Preview", chest_crop_square)

    cv2.imshow("Data Collector Sandbox", frame)
    
    # === PRINT LIVE DISTANCE FEEDBACK TO TERMINAL ===
    if current_distance is not None:
        distance_feet = current_distance / 12.0
        print(f"Status: Tracking | Live Distance: {distance_feet:.1f} ft ({int(current_distance)} in)     ", end="\r")
    else:
        print("Status: Searching for shoulders...                                                     ", end="\r")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32: # SPACEBAR keycode
        if chest_crop_square is not None:
            img_name = f"crop_{int(time.time())}_{img_counter}.jpg"
            cv2.imwrite(os.path.join(save_dir, img_name), chest_crop_square)
            print(f"Captured and Saved: {img_name}")
            img_counter += 1
        else:
            print("Capture failed: You must be closer than 3.5 feet and clearly visible!")

cap.release()
cv2.destroyAllWindows()
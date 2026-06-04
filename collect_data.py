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
REAL_SHOULDER_WIDTH_INCHES = 17.0
FOCAL_LENGTH_FACTOR = 318 
# ==============================================================================

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(1)
Offset = -60

# --- INITIALIZE SMART COUNTER ---
existing_files = [f for f in os.listdir(save_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
img_counter = len(existing_files)

print(f"\nReady to collect data! Images will save to: {save_dir}")
print(f"Current folder contains {img_counter} images. Starting sequence at index: {img_counter}")
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
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            keypoints = r.keypoints.xy[0].cpu().numpy()
            
            if len(keypoints) > 6:
                ls_x, ls_y = int(keypoints[5][0]), int(keypoints[5][1])
                rs_x, rs_y = int(keypoints[6][0]), int(keypoints[6][1])
                
                if ls_x > 0 and rs_x > 0:
                    pixel_width = np.sqrt((ls_x - rs_x)**2 + (ls_y - rs_y)**2)
                    if pixel_width > 0:
                        current_distance = (REAL_SHOULDER_WIDTH_INCHES * FOCAL_LENGTH_FACTOR) / pixel_width
                    
                    # === INTEGRATED: TIGHT CENTERED BOX LOGIC ===
                    # 1. Shrink the box to 65% of your shoulder width
                    box_width = int(pixel_width * 0.65)
                    box_height = box_width  # Enforce a perfect mathematical square crop!
                    
                    # 2. Shift X to keep it dead-center over your sternum
                    shoulder_center_x = min(ls_x, rs_x) + (abs(ls_x - rs_x) // 2)
                    x_min = shoulder_center_x - (box_width // 2)
                    x_max = x_min + box_width
                    
                    # 3. Apply vertical ceiling offset
                    y_min = min(ls_y, rs_y) + Offset
                    y_max = y_min + box_height
                    # ===========================================
                    
                    # Visual feedback colors
                    is_in_range = current_distance is not None and (current_distance / 12.0) <= 3.5
                    box_color = (0, 0, 255) if is_in_range else (255, 0, 0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                    
                    if is_in_range:
                        crop = clean_frame[y_min:y_max, x_min:x_max]
                        if crop.size > 0:
                            chest_crop_square = cv2.resize(crop, (224, 224))
                            cv2.imshow("Live Target Crop Preview", chest_crop_square)

    cv2.imshow("Data Collector Sandbox", frame)
    
    if current_distance is not None:
        distance_feet = current_distance / 12.0
        print(f"Status: Tracking | Live Distance: {distance_feet:.1f} ft ({int(current_distance)} in)     ", end="\r")
    else:
        print("Status: Searching for shoulders...                                                             ", end="\r")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32: # SPACEBAR
        if chest_crop_square is not None:
            img_name = f"crop_{int(time.time())}_{img_counter}.jpg"
            cv2.imwrite(os.path.join(save_dir, img_name), chest_crop_square)
            print(f"Captured and Saved: {img_name}")
            img_counter += 1
        else:
            print("Capture failed: You must be closer than 3.5 feet and clearly visible!")

cap.release()
cv2.destroyAllWindows()
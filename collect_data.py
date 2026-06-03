import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

# --- SET UP DIRECTORIES ---
folder_choice = input("Enter folder name to save images to (e.g., 'badge' or 'no_badge'): ").strip()
save_dir = os.path.join("dataset", "raw", folder_choice)
os.makedirs(save_dir, exist_ok=True)

# Camera Calibration Constants
REAL_SHOULDER_WIDTH_INCHES = 17.0
FOCAL_LENGTH_FACTOR = 530.0 

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)
img_counter = 0
Offset = -60

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
        if r.keypoints is not None:
            keypoints = r.keypoints.xy[0].cpu().numpy()
            if len(keypoints) > 6:
                ls_x, ls_y = int(keypoints[5][0]), int(keypoints[5][1])
                rs_x, rs_y = int(keypoints[6][0]), int(keypoints[6][1])
                
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
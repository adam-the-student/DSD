import cv2
import time
import numpy as np
from ultralytics import YOLO

# --- CALIBRATION SETTINGS ---
REAL_SHOULDER_WIDTH_INCHES = 17.0
FOCAL_LENGTH_FACTOR = 530.0 

def display_metrics(start_time, distance_inches=None):
    elapsed_time = time.time() - start_time
    if elapsed_time == 0: elapsed_time = 0.001
    fps = 1.0 / elapsed_time
    
    if distance_inches is not None:
        distance_feet = distance_inches / 12.0
        distance_str = f"| Distance: {distance_feet:.1f} ft | "
        if distance_feet <= 3.5:
            distance_str += "[TARGET IN RANGE - CHECKING BADGE]"
        else:
            distance_str += "[TOO FAR - STAND CLOSER]"
    else:
        distance_str = f"| Distance: Searching..."

    print(f"Performance: {int(fps)} FPS {distance_str}          ", end="\r")

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)
#offset of captured chest
offset = -60

print("Starting Core Logic Sandbox. Press 'q' to quit.\n")

while True:
    frame_start_time = time.time()
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    
    # Store a completely clean copy of the frame BEFORE drawing lines on it
    # We use this clean copy to generate our pure crop for the TFLite model
    clean_frame = frame.copy() 
    
    results = model(frame, stream=True)
    current_distance = None

    for r in results:
        if r.keypoints is not None:
            keypoints = r.keypoints.xy[0].cpu().numpy()
            
            if len(keypoints) > 6:
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                
                ls_x, ls_y = int(left_shoulder[0]), int(left_shoulder[1])
                rs_x, rs_y = int(right_shoulder[0]), int(right_shoulder[1])
                
                if ls_x > 0 and rs_x > 0:
                    # 1. Calculate distance first
                    pixel_width = np.sqrt((ls_x - rs_x)**2 + (ls_y - rs_y)**2)
                    if pixel_width > 0:
                        current_distance = (REAL_SHOULDER_WIDTH_INCHES * FOCAL_LENGTH_FACTOR) / pixel_width
                    
                    # 2. Draw Skeleton overlays on our display frame
                    cv2.circle(frame, (ls_x, ls_y), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (rs_x, rs_y), 6, (0, 255, 0), -1)
                    cv2.line(frame, (ls_x, ls_y), (rs_x, rs_y), (0, 255, 0), 2)
                    
                    # 3. Dynamic Chest Box math
                    box_height = int(pixel_width * 0.6)
                    x_min, x_max = min(ls_x, rs_x), max(ls_x, rs_x)
                    y_min = min(ls_y, rs_y) + offset
                    y_max = y_min + box_height
                    
                    # 4. If person is closer than 3.5 feet, extract the crop!
                    if current_distance is not None and (current_distance / 12.0) <= 3.5:
                        # Draw box in RED when actively processing
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                        
                        # SLICE the image array using NumPy indexing: [y_range, x_range]
                        # We pull this from clean_frame so the green tracking dots aren't in the crop!
                        chest_crop = clean_frame[y_min:y_max, x_min:x_max]
                        
                        # Verify the crop isn't empty (avoids edge-of-screen crashes)
                        if chest_crop.size > 0:
                            # Show what the Stage 2 Classifier will actually see
                            cv2.imshow("Chest Patch Crop (Stage 2 Input)", chest_crop)
                    else:
                        # Draw box in BLUE when person is too far away to care
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        # Clear crop window if they step away
                        if cv2.getWindowProperty("Chest Patch Crop (Stage 2 Input)", cv2.WND_PROP_VISIBLE) >= 1:
                            cv2.destroyWindow("Chest Patch Crop (Stage 2 Input)")

    cv2.imshow("Main Camera View", frame)
    display_metrics(frame_start_time, current_distance)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
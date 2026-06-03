import cv2
import time
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import os

# --- CALIBRATION SETTINGS ---
REAL_SHOULDER_WIDTH_INCHES = 17.0
FOCAL_LENGTH_FACTOR = 318.0  # Your calibrated Surface factor
TFLITE_MODEL_PATH = "model.tflite"

# --- INITIALIZE THE TFLITE INTERPRETER ---
print("Loading Stage 2 TFLite Classifier...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

def display_metrics(start_time, distance_inches=None, badge_result="Scanning..."):
    elapsed_time = time.time() - start_time
    if elapsed_time == 0: elapsed_time = 0.001
    fps = 1.0 / elapsed_time
    
    if distance_inches is not None:
        distance_feet = distance_inches / 12.0
        distance_str = f"| Distance: {distance_feet:.1f} ft | Status: {badge_result}"
    else:
        distance_str = f"| Distance: Searching..."

    print(f"Performance: {int(fps)} FPS {distance_str}                               ", end="\r")

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(1)  # LOCKED TO CAMERA INDEX 1
offset = -60  # Perfectly matches your dataset collector!

print("Starting Full Two-Stage Pipeline. Press 'q' to quit.\n")

while True:
    frame_start_time = time.time()
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy() 
    
    results = model(frame, verbose=False)
    current_distance = None
    badge_status = "IDLE (Stand Closer)"

    for r in results:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            keypoints = r.keypoints.xy[0].cpu().numpy()
            
            if len(keypoints) > 6:
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                
                ls_x, ls_y = int(left_shoulder[0]), int(left_shoulder[1])
                rs_x, rs_y = int(right_shoulder[0]), int(right_shoulder[1])
                
                if ls_x > 0 and rs_x > 0:
                    pixel_width = np.sqrt((ls_x - rs_x)**2 + (ls_y - rs_y)**2)
                    if pixel_width > 0:
                        current_distance = (REAL_SHOULDER_WIDTH_INCHES * FOCAL_LENGTH_FACTOR) / pixel_width
                    
                    cv2.circle(frame, (ls_x, ls_y), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (rs_x, rs_y), 6, (0, 255, 0), -1)
                    cv2.line(frame, (ls_x, ls_y), (rs_x, rs_y), (0, 255, 0), 2)
                    
                    # FIX: Enforce a perfect mathematical square to match collect_data.py exactly!
                    box_width = int(pixel_width)
                    box_height = box_width 
                    
                    x_min, x_max = min(ls_x, rs_x), max(ls_x, rs_x)
                    y_min = min(ls_y, rs_y) + offset
                    y_max = y_min + box_height
                    
                    # IF IN RANGE: Run the image crop through the TFLite brain
                    if current_distance is not None and (current_distance / 12.0) <= 3.5:
                        chest_crop = clean_frame[y_min:y_max, x_min:x_max]
                        
                        if chest_crop.size > 0:
                            # 1. Preprocess image to match training format
                            resized_crop = cv2.resize(chest_crop, (input_width, input_height))
                            rgb_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                            normalized_crop = rgb_crop.astype(np.float32) / 255.0
                            input_data = np.expand_dims(normalized_crop, axis=0)
                            
                            # 2. Run inference
                            interpreter.set_tensor(input_details[0]['index'], input_data)
                            interpreter.invoke()
                            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                            
                            # --- BINARY SIGMOID MATH EVALUATION ---
                            prediction_score = output_data[0]
                            
                            if prediction_score < 0.50:
                                confidence = (1.0 - prediction_score) * 100
                                badge_status = f"BADGE DETECTED ({int(confidence)}%)"
                                box_color = (0, 255, 0) # Green box
                            else:
                                confidence = prediction_score * 100
                                badge_status = f"NO BADGE DETECTED ({int(confidence)}%)"
                                box_color = (0, 0, 255) # Red box
                                
                            # Draw visual feedback box and text on the main window
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                            cv2.putText(frame, badge_status, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            
                            # FIX: Displays the exact crisp array seen by the AI
                            cv2.imshow("Chest Patch Crop (Stage 2 Input)", resized_crop)
                    else:
                        # Draw a passive blue box when out of range
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        if cv2.getWindowProperty("Chest Patch Crop (Stage 2 Input)", cv2.WND_PROP_VISIBLE) >= 1:
                            cv2.destroyWindow("Chest Patch Crop (Stage 2 Input)")

    cv2.imshow("Main Camera View", frame)
    display_metrics(frame_start_time, current_distance, badge_status)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
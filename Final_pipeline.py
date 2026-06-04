# Note to self before continuing the project, make the bounding box wider and retrain with that
import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION TUNING CORNER ---
CLASSIFIER_IMG_SIZE = (224, 224)  
CONFIDENCE_THRESHOLD = 60  # Minimum % required to change state (stops flickering)

# --- CALIBRATION SETTINGS ---
REAL_SHOULDER_WIDTH_INCHES = 17.0
FOCAL_LENGTH_FACTOR = 318.0  # Calibrated Surface factor

# Distance constraints measured in FEET
MIN_DISTANCE_FEET = 1   # Too close boundary (~18 inches)
MAX_DISTANCE_FEET = 4.5   # Too far back boundary (~54 inches)

print("Initializing Stage 1 Pose Joint Tracker...")
stage1_detector = YOLO("yolov8n-pose.pt") 

print("Initializing Stage 2 Custom Badge Classifier...")
stage2_classifier = YOLO("models/badgeClassifier.pt")

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not access the webcam stream.")
    exit()

print("\n🚀 Starting Calibrated Distance Two-Stage Pipeline. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detector_results = stage1_detector(frame, verbose=False)
    
    crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, 0, 0
    cropped_chest_frame = None
    distance_status = "OK"
    estimated_ft = 0.0

    for result in detector_results:
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            kpts = result.keypoints.xy[0]
            left_shoulder = kpts[5]
            right_shoulder = kpts[6]
            
            if left_shoulder[0] > 0 and right_shoulder[0] > 0:
                ls_x, ls_y = int(left_shoulder[0]), int(left_shoulder[1])
                rs_x, rs_y = int(right_shoulder[0]), int(right_shoulder[1])
                
                # Calculate pixel width between physical shoulder joints
                shoulder_width_pixels = abs(ls_x - rs_x)
                
                # NATIVE CALIBRATED DISTANCE SENSING (Calculates true distance in feet)
                if shoulder_width_pixels > 0:
                    # Target distance calculation using triangular similarity setup
                    distance_inches = (REAL_SHOULDER_WIDTH_INCHES * FOCAL_LENGTH_FACTOR) / shoulder_width_pixels
                    estimated_ft = round(distance_inches / 12.0, 1)
                
                # Evaluate distance status using real feet metrics instead of raw pixels
                if estimated_ft > MAX_DISTANCE_FEET:
                    distance_status = "TOO FAR"
                elif estimated_ft < MIN_DISTANCE_FEET:
                    distance_status = "TOO CLOSE"
                else:
                    distance_status = "OK"

                # Dynamic scaling multipliers anchoring your exact bounding box canvas
                crop_x1 = min(ls_x, rs_x) + int(shoulder_width_pixels * 0.20)
                crop_x2 = max(ls_x, rs_x) - int(shoulder_width_pixels * 0.20)
                crop_y1 = min(ls_y, rs_y) - int(shoulder_width_pixels * 0.25)
                crop_y2 = min(ls_y, rs_y) + int(shoulder_width_pixels * 0.45)
                
                crop_x1 = max(0, crop_x1)
                crop_y1 = max(0, crop_y1)
                crop_x2 = min(frame.shape[1], crop_x2)
                crop_y2 = min(frame.shape[0], crop_y2)
                
                cropped_chest_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                break

    # --- CLASSIFICATION & THRESHOLDING GATE ---
    badge_status = "SCANNING..."
    text_color = (0, 165, 255)  # Amber Warning

    # Real-world distance threshold filter gate
    if distance_status != "OK":
        badge_status = f"DISTANCE ALERT: {distance_status}"
        text_color = (0, 165, 255)
    elif cropped_chest_frame is not None and cropped_chest_frame.size > 0:
        resized_crop = cv2.resize(cropped_chest_frame, CLASSIFIER_IMG_SIZE)
        classifier_results = stage2_classifier(resized_crop, verbose=False)
        
        class_mapping = classifier_results[0].names  
        top_prediction_id = classifier_results[0].probs.top1
        confidence_percentage = int(classifier_results[0].probs.top1conf.item() * 100)
        predicted_folder_name = class_mapping[top_prediction_id]

        if confidence_percentage >= CONFIDENCE_THRESHOLD:
            if predicted_folder_name == "badge":
                badge_status = f"BADGE DETECTED ({confidence_percentage}%)"
                text_color = (0, 255, 0)
            else:
                badge_status = f"NO BADGE DETECTED ({confidence_percentage}%)"
                text_color = (0, 0, 255)
        else:
            badge_status = f"CALCULATING... ({confidence_percentage}%)"
            text_color = (0, 165, 255)

        cv2.imshow("Chest Patch View", cropped_chest_frame)

    # --- UI RENDERING CORE ---
    if crop_x2 > 0 and crop_y2 > 0:
        cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), 2)
        
        # Format clean, real-time spatial data telemetry on top of the tracker box
        live_telemetry = f"{badge_status} | Dist: {estimated_ft}ft"

        mid_y = int(crop_y1 + (crop_y2 - crop_y1) * 0.3)
        # cv2.line(frame, (crop_x1, mid_y), (crop_x2, mid_y), (0, 255, 0), 2)
        # cv2.circle(frame, (crop_x1, mid_y), 5, (0, 255, 0), -1)
        # cv2.circle(frame, (crop_x2, mid_y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (ls_x,ls_y),5,(255,0,0),2)
        cv2.circle(frame, (rs_x,rs_y),5,(255,0,0),2)
        cv2.putText(
            frame, 
            live_telemetry, 
            (crop_x1, crop_y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.50, 
            text_color, 
            2, 
            cv2.LINE_AA
        )

    cv2.imshow("Main Camera View", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
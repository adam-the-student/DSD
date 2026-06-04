# Note to self before continuing the project, make the bounding box wider and retrain with that
import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
import threading
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import uvicorn

# --- CONFIGURATION TUNING CORNER ---
CLASSIFIER_IMG_SIZE = (224, 224)  
CONFIDENCE_THRESHOLD = 60  # Minimum % required to change state (stops flickering)

# --- CALIBRATION SETTINGS ---
REAL_SHOULDER_WIDTH_INCHES = 17.0
FOCAL_LENGTH_FACTOR = 318.0  # Calibrated Surface factor

# Distance constraints measured in FEET
MIN_DISTANCE_FEET = 1   # Too close boundary (~18 inches)
MAX_DISTANCE_FEET = 4.5   # Too far back boundary (~54 inches)

# Global Telemetry Storage (Provides cross-thread memory space between YOLO and FastAPI)
telemetry_data = {
    "badge_status": "INITIALIZING...",
    "distance_status": "SEARCHING",
    "estimated_ft": 0.0
}

# --- INITIALIZE NETWORK SERVER LAYER ---
app = FastAPI()

# Serve your custom lightweight static access panel directly over root route
@app.get("/")
async def get_dashboard():
    return FileResponse("index.html")

# WebSocket Endpoint for direct real-time telemetry streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Continuously broadcast global telemetry values to browser clients 20x a second
            await websocket.send_json(telemetry_data)
            await asyncio.sleep(0.05)
    except Exception as e:
        pass
    finally:
        await websocket.close()

# --- BACKGROUND THREAD CORE VISION PIPELINE FUNCTION ---
def run_vision_pipeline():
    global telemetry_data
    
    print("Initializing Stage 1 Pose Joint Tracker...")
    stage1_detector = YOLO("yolov8n-pose.pt") 

    print("Initializing Stage 2 Custom Badge Classifier...")
    stage2_classifier = YOLO("models/badgeClassifier.pt")

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not access the webcam stream.")
        return

    print("\n🚀 Starting Calibrated Distance Two-Stage Pipeline Background Engine.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detector_results = stage1_detector(frame, verbose=False)
        
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, 0, 0
        cropped_chest_frame = None
        distance_status = "OK"
        estimated_ft = 0.0
        local_badge_status = "SCANNING..."

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
        if distance_status != "OK":
            local_badge_status = f"DISTANCE ALERT: {distance_status}"
        elif cropped_chest_frame is not None and cropped_chest_frame.size > 0:
            resized_crop = cv2.resize(cropped_chest_frame, CLASSIFIER_IMG_SIZE)
            classifier_results = stage2_classifier(resized_crop, verbose=False)
            
            class_mapping = classifier_results[0].names  
            top_prediction_id = classifier_results[0].probs.top1
            confidence_percentage = int(classifier_results[0].probs.top1conf.item() * 100)
            predicted_folder_name = class_mapping[top_prediction_id]

            if confidence_percentage >= CONFIDENCE_THRESHOLD:
                if predicted_folder_name == "badge":
                    local_badge_status = f"BADGE DETECTED ({confidence_percentage}%)"
                else:
                    local_badge_status = f"NO BADGE DETECTED ({confidence_percentage}%)"
            else:
                local_badge_status = f"CALCULATING... ({confidence_percentage}%)"

            # Render localized windows on the host machine if running desktop debugging
            cv2.imshow("Chest Patch View", cropped_chest_frame)

        # Update Thread-Safe Memory Map dictionary for WebSocket Broadcaster
        telemetry_data = {
            "badge_status": local_badge_status,
            "distance_status": distance_status,
            "estimated_ft": estimated_ft
        }

        # --- LOCAL HOST MONITOR RENDERING ---
        if crop_x2 > 0 and crop_y2 > 0:
            box_color = (0, 255, 0) if "BADGE DETECTED" in local_badge_status else (0, 0, 255)
            if distance_status != "OK" or "CALCULATING" in local_badge_status:
                box_color = (0, 165, 255)

            cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), box_color, 2)
            cv2.circle(frame, (ls_x, ls_y), 5, (255, 0, 0), 2)
            cv2.circle(frame, (rs_x, rs_y), 5, (255, 0, 0), 2)
            
            live_telemetry = f"{local_badge_status} | Dist: {estimated_ft}ft"
            cv2.putText(frame, live_telemetry, (crop_x1, crop_y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_color, 2, cv2.LINE_AA)

        cv2.imshow("Main Camera View", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Trigger manual dictionary wipe if exiting via OpenCV UI
            telemetry_data = {
                "badge_status": "OFFLINE / SHUTDOWN",
                "distance_status": "OFFLINE",
                "estimated_ft": 0.0
            }
            break

    cap.release()
    cv2.destroyAllWindows()

# --- ENTRY SYSTEM BOOT APEX ---
if __name__ == "__main__":
    try:
        # 1. Spin up the OpenCV + Dual-Stage YOLO vision loop inside an isolation thread
        vision_thread = threading.Thread(target=run_vision_pipeline, daemon=True)
        vision_thread.start()

        # 2. Fire up the local network server over internal port 8000
        # Accessible to anyone on local Wi-Fi at http://<your_pi_ip>:8000
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
        
    except KeyboardInterrupt:
        print("\nStopping network broadcast server...")
        
    finally:
        # Clear out telemetry cache values so remaining socket frames push offline status
        telemetry_data = {
            "badge_status": "OFFLINE / SHUTDOWN",
            "distance_status": "OFFLINE",
            "estimated_ft": 0.0
        }
        print("System safely shut down.")
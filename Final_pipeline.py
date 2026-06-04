# Note to self before continuing the project, make the bounding box wider and retrain with that
import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
import threading
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import uvicorn
import csv
import os
from datetime import datetime
import time
import random

# --- RANDOM SAMPLING CONFIGURATION ---
# 0.005 means a 0.5% chance per valid frame. At 30 FPS, this captures roughly 
# one frame every 6-7 seconds of continuous tracking.
RANDOM_HARVEST_PROBABILITY = 0.005

def harvest_random_frame(crop_frame):
    """Saves a random valid chest crop to disk to build a baseline training distribution."""
    current_time = int(time.time())
    file_name = f"rand_{current_time}.jpg"
    file_path = os.path.join(EDGE_CASE_DIR, file_name)
    
    cv2.imwrite(file_path, crop_frame)
    
    # Log the event to your universal system telemetry CSV file
    log_system_telemetry(
        metric_name="random_harvest", 
        data_value=f"Saved baseline frame: {file_name}", 
        log_level="INFO"
    )
    print(f"🎲 Random Baseline Frame Harvested: {file_name}")

# --- UNIVERSAL SYSTEM LOG CONFIGURATION ---
CSV_FILE_PATH = "system_telemetry_log.csv"

def initialize_universal_logger():
    """Establishes the base data matrix infrastructure if not already present on disk."""
    if not os.path.exists(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            # A completely uniform schema designed to hold any key-value property pair
            writer.writerow(["Timestamp", "Log_Level", "Metric_Name", "Data_Value"])
        print(f"📁 Initialized universal systems logger matrix at: {CSV_FILE_PATH}")

# Boot up the file engine immediately
initialize_universal_logger()

def log_system_telemetry(metric_name: str, data_value, log_level: str = "INFO"):
    """
    Appends any arbitrary tracking property or status flag directly to disk.
    
    Usage Examples:
       log_system_telemetry("badge_status", "BADGE_DETECTED (88%)")
       log_system_telemetry("estimated_ft", 2.4)
       log_system_telemetry("camera_alert", "TOO CLOSE", log_level="WARNING")
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(CSV_FILE_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, log_level.upper(), metric_name, str(data_value)])
        return current_time
    except Exception as e:
        print(f"⚠️ Telemetry Disk Write Failure: {e}")
        return current_time
    
# --- HARVEST ENGINE CONFIGURATION ---
EDGE_CASE_DIR = "harvested_edge_cases"
os.makedirs(EDGE_CASE_DIR, exist_ok=True)

# Cooldown tracker to avoid writing 30 frames a second when standing still
last_harvest_time = 0.0
HARVEST_COOLDOWN_SECONDS = 1.5  # Wait at least 1.5 seconds between frame saves

def harvest_uncertain_frame(crop_frame, confidence_pct):
    """Saves an ambiguous chest crop to disk for future training dataset expansion."""
    global last_harvest_time
    current_time = time.time()
    
    # Enforce cooldown gate so we don't spam the hard drive
    if current_time - last_harvest_time >= HARVEST_COOLDOWN_SECONDS:
        timestamp_id = int(current_time)
        file_name = f"edge_{timestamp_id}_{confidence_pct}.jpg"
        file_path = os.path.join(EDGE_CASE_DIR, file_name)
        
        # Save the crisp image file
        cv2.imwrite(file_path, crop_frame)
        last_harvest_time = current_time
        
        # Log this event into your universal telemetry CSV file!
        log_system_telemetry(
            metric_name="data_harvest", 
            data_value=f"Saved ambiguous frame: {file_name}", 
            log_level="WARNING"
        )
        print(f"📸 Edge Case Harvested: {file_name} saved at {confidence_pct}% confidence.")

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
            # ====== INTEGRATED: RANDOM BASELINE HARVESTER ======
            # Roll the dice on any frame where a person is inside the valid zone
            if random.random() < RANDOM_HARVEST_PROBABILITY:
                harvest_random_frame(cropped_chest_frame)
            # ===================================================
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
                if confidence_percentage >= 40:
                    harvest_uncertain_frame(cropped_chest_frame, confidence_percentage)

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
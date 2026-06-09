# Note to self before continuing the project, make the bounding box wider and retrain with that
import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn
import csv
import os
from datetime import datetime
import time
import random
from picamera2 import Picamera2

# --- GLOBAL LIFECYCLE FLAGS ---
vision_pipeline_active = True  
vision_thread = None           
picam = None                   

# --- RANDOM SAMPLING CONFIGURATION ---
# 🟢 Upped to 5% chance per frame for harvesting a solid baseline production dataset
RANDOM_HARVEST_PROBABILITY = 0.05

# --- UNIVERSAL SYSTEM LOG CONFIGURATION ---
CSV_FILE_PATH = "system_telemetry_log.csv"

def initialize_universal_logger():
    """Establishes the base data matrix infrastructure if not already present on disk."""
    if not os.path.exists(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Log_Level", "Metric_Name", "Data_Value"])
        print(f"📁 Initialized universal systems logger matrix at: {CSV_FILE_PATH}")

initialize_universal_logger()

def log_system_telemetry(metric_name: str, data_value, log_level: str = "INFO"):
    """Appends any arbitrary tracking property or status flag directly to disk."""
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

last_harvest_time = 0.0
HARVEST_COOLDOWN_SECONDS = 1.5  

def harvest_uncertain_frame(crop_frame, confidence_pct):
    """Saves an ambiguous chest crop to disk for future training dataset expansion."""
    global last_harvest_time
    current_time = time.time()
    
    if current_time - last_harvest_time >= HARVEST_COOLDOWN_SECONDS:
        timestamp_id = int(current_time)
        file_name = f"edge_{timestamp_id}_{confidence_pct}.jpg"
        file_path = os.path.join(EDGE_CASE_DIR, file_name)
        
        cv2.imwrite(file_path, crop_frame)
        last_harvest_time = current_time
        
        log_system_telemetry(
            metric_name="data_harvest", 
            data_value=f"Saved ambiguous frame: {file_name}", 
            log_level="WARNING"
        )
        print(f"📸 Edge Case Harvested: {file_name} saved at {confidence_pct}% confidence.")

def harvest_random_frame(crop_frame):
    """Saves a random valid chest crop to disk to build a baseline training distribution."""
    current_time = int(time.time())
    file_name = f"rand_{current_time}.jpg"
    file_path = os.path.join(EDGE_CASE_DIR, file_name)
    
    cv2.imwrite(file_path, crop_frame)
    
    log_system_telemetry(
        metric_name="random_harvest", 
        data_value=f"Saved baseline frame: {file_name}", 
        log_level="INFO"
    )
    print(f"🎲 Random Baseline Frame Harvested: {file_name}")

# --- TRACKING STATE MACHINE LAYER ---
class BadgeTrackerStateMachine:
    def __init__(self):
        self.state = "IDLE"
        self.current_user_max_confidence = 0
        self.current_user_final_decision = "UNKNOWN"
        self.frames_since_last_seen = 0
        self.max_lost_frames = 15  

    def update_presence(self, person_detected: bool):
        """Tracks arrivals and departures to reset or trigger logging summary events."""
        if person_detected:
            self.frames_since_last_seen = 0
            if self.state == "IDLE":
                self.state = "TRACKING"
                print("👤 Person entered tracking zone.")
                log_system_telemetry("state_machine", "Person arrived", "INFO")
        else:
            if self.state != "IDLE":
                self.frames_since_last_seen += 1
                if self.frames_since_last_seen >= self.max_lost_frames:
                    self.trigger_departure_event()

    def update_evaluation(self, decision: str, confidence: int):
        """Updates internal memory metrics with the highest confidence predictions."""
        if self.state in ["TRACKING", "EVALUATING"]:
            self.state = "EVALUATING"
            
            if confidence > self.current_user_max_confidence:
                self.current_user_max_confidence = confidence
                self.current_user_final_decision = decision

            if confidence >= 85:
                self.state = "LOCKED"
                print(f"🔒 State LOCKED: {decision} confirmed at {confidence}%.")

    def trigger_departure_event(self):
        """Logs a clean, consolidated interaction event to the telemetry CSV upon departure."""
        log_level = "INFO" if "BADGE DETECTED" in self.current_user_final_decision else "ERROR"
        
        log_system_telemetry(
            metric_name="wearer_departure_summary",
            data_value=f"Decision: {self.current_user_final_decision} | Max Conf: {self.current_user_max_confidence}%",
            log_level=log_level
        )
        print(f"🚶 Person departed. Final Summary logged: {self.current_user_final_decision} ({self.current_user_max_confidence}%)")
        
        self.state = "IDLE"
        self.current_user_max_confidence = 0
        self.current_user_final_decision = "UNKNOWN"
        self.frames_since_last_seen = 0

# --- CONFIGURATION TUNING CORNER ---
CLASSIFIER_IMG_SIZE = (224, 224)  
CONFIDENCE_THRESHOLD = 60  

# --- CALIBRATION SETTINGS ---
REAL_SHOULDER_WIDTH_INCHES = 17.0
FOCAL_LENGTH_FACTOR = 1600  

MIN_DISTANCE_FEET = 1   
MAX_DISTANCE_FEET = 4.5   

telemetry_data = {
    "badge_status": "INITIALIZING...",
    "distance_status": "SEARCHING",
    "estimated_ft": 0.0
}

# --- INITIALIZE NETWORK SERVER LAYER ---
app = FastAPI()

@app.get("/")
async def get_dashboard():
    return FileResponse("Index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(telemetry_data)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"📡 WebSocket Link Stream Reset: {e}")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass

# --- BACKGROUND THREAD CORE VISION PIPELINE FUNCTION ---
def run_vision_pipeline():
    global telemetry_data, picam
    
    tracker = BadgeTrackerStateMachine()
    
    print("Initializing Stage 1 Pose Joint Tracker...")
    stage1_detector = YOLO("yolov8n-pose.pt") 

    print("Initializing Stage 2 Custom Badge Classifier...")
    stage2_classifier = YOLO("models/badgeClassifier.pt")

    print("\n🚀 Starting Calibrated Distance Two-Stage Pipeline Background Engine.")

    while True:
        try:
            raw_frame = picam.capture_array()
            frame = cv2.flip(raw_frame, -1)
        except Exception as e:
            print(f"Frame capture interruption: {e}")
            time.sleep(0.1)
            continue

        detector_results = stage1_detector(frame, verbose=False, imgsz=320)
        
        person_in_frame = False
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, 0, 0
        cropped_chest_frame = None
        distance_status = "SEARCHING"
        estimated_ft = 0.0
        local_badge_status = "SCANNING..."
        confidence_percentage = 0

        for result in detector_results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                kpts = result.keypoints.xy[0]
                left_shoulder = kpts[5]
                right_shoulder = kpts[6]
                
                if left_shoulder[0] > 0 and right_shoulder[0] > 0:
                    person_in_frame = True
                    ls_x, ls_y = int(left_shoulder[0]), int(left_shoulder[1])
                    rs_x, rs_y = int(right_shoulder[0]), int(right_shoulder[1])
                    
                    shoulder_width_pixels = abs(ls_x - rs_x)
                    
                    if shoulder_width_pixels > 0:
                        distance_inches = (REAL_SHOULDER_WIDTH_INCHES * FOCAL_LENGTH_FACTOR) / shoulder_width_pixels
                        estimated_ft = round(distance_inches / 12.0, 1)
                    
                    if estimated_ft > MAX_DISTANCE_FEET:
                        distance_status = "TOO FAR"
                    elif estimated_ft < MIN_DISTANCE_FEET:
                        distance_status = "TOO CLOSE"
                    else:
                        distance_status = "OK"

                    # 🔒 UNTOUCHED: Keeping original tight bounding box multipliers
                    crop_x1 = min(ls_x, rs_x) + int(shoulder_width_pixels * 0.20)
                    crop_x2 = max(ls_x, rs_x) - int(shoulder_width_pixels * 0.20)
                    crop_y1 = min(ls_y, rs_y) - int(shoulder_width_pixels * 0.25)
                    crop_y2 = min(ls_y, rs_y) + int(shoulder_width_pixels * 0.45)
                    
                    crop_x1 = max(0, crop_x1)
                    crop_y1 = max(0, crop_y1)
                    crop_x2 = min(frame.shape[1], crop_x2)
                    crop_y2 = min(frame.shape[0], crop_y2)
                    
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        cropped_chest_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    break

        tracker.update_presence(person_in_frame)

        if tracker.state == "LOCKED":
            local_badge_status = tracker.current_user_final_decision
        
        elif distance_status == "OK" and cropped_chest_frame is not None and cropped_chest_frame.size > 0:
            resized_crop = cv2.resize(cropped_chest_frame, CLASSIFIER_IMG_SIZE)
            
            # ====== ACTIVE PIPELINE BASELINE PRODUCTION HARVESTER ======
            if random.random() < RANDOM_HARVEST_PROBABILITY:
                harvest_random_frame(cropped_chest_frame)
            # ===========================================================
            
            classifier_results = stage2_classifier(resized_crop, verbose=False, imgsz=224)
            
            class_mapping = classifier_results[0].names  
            top_prediction_id = classifier_results[0].probs.top1
            confidence_percentage = int(classifier_results[0].probs.top1conf.item() * 100)
            predicted_folder_name = class_mapping[top_prediction_id]

            if confidence_percentage >= CONFIDENCE_THRESHOLD:
                if predicted_folder_name == "badge":
                    local_badge_status = f"BADGE DETECTED"
                else:
                    local_badge_status = f"NO BADGE DETECTED"
            else:
                local_badge_status = f"CALCULATING..."
                if confidence_percentage >= 40:
                    harvest_uncertain_frame(cropped_chest_frame, confidence_percentage)

            if tracker.state != "LOCKED" and local_badge_status in ["BADGE DETECTED", "NO BADGE DETECTED"]:
                tracker.update_evaluation(local_badge_status, confidence_percentage)

            cv2.imshow("Chest Patch View", cropped_chest_frame)

        telemetry_data = {
            "badge_status": f"{local_badge_status} ({tracker.current_user_max_confidence}%)" if tracker.state != "IDLE" else "SCANNING...",
            "distance_status": distance_status,
            "estimated_ft": estimated_ft if person_in_frame else 0.0
        }

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
            break

    cv2.destroyAllWindows()
    print("🎥 Vision pipeline loop processing suspended.")

# --- ENTRY SYSTEM BOOT APEX ---
if __name__ == "__main__":
    print("Initializing Master Hardware Camera Stream...")
    picam = Picamera2()
    config = picam.create_video_configuration()
    config["main"]["size"] = (1920, 1080)
    config["main"]["format"] = "RGB888"
    config["controls"]["FrameRate"] = 30
    picam.configure(config)
    picam.start()
    print("🔌 Native Raspberry Pi 5 Hardware Camera Stream Initialized.")

    try:
        vision_thread = threading.Thread(target=run_vision_pipeline, daemon=True)
        vision_thread.start()

        print("Starting FastAPI Dashboard Gateway Server...")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
        
    except KeyboardInterrupt:
        print("\nStopping network broadcast server...")
        
    finally:
        print("Safely releasing camera hardware channels...")
        try:
            picam.stop()
            picam.close()
        except:
            pass
            
        telemetry_data = {
            "badge_status": "OFFLINE / SHUTDOWN",
            "distance_status": "OFFLINE",
            "estimated_ft": 0.0
        }
        print("System safely shut down.")
import os
import sys
IS_SSH_SESSION = "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ

if IS_SSH_SESSION:
    os.environ["OPENCV_HEADLESS"] = "1"
import csv
import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import time
import random
from picamera2 import Picamera2
import libcamera
from libcamera import Transform
import queue

# Import utility layers
from telemetry import log_system_telemetry, get_daily_csv_path, initialize_universal_logger
from privacy import save_anonymized_and_encrypted_frame

# Initialize dynamic telemetry columns on startup
initialize_universal_logger()

# --- GLOBAL LIFECYCLE FLAGS ---
vision_pipeline_active = True  
vision_thread = None           
picam = None                   
ALLOW_GUI_DISPLAY = not IS_SSH_SESSION  

# --- RANDOM SAMPLING CONFIGURATION ---
RANDOM_HARVEST_PROBABILITY = 0.01

# --- PERFORMANCE HARVEST THROTTLES ---
last_harvest_time = 0.0
HARVEST_COOLDOWN_SECONDS = 5.0  

# --- GLOBAL ASYNC EVENT BRIDGE ---
main_event_loop = None
event_queue = queue.Queue()

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
                    return True # Signifies departure needs processing
        return False

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

    def trigger_departure_event(self, frame, detector_results):
        """Logs interaction events quietly and drops the payload into the cross-thread queue."""
        global event_queue
        from datetime import datetime

        # Dynamic profile parsing for the log string file layout
        profile_string = self.current_user_final_decision.title()
        log_level = "INFO" if "NO" not in self.current_user_final_decision.upper() and self.current_user_final_decision != "UNKNOWN" else "ERROR"
        
        # 1. Update disk backup ledger
        log_system_telemetry(
            metric_name="wearer_departure_summary",
            data_value=f"Decision: {self.current_user_final_decision} | Max Conf: {self.current_user_max_confidence}%",
            log_level=log_level
        )
        
        # 2. Package event data for the web layout
        payload = {
            "is_entry_event": True,
            "time": datetime.now().strftime("%H:%M:%S"),
            "profile": profile_string,
            "confidence": f"{self.current_user_max_confidence}%",
            "proximity": "3.5 ft"
        }
        event_queue.put(payload)
                    
        print(f"🚶 Person departed. Summary logged: {profile_string}")
        
        # Reset State Engine
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
MAX_DISTANCE_FEET = 20  

telemetry_data = {
    "badge_status": "INITIALIZING...",
    "distance_status": "SEARCHING",
    "estimated_ft": 0.0,
    "is_entry_event": False
}

connected_clients = set()

# --- INITIALIZE NETWORK SERVER LAYER ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/")
async def get_dashboard():
    return FileResponse("web/Index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global main_event_loop, telemetry_data
    await websocket.accept()
    connected_clients.add(websocket)
    
    main_event_loop = asyncio.get_running_loop()
    
    startup_history = []
    target_csv = get_daily_csv_path()
    
    if os.path.exists(target_csv) and os.path.getsize(target_csv) > 0:
        try:
            with open(target_csv, mode='r', errors='ignore') as file:
                explicit_headers = ["timestamp", "log_level", "metric_name", "data_value"]
                reader = csv.DictReader(file, fieldnames=explicit_headers)
                
                for row in reader:
                    if not row.get("metric_name") or not row.get("data_value"):
                        continue
                        
                    metric_name = row["metric_name"].strip()
                    raw_val = row["data_value"].strip()
                    
                    if metric_name == "wearer_departure_summary":
                        if "Decision:" in raw_val:
                            # Extract profile label securely without assuming folder names
                            parsed_decision = raw_val.split("Decision:")[-1].split("|")[0].strip()
                            
                            if "UNKNOWN" in raw_val:
                                profile_label = "❌ Unknown Status"
                            else:
                                profile_label = parsed_decision.title()
                            
                            conf = "N/A"
                            if "Max Conf:" in raw_val:
                                conf = raw_val.split("Max Conf:")[-1].strip()
                                
                            time_stamp = row.get("timestamp", "").strip()
                            if " " in time_stamp:
                                time_stamp = time_stamp.split(" ")[-1]
                                
                            startup_history.append({
                                "time": time_stamp,
                                "profile": profile_label,
                                "confidence": conf,
                                "proximity": "3.5 ft"
                            })
        except Exception as e:
            pass

    if startup_history:
        await websocket.send_json({
            "is_startup_history": True,
            "history": startup_history
        })

    try:
        while True:
            global event_queue
            
            try:
                while not event_queue.empty():
                    live_alert = event_queue.get_nowait()
                    await websocket.send_json(live_alert)
                    event_queue.task_done()
                    await asyncio.sleep(0.01)
            except queue.Empty:
                pass
            
            current_snapshot = dict(telemetry_data)
            await websocket.send_json(current_snapshot)
                
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.remove(websocket)

def run_vision_pipeline():
    global telemetry_data, picam, ALLOW_GUI_DISPLAY, last_harvest_time
    
    tracker = BadgeTrackerStateMachine()
    
    print("Initializing Stage 1 Pose Joint Tracker via NPU...")
    stage1_detector = YOLO("yolov8n-pose")

    print("Initializing Stage 2 Custom Model Zoo Classifier...")
    stage2_classifier = YOLO("models/dosClassifier.pt")

    print("\n🚀 Starting Calibrated Distance Two-Stage Pipeline Background Engine.")

    while True:
        try:
            raw_frame = picam.capture_array()
            frame = cv2.resize(raw_frame, (640, 480))
            frame = np.ascontiguousarray(frame)
        except Exception as e:
            print(f"Frame capture interruption: {e}")
            time.sleep(0.1)
            continue

        detector_results = stage1_detector(frame, verbose=False, imgsz=640)
        
        person_in_frame = False
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, 0, 0
        cropped_chest_frame = None
        distance_status = "SEARCHING"
        estimated_ft = 0.0
        local_badge_status = "SCANNING..."
        
        # Dictionary container to hold dynamic classes and scores without hardcoding
        active_probabilities = {}
        top_confidence = 0
        predicted_folder_name = "UNKNOWN"

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

                    crop_x1 = min(ls_x, rs_x) + int(shoulder_width_pixels * 0.20)
                    crop_x2 = max(ls_x, rs_x) - int(shoulder_width_pixels * 0.20)
                    crop_y1 = min(ls_y, rs_y) - int(shoulder_width_pixels * 0.25)
                    crop_y2 = min(ls_y, rs_y) + int(shoulder_width_pixels * 0.45)
                    
                    crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
                    crop_x2, crop_y2 = min(frame.shape[1], crop_x2), min(frame.shape[0], crop_y2)
                    
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        cropped_chest_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    break

        has_departed = tracker.update_presence(person_in_frame)
        if has_departed:
            tracker.trigger_departure_event(frame, detector_results)

        if tracker.state == "LOCKED":
            local_badge_status = tracker.current_user_final_decision
        
        # 1. INDEPENDENT RANDOM HARVESTER 
        if distance_status == "OK" and cropped_chest_frame is not None and cropped_chest_frame.size > 0:
            if random.random() < RANDOM_HARVEST_PROBABILITY:
                identified_as_tech = False 
                enc_file = save_anonymized_and_encrypted_frame(
                    cropped_chest_frame, detector_results, prefix="rand", 
                    crop_offsets=(crop_x1, crop_y1), is_rad_tech=identified_as_tech
                )
                if enc_file:
                    log_system_telemetry("random_harvest", f"Saved baseline frame: {enc_file}", "INFO")

        # 2. FULLY DYNAMIC DECODER BLOCK (NO HARDCODED STRINGS)
        if tracker.state != "LOCKED" and distance_status == "OK" and cropped_chest_frame is not None and cropped_chest_frame.size > 0:
            resized_crop = cv2.resize(cropped_chest_frame, CLASSIFIER_IMG_SIZE)
            classifier_results = stage2_classifier(resized_crop, verbose=False, imgsz=224)
            
            # Pull dataset mappings and values dynamically directly from model properties
            class_mapping = classifier_results[0].names  
            probs_tensor = classifier_results[0].probs.data.cpu().numpy()
            
            # Automatically populate all outputs into a dictionary container
            for idx, class_name in class_mapping.items():
                active_probabilities[class_name.upper()] = int(probs_tensor[idx] * 100)

            top_prediction_id = classifier_results[0].probs.top1
            predicted_folder_name = class_mapping[top_prediction_id].upper()
            top_confidence = int(classifier_results[0].probs.top1conf.item() * 100)

            # Assign status using the folder name dynamically
            if top_confidence >= CONFIDENCE_THRESHOLD:
                local_badge_status = f"{predicted_folder_name} DETECTED"
            else:
                local_badge_status = "CALCULATING..."
                current_time = time.time()
                if top_confidence >= 40 and (current_time - last_harvest_time >= HARVEST_COOLDOWN_SECONDS):
                    identified_as_tech = False
                    enc_file = save_anonymized_and_encrypted_frame(
                        cropped_chest_frame, detector_results, prefix="edge", 
                        extra_suffix=str(top_confidence), crop_offsets=(crop_x1, crop_y1),
                        is_rad_tech=identified_as_tech
                    )
                    if enc_file:
                        last_harvest_time = current_time
                        log_system_telemetry("data_harvest", f"Saved ambiguous frame: {enc_file}", "WARNING")
            
            if local_badge_status != "CALCULATING...":
                tracker.update_evaluation(local_badge_status, top_confidence)

        has_active_event = telemetry_data.get("is_entry_event", False)
        
        evt_time = telemetry_data.get("time", None)
        evt_prof = telemetry_data.get("profile", None)
        evt_conf = telemetry_data.get("confidence", None)
        evt_prox = telemetry_data.get("proximity", None)

        # Build dynamic HUD score readouts loop securely
        score_readouts = " | ".join([f"{k}: {v}%" for k, v in active_probabilities.items()])
        HUD_badge_string = f"{local_badge_status} ({score_readouts})" if tracker.state != "IDLE" else "SCANNING..."
        
        telemetry_data.update({
            "badge_status": HUD_badge_string,
            "distance_status": distance_status,
            "estimated_ft": estimated_ft if person_in_frame else 0.0,
            "is_entry_event": has_active_event
        })
        
        if has_active_event:
            telemetry_data["time"] = evt_time
            telemetry_data["profile"] = evt_prof
            telemetry_data["confidence"] = evt_conf
            telemetry_data["proximity"] = evt_prox

        # --- ONSCREEN RENDERING OVERLAYS ---
        if crop_x2 > 0 and crop_y2 > 0 and ALLOW_GUI_DISPLAY:  
            # Select bounding box boundaries colors dynamically
            box_color = (0, 255, 0) if "NO" not in local_badge_status and "DETECTED" in local_badge_status else (0, 0, 255)
            if distance_status != "OK" or "CALCULATING" in local_badge_status:
                box_color = (0, 165, 255)

            cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), box_color, 2)
            cv2.circle(frame, (ls_x, ls_y), 5, (255, 0, 0), 2)
            cv2.circle(frame, (rs_x, rs_y), 5, (255, 0, 0), 2)
            
            # Dynamic line renderings
            text_line1 = f"STATUS: {local_badge_status}"
            text_line2 = score_readouts
            text_line3 = f"PROXIMITY: {estimated_ft} ft"
            
            cv2.putText(frame, text_line1, (crop_x1, crop_y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2, cv2.LINE_AA)
            cv2.putText(frame, text_line2, (crop_x1, crop_y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, text_line3, (crop_x1, crop_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA)

        if ALLOW_GUI_DISPLAY:
            try:
                cv2.imshow("Main Camera View", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                print("\n🖥️ Headless mode active. Disabling GUI display canvas windows.")
                ALLOW_GUI_DISPLAY = False
        else:
            time.sleep(0.001)

    try:
        cv2.destroyAllWindows()
    except:
        pass

# --- SYSTEM INITIALIZATION BOOT ---
if __name__ == "__main__":
    print("Initializing Master Hardware Camera Stream...")
    picam = Picamera2()
    
    config = picam.create_video_configuration(
        {"size": (1280, 960), "format": "RGB888"}
    )
    config["main"]["buffer_count"] = 6
    config["controls"]["FrameRate"] = 60
    
    picam.configure(config)
    picam.start()
    print("🔌 Optimized 60FPS Native Hardware Camera Stream Initialized.")

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
            "estimated_ft": 0.0,
            "is_entry_event": False
        }
        print("System safely shut down.")
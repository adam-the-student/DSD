# Final_pipeline.py
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

        if "NO BADGE" in self.current_user_final_decision or self.current_user_final_decision == "UNKNOWN":
            log_level = "ERROR"
            profile_string = "No Badge"
        else:
            log_level = "INFO"
            profile_string = "Valid Badge Entry"
        
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
                    
        # 🟢 Clean, original terminal message layout style
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
                            if "BADGE DETECTED" in raw_val and "NO BADGE" not in raw_val:
                                profile_label = "Valid Badge Entry"
                            else:
                                profile_label = "No Badge"
                            
                            if "UNKNOWN" in raw_val:
                                profile_label = "❌ Unknown Status"
                            
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
            
            # Flush live milestone packets quietly
            try:
                while not event_queue.empty():
                    live_alert = event_queue.get_nowait()
                    await websocket.send_json(live_alert)
                    event_queue.task_done()
                    await asyncio.sleep(0.01)
            except queue.Empty:
                pass
            
            # Send continuous HUD updates
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
    
    print("Initializing Stage 1 Pose Joint Tracker...")
    stage1_detector = YOLO("yolov8n-pose.pt") 

    print("Initializing Stage 2 Custom Badge Classifier...")
    stage2_classifier = YOLO("models/badgeClassifier.pt")

    print("\n🚀 Starting Calibrated Distance Two-Stage Pipeline Background Engine.")

    while True:
        try:
            raw_frame = picam.capture_array()
            frame = raw_frame
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

                    crop_x1 = min(ls_x, rs_x) + int(shoulder_width_pixels * 0.20)
                    crop_x2 = max(ls_x, rs_x) - int(shoulder_width_pixels * 0.20)
                    crop_y1 = min(ls_y, rs_y) - int(shoulder_width_pixels * 0.25)
                    crop_y2 = min(ls_y, rs_y) + int(shoulder_width_pixels * 0.45)
                    
                    crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
                    crop_x2, crop_y2 = min(frame.shape[1], crop_x2), min(frame.shape[0], crop_y2)
                    
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        cropped_chest_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    break

        # Check departure pulse
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
                    tech_label = " [Tech Bypass]" if identified_as_tech else ""
                    print(f"🎲 Random Baseline Crop Harvested: {enc_file}{tech_label}")

        # 2. INDEPENDENT CLASSIFIER BLOCK
        if tracker.state != "LOCKED" and distance_status == "OK" and cropped_chest_frame is not None and cropped_chest_frame.size > 0:
            resized_crop = cv2.resize(cropped_chest_frame, CLASSIFIER_IMG_SIZE)
            classifier_results = stage2_classifier(resized_crop, verbose=False, imgsz=224)
            
            class_mapping = classifier_results[0].names  
            top_prediction_id = classifier_results[0].probs.top1
            confidence_percentage = int(classifier_results[0].probs.top1conf.item() * 100)
            predicted_folder_name = class_mapping[top_prediction_id]

            if confidence_percentage >= CONFIDENCE_THRESHOLD:
                local_badge_status = "BADGE DETECTED" if predicted_folder_name == "badge" else "NO BADGE DETECTED"
            else:
                local_badge_status = "CALCULATING..."
                current_time = time.time()
                if confidence_percentage >= 40 and (current_time - last_harvest_time >= HARVEST_COOLDOWN_SECONDS):
                    identified_as_tech = False
                    enc_file = save_anonymized_and_encrypted_frame(
                        cropped_chest_frame, detector_results, prefix="edge", 
                        extra_suffix=str(confidence_percentage), crop_offsets=(crop_x1, crop_y1),
                        is_rad_tech=identified_as_tech
                    )
                    if enc_file:
                        last_harvest_time = current_time
                        log_system_telemetry("data_harvest", f"Saved ambiguous frame: {enc_file}", "WARNING")
                        tech_label = " [Tech Bypass]" if identified_as_tech else ""
                        print(f"📸 Edge Case Crop Harvested: {enc_file} saved securely ({confidence_percentage}% confidence){tech_label}")
            
            if local_badge_status in ["BADGE DETECTED", "NO BADGE DETECTED"]:
                tracker.update_evaluation(local_badge_status, confidence_percentage)
            else:
                local_badge_status = "CALCULATING..."

        # 🟢 FIX: Keep the event active across frame updates so the WebSocket has time to catch it!
        has_active_event = telemetry_data.get("is_entry_event", False)
        
        # Pull transactional values if they are alive in dictionary memory right now
        evt_time = telemetry_data.get("time", None)
        evt_prof = telemetry_data.get("profile", None)
        evt_conf = telemetry_data.get("confidence", None)
        evt_prox = telemetry_data.get("proximity", None)

        telemetry_data.update({
            "badge_status": f"{local_badge_status} ({tracker.current_user_max_confidence}%)" if tracker.state != "IDLE" else "SCANNING...",
            "distance_status": distance_status,
            "estimated_ft": estimated_ft if person_in_frame else 0.0,
            "is_entry_event": has_active_event
        })
        
        # If an event is currently pending transmission, lock its keys back onto the frame payload state
        if has_active_event:
            telemetry_data["time"] = evt_time
            telemetry_data["profile"] = evt_prof
            telemetry_data["confidence"] = evt_conf
            telemetry_data["proximity"] = evt_prox

        if crop_x2 > 0 and crop_y2 > 0 and ALLOW_GUI_DISPLAY:  
            box_color = (0, 255, 0) if "BADGE DETECTED" in local_badge_status else (0, 0, 255)
            if distance_status != "OK" or "CALCULATING" in local_badge_status:
                box_color = (0, 165, 255)

            cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), box_color, 2)
            cv2.circle(frame, (ls_x, ls_y), 5, (255, 0, 0), 2)
            cv2.circle(frame, (rs_x, rs_y), 5, (255, 0, 0), 2)
            
            live_telemetry = f"{local_badge_status} | Dist: {estimated_ft}ft"
            cv2.putText(frame, live_telemetry, (crop_x1, crop_y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_color, 2, cv2.LINE_AA)

        if ALLOW_GUI_DISPLAY:
            try:
                cv2.imshow("Main Camera View", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                print("\n🖥️ No display environment detected (Headless SSH Session). Disabling window rendering layout.")
                ALLOW_GUI_DISPLAY = False
        else:
            time.sleep(0.001)

    try:
        cv2.destroyAllWindows()
    except:
        pass

# --- ENTRY SYSTEM BOOT APEX ---
if __name__ == "__main__":
    import json
    print("Initializing Master Hardware Camera Stream...")
    picam = Picamera2()
    
    config = picam.create_video_configuration(
        {"size": (1280, 720), "format": "RGB888"}, 
        transform=Transform(hflip=True, vflip=True)
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
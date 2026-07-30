import os
import sys
I_SSH_SESSION = "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ

if I_SSH_SESSION:
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
import json
import random
from picamera2 import Picamera2
import libcamera
from libcamera import Transform
import queue

# Import native Hailo platform tools aligned with your data collector sandbox
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType

# Import custom helper classes
from telemetry import log_system_telemetry, get_daily_csv_path, initialize_universal_logger
from privacy import save_anonymized_and_encrypted_frame
from tamagotchi import TamagotchiEngine

# Initialize dynamic telemetry columns on startup
initialize_universal_logger()

# --- GLOBAL LIFECYCLE FLAGS ---
vision_pipeline_active = True  
vision_thread = None           
picam = None                   
ALLOW_GUI_DISPLAY = not I_SSH_SESSION  

# --- RANDOM SAMPLING CONFIGURATION ---
RANDOM_HARVEST_PROBABILITY = 0.10  

# --- PERFORMANCE HARVEST THROTTLES ---
last_harvest_time = 0.0
HARVEST_COOLDOWN_SECONDS = 5.0  

# --- GLOBAL ASYNC EVENT BRIDGE ---
main_event_loop = None
event_queue = queue.Queue()
pet = None  # Global anchor reference shared with network threads

import traceback

def log_thread_crashes(func):
    """Decorator to catch and log any crash inside the decorated function."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_details = traceback.format_exc()
            error_msg = f"Fatal Crash in {func.__name__}: {str(e)} | Line: {error_details.splitlines()[-2]}"
            
            # Log straight to your telemetry CSV
            try:
                from telemetry import log_system_telemetry
                log_system_telemetry("thread_crash_event", error_msg, "CRITICAL")
            except Exception:
                pass
                
            print(f"\n💥 CRITICAL CRASH IN {func.__name__}:\n{error_details}")
    return wrapper

# --- TRACKING STATE MACHINE LAYER ---
class BadgeTrackerStateMachine:
    def __init__(self):
        self.state = "IDLE"
        self.current_user_max_confidence = 0
        self.current_user_final_decision = "UNKNOWN"
        self.frames_since_last_seen = 0
        self.max_lost_frames = 15  
        self.current_user_last_distance = 0.0

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

    def update_evaluation(self, decision: str, confidence: int, estimated_ft, pet_engine_reference):
        """Updates internal memory metrics and handles dynamic updates even if locked."""
        if self.state in ["TRACKING", "EVALUATING", "LOCKED"]:
            
            # 🟢 UPGRADE ALLOWED: If we were unbadged or unknown, let a high-confidence badge detection override it
            is_currently_violating = "NO" in self.current_user_final_decision.upper() or self.current_user_final_decision == "UNKNOWN"
            is_new_read_compliant = "NO" not in decision.upper() and "DETECTED" in decision.upper()

            if self.state == "LOCKED" and is_currently_violating and is_new_read_compliant and confidence >= 60:
                print(f"🔄 compliance status upgraded live: Shifted to {decision} at {confidence}%.")
                self.current_user_max_confidence = confidence
                self.current_user_final_decision = decision
            
            # Standard initial locking behavior for a fresh tracking session
            elif self.state != "LOCKED":
                self.state = "EVALUATING"
                self.current_user_last_distance = estimated_ft
                
                if confidence > self.current_user_max_confidence:
                    self.current_user_max_confidence = confidence
                    self.current_user_final_decision = decision

                if confidence >= 60:
                    self.state = "LOCKED"
                    print(f"🔒 State LOCKED: {decision} confirmed at {confidence}%.")
                    
                    if "NO" not in decision.upper() and decision != "UNKNOWN":
                        pet_engine_reference.register_successful_feeding()

        
    def trigger_departure_event(self, frame):
        """Logs interaction events quietly and drops the payload into the cross-thread queue."""
        global event_queue, pet, telemetry_data, connected_clients, main_event_loop
        from datetime import datetime

        profile_string = self.current_user_final_decision.title()
        log_level = "INFO" if "NO" not in self.current_user_final_decision.upper() and self.current_user_final_decision != "UNKNOWN" else "ERROR"
        
        saved_dist = self.current_user_last_distance

        # 1. Update disk backup ledger using internal state machine memory slot
        log_system_telemetry(
            metric_name="wearer_departure_summary",
            data_value=f"Decision: {self.current_user_final_decision} | Max Conf: {self.current_user_max_confidence}% | Distance: {saved_dist} ft",
            log_level=log_level
        )
        
        # 2. Package event data dynamically for the plaintext web layout
        payload = {
            "is_entry_event": True,
            "time": datetime.now().strftime("%H:%M:%S"),
            "profile": profile_string,
            "confidence": f"{self.current_user_max_confidence}%",
            "proximity": f"{saved_dist} ft"
        }
        event_queue.put(payload)
                    
        print(f"🚶 Person departed. Summary logged: {profile_string} at {saved_dist} ft")
        
        if pet is not None:
            pet.reset_user()
            telemetry_data["daily_goal"] = pet.DAILY_GOAL
            telemetry_data["successful_feedings"] = pet.successful_feedings
            telemetry_data["pet_status"] = pet.get_status()
            
            if main_event_loop is not None and connected_clients:
                broadcast_payload = dict(telemetry_data)
                for client in list(connected_clients):
                    try:
                        asyncio.run_coroutine_threadsafe(
                            client.send_json(broadcast_payload), 
                            main_event_loop
                        )
                    except Exception:
                        pass

        # Reset State Engine
        self.state = "IDLE"
        self.current_user_max_confidence = 0
        self.current_user_final_decision = "UNKNOWN"
        self.frames_since_last_seen = 0

# --- CONFIGURATION TUNING CORNER ---
IMAGE_DIRECTORY = "harvested_edge_cases"
CLASSIFIER_IMG_SIZE = (224, 224)  
CONFIDENCE_THRESHOLD = 60  

# --- CALIBRATION SETTINGS ---
REAL_SHOULDER_WIDTH_INCHES = 17.0
FOCAL_LENGTH_FACTOR = 318  

MIN_DISTANCE_FEET = 0.5    
MAX_DISTANCE_FEET = 7  

telemetry_data = {
    "badge_status": "INITIALIZING...",
    "distance_status": "SEARCHING",
    "estimated_ft": 0.0,
    "is_entry_event": False,
    "pet_status": "UNKNOWN",
    "successful_feedings": 0,
    "daily_goal": 5
}

connected_clients = set()

# --- INITIALIZE NETWORK SERVER LAYER ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/")
async def get_dashboard():
    return FileResponse("web/Index.html")

@app.get("/cat")
async def get_cat_dashboard():
    return FileResponse("web/Cat.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global main_event_loop, telemetry_data, pet
    await websocket.accept()
    connected_clients.add(websocket)
    
    main_event_loop = asyncio.get_running_loop()
    target_csv = get_daily_csv_path()
    
    if os.path.exists(target_csv) and os.path.getsize(target_csv) > 0:
        try:
            with open(target_csv, mode='r', encoding='utf-8', errors='ignore') as file:
                raw_history_text = file.read()
                
            await websocket.send_json({
                "is_startup_history": True,
                "raw_text": raw_history_text
            })
        except Exception as e:
            print(f"⚠️ Startup history log read error: {e}")
    else:
        await websocket.send_json({
            "is_startup_history": True,
            "raw_text": "--- Telemetry log file is currently empty or uninitialized on disk ---"
        })

    try:
        while True:
            global event_queue
            
            try:
                raw_incoming = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if raw_incoming:
                    parsed_incoming = json.loads(raw_incoming)
                    if "set_daily_goal" in parsed_incoming:
                        new_target = int(parsed_incoming["set_daily_goal"])
                        telemetry_data["daily_goal"] = new_target
                        if pet is not None:
                            pet.DAILY_GOAL = new_target
                            pet.save_state_to_disk()
                        print(f"⚙️ MANAGER UPDATE: Daily target shifted to {new_target} and saved to disk.")
            except asyncio.TimeoutError:
                pass 
            except (WebSocketDisconnect, RuntimeError):
                raise 
            except Exception as e:
                print(f"⚠️ Internal manager parsing error: {e}")
            
            try:
                while not event_queue.empty():
                    live_alert = event_queue.get_nowait()
                    try:
                        await websocket.send_json(live_alert)
                    except (RuntimeError, Exception):
                        pass
                    event_queue.task_done()
                    await asyncio.sleep(0.01)
            except queue.Empty:
                pass
            
            if pet is not None:
                telemetry_data["daily_goal"] = pet.DAILY_GOAL
                telemetry_data["successful_feedings"] = pet.successful_feedings
                telemetry_data["pet_status"] = pet.get_status()

            await websocket.send_json(dict(telemetry_data))
            await asyncio.sleep(0.05)
            
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        connected_clients.discard(websocket)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

@log_thread_crashes
def run_vision_pipeline():
    global telemetry_data, picam, ALLOW_GUI_DISPLAY, last_harvest_time, pet
    
    tracker = BadgeTrackerStateMachine()
    pet = TamagotchiEngine(daily_goal=5)
    
    fps_frame_timestamps = []
    current_calculated_fps = 0.0

    current_cycle_day = time.strftime('%Y-%m-%d')
    telemetry_data["daily_goal"] = pet.DAILY_GOAL

    # --- STAGE 1: NATIVE NPU SETUP ---
    model_path = "models/yolov8s-pose-h10.hef"
    print(f"Loading hardware network: {model_path} onto AI HAT+...")

    vdevice_params = VDevice.create_params()
    vdevice_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    target_device = VDevice(vdevice_params)
    infer_model = target_device.create_infer_model(model_path)

    infer_model.input().set_format_type(FormatType.UINT8)
    for output_name in infer_model.output_names:
        infer_model.output(output_name).set_format_type(FormatType.FLOAT32)

    input_name = infer_model.input_names[0]

    layer_pairs = [
        {"score": "yolov8s_pose/conv71", "keypoint": "yolov8s_pose/conv72", "stride": 32},  
        {"score": "yolov8s_pose/conv58", "keypoint": "yolov8s_pose/conv59", "stride": 16},  
        {"score": "yolov8s_pose/conv44", "keypoint": "yolov8s_pose/conv45", "stride": 8}    
    ]

    # --- STAGE 2: CPU CLASSIFIER SETUP ---
    print("Initializing Stage 2 Custom Model Zoo Classifier via CPU...")
    stage2_classifier = YOLO("models/dosClassifier3.pt")

    print("\n🚀 Starting Calibrated Distance Two-Stage Pipeline Background Engine.")

    with infer_model.configure() as configured_infer_model:
        bindings = configured_infer_model.create_bindings()
        
        output_buffers = {}
        for name in infer_model.output_names:
            output_buffers[name] = np.empty(infer_model.output(name).shape, dtype=np.float32)
            bindings.output(name).set_buffer(output_buffers[name])
        
        while True:
            try:
                start_loop_time = time.time()
                fps_frame_timestamps.append(start_loop_time)
                if len(fps_frame_timestamps) > 30:
                    fps_frame_timestamps.pop(0)
                
                if len(fps_frame_timestamps) > 1:
                    total_duration = fps_frame_timestamps[-1] - fps_frame_timestamps[0]
                    current_calculated_fps = round((len(fps_frame_timestamps) - 1) / total_duration, 1) if total_duration > 0 else 0.0
                raw_frame = picam.capture_array()
                frame = cv2.resize(raw_frame, (640, 480))
                frame = np.ascontiguousarray(frame)
            except Exception as e:
                print(f"Frame capture interruption: {e}")
                time.sleep(0.1)
                continue

            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame = np.ascontiguousarray(frame)
                
            clean_frame = frame.copy()

            hailo_input_frame = cv2.resize(frame, (640, 640))
            input_array = np.expand_dims(hailo_input_frame, axis=0).astype(np.uint8)
            bindings.input(input_name).set_buffer(input_array)
            
            configured_infer_model.run([bindings], timeout=1000)
            
            person_in_frame = False
            crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, 0, 0
            cropped_chest_frame = None
            distance_status = "SEARCHING"
            estimated_ft = 0.0
            local_badge_status = "SCANNING..."
            
            active_probabilities = {}
            top_confidence = 0
            predicted_folder_name = "UNKNOWN"

            # Safely instantiate NPU boundary metrics to block UnboundLocalError exceptions
            final_conf = 0.0
            best_score = -999.0
            best_ls_x, best_ls_y = 0, 0
            best_rs_x, best_rs_y = 0, 0

            try:
                for pair in layer_pairs:
                    score_tensor = output_buffers[pair["score"]]      
                    keypoint_tensor = output_buffers[pair["keypoint"]]  
                    stride = pair["stride"]
                    
                    H, W, _ = score_tensor.shape
                    
                    for h in range(H):
                        for w in range(W):
                            raw_score = score_tensor[h, w, 0]
                            
                            if raw_score > best_score:
                                kp_data = keypoint_tensor[h, w]
                                
                                ls_x_offset = kp_data[15]
                                ls_y_offset = kp_data[16]
                                rs_x_offset = kp_data[18]
                                rs_y_offset = kp_data[19]
                                
                                global_ls_x = ((ls_x_offset * 2.0) + w) * stride
                                global_ls_y = ((ls_y_offset * 2.0) + h) * stride
                                global_rs_x = ((rs_x_offset * 2.0) + w) * stride  
                                global_rs_y = ((rs_y_offset * 2.0) + h) * stride
                                
                                best_score = raw_score
                                
                                best_ls_x = int(global_ls_x)
                                best_ls_y = int(global_ls_y * (480 / 640))
                                best_rs_x = int(global_rs_x)
                                best_rs_y = int(global_rs_y * (480 / 640))

                final_conf = sigmoid(best_score)

                if final_conf > 0.65 and best_ls_x > 0 and best_rs_x > 0:
                    pixel_width = np.sqrt((best_ls_x - best_rs_x)**2 + (best_ls_y - best_rs_y)**2)
                    
                    if pixel_width > 0:
                        distance_inches = (REAL_SHOULDER_WIDTH_INCHES * FOCAL_LENGTH_FACTOR) / pixel_width
                        estimated_ft = round(distance_inches / 12.0, 1)
                    
                    if estimated_ft > MAX_DISTANCE_FEET:
                        distance_status = "TOO FAR"
                    elif estimated_ft < MIN_DISTANCE_FEET:
                        distance_status = "TOO CLOSE"
                    else:
                        distance_status = "OK"

                    if distance_status == "OK":
                        person_in_frame = True
                    else:
                        person_in_frame = False

                    box_width = int(pixel_width * 0.65)
                    box_height = box_width  
                    
                    shoulder_center_x = min(best_ls_x, best_rs_x) + (abs(best_ls_x - best_rs_x) // 2)
                    crop_x1 = shoulder_center_x - (box_width // 2)
                    crop_x2 = crop_x1 + box_width
                    
                    dynamic_upward_shift = int(box_height * 0.25)
                    crop_y1 = min(best_ls_y, best_rs_y) - dynamic_upward_shift
                    crop_y2 = crop_y1 + box_height
                    
                    crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
                    crop_x2, crop_y2 = min(frame.shape[1], crop_x2), min(frame.shape[0], crop_y2)
                    
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        cropped_chest_frame = clean_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                else:
                    person_in_frame = False
            except Exception as parse_err:
                person_in_frame = False

            # --- CALL EVERY FRAME INDEPENDENTLY ---
            has_departed = tracker.update_presence(person_in_frame)
            if has_departed:
                tracker.trigger_departure_event(frame)

            if tracker.state == "LOCKED":
                local_badge_status = tracker.current_user_final_decision
            
            current_time = time.time()
            current_date_str = time.strftime('%Y-%m-%d')
            full_target_dir = os.path.join(IMAGE_DIRECTORY, current_date_str)

            # =====================================================================
            # 🎲 GLOBAL RANDOM HARVEST LAYER (Decoupled from LOCKED constraints)
            # =====================================================================
            if person_in_frame and cropped_chest_frame is not None and cropped_chest_frame.size > 0:
                if random.random() < RANDOM_HARVEST_PROBABILITY:
                    os.makedirs(full_target_dir, exist_ok=True) 
                    daily_rand_prefix = os.path.join(current_date_str, "rand_frame")
                    
                    enc_file = save_anonymized_and_encrypted_frame(
                        cropped_chest_frame, 
                        None, 
                        prefix=daily_rand_prefix, 
                        extra_suffix="",
                        crop_offsets=(crop_x1, crop_y1)
                    )
                    if enc_file:
                        log_system_telemetry("random_harvest", f"Saved global random crop frame: {enc_file}", "INFO")

            # =====================================================================
            # 🔍 STAGE 2 CLASSIFIER & VALIDATION PERIODIC HARVEST LAYER
            # =====================================================================
            if tracker.state != "LOCKED" and distance_status == "OK" and cropped_chest_frame is not None and cropped_chest_frame.size > 0:
                resized_crop = cv2.resize(cropped_chest_frame, CLASSIFIER_IMG_SIZE)
                classifier_results = stage2_classifier(resized_crop, verbose=False, imgsz=224)
                
                class_mapping = classifier_results[0].names  
                probs_tensor = classifier_results[0].probs.data.cpu().numpy()
                
                for idx, class_name in class_mapping.items():
                    active_probabilities[class_name.upper()] = int(probs_tensor[idx] * 100)

                top_prediction_id = classifier_results[0].probs.top1
                predicted_folder_name = class_mapping[top_prediction_id].upper()
                top_confidence = int(classifier_results[0].probs.top1conf.item() * 100)

                if top_confidence >= CONFIDENCE_THRESHOLD:
                    local_badge_status = f"{predicted_folder_name} DETECTED"
                else:
                    local_badge_status = "CALCULATING..."
                
                # 1. Periodic validation harvest crops
                if current_time - last_harvest_time >= HARVEST_COOLDOWN_SECONDS:
                    os.makedirs(full_target_dir, exist_ok=True) 
                    daily_prefix = os.path.join(current_date_str, "frame")
                    enc_file = save_anonymized_and_encrypted_frame(
                        cropped_chest_frame, 
                        None, 
                        prefix=daily_prefix,      
                        extra_suffix="",     
                        crop_offsets=(crop_x1, crop_y1)
                    )
                    if enc_file:
                        last_harvest_time = current_time
                        log_system_telemetry("frame_harvest", f"Saved validation audit frame: {enc_file}", "INFO")
                
                if local_badge_status != "CALCULATING...":
                    tracker.update_evaluation(local_badge_status, top_confidence, estimated_ft, pet)

            has_active_event = telemetry_data.get("is_entry_event", False)
            evt_time = telemetry_data.get("time", None)
            evt_prof = telemetry_data.get("profile", None)
            evt_conf = telemetry_data.get("confidence", None)
            evt_prox = telemetry_data.get("proximity", None)

            score_readouts = " | ".join([f"{k}: {v}%" for k, v in active_probabilities.items()])
            HUD_badge_string = f"{local_badge_status} ({score_readouts})" if tracker.state != "IDLE" else "SCANNING..."
            
            pet.DAILY_GOAL = telemetry_data.get("daily_goal", 5)

            telemetry_data.update({
                "badge_status": HUD_badge_string,
                "distance_status": distance_status,
                "estimated_ft": estimated_ft if person_in_frame else 0.0,
                "is_entry_event": has_active_event,
                "pet_status": pet.get_status(),
                "successful_feedings": pet.successful_feedings,
                "daily_goal": pet.DAILY_GOAL
            })
            
            if has_active_event:
                telemetry_data["time"] = evt_time
                telemetry_data["profile"] = evt_prof
                telemetry_data["confidence"] = evt_conf
                telemetry_data["proximity"] = evt_prox

            # --- AUTOMATED 24-HOUR MIDNIGHT RESET LOOP ---
            check_today = time.strftime('%Y-%m-%d')
            if check_today != current_cycle_day:
                final_fate = pet.end_shift_and_reset()
                log_system_telemetry(
                    metric_name="tamagotchi_daily_lifecycle_summary",
                    data_value=f"Date concluded: {current_cycle_day} | Final Pet Status: {final_fate}",
                    log_level="INFO" if final_fate == "ALIVE" else "CRITICAL"
                )
                print(f"⏰ Midnight Rollover! Shift concluded pet status as: {final_fate}. Resetting counters.")
                current_cycle_day = check_today
                telemetry_data["daily_goal"] = pet.DAILY_GOAL

            # --- ONSCREEN RENDERING OVERLAYS ---
            if ALLOW_GUI_DISPLAY:  
                conf_text = f"NPU Conf: {int(final_conf * 100)}%"
                cv2.putText(frame, conf_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                
                fps_text = f"Engine: {current_calculated_fps} FPS"
                cv2.putText(frame, fps_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 250, 50), 2, cv2.LINE_AA)

                if person_in_frame:
                    box_color = (0, 255, 0) if "NO" not in local_badge_status and "DETECTED" in local_badge_status else (0, 0, 255)
                    if distance_status != "OK" or "CALCULATING" in local_badge_status:
                        box_color = (0, 165, 255)

                    cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), box_color, 2)
                    cv2.circle(frame, (best_ls_x, best_ls_y), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (best_rs_x, best_rs_y), 5, (0, 255, 0), -1)
                    
                    text_line1 = f"STATUS: {local_badge_status}"
                    text_line2 = score_readouts
                    text_line3 = f"PROXIMITY: {estimated_ft} ft"
                    
                    cv2.putText(frame, text_line1, (crop_x1, crop_y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2, cv2.LINE_AA)
                    cv2.putText(frame, text_line2, (crop_x1, crop_y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, text_line3, (crop_x1, crop_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA)

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
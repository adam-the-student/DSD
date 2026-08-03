import cv2
import numpy as np
import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import time
import json
import queue
from datetime import datetime

# Import custom helper classes (ensure these files are in the same folder)
from telemetry import log_system_telemetry, get_daily_csv_path, initialize_universal_logger
from tamagotchi import TamagotchiEngine

initialize_universal_logger()

# --- GLOBAL ASYNC EVENT BRIDGE ---
main_event_loop = None
event_queue = queue.Queue()
pet = None  

# --- TRACKING STATE MACHINE LAYER (Exact same as your final code) ---
class BadgeTrackerStateMachine:
    def __init__(self):
        self.state = "IDLE"
        self.current_user_max_confidence = 0
        self.current_user_final_decision = "UNKNOWN"
        self.frames_since_last_seen = 0
        self.max_lost_frames = 15  
        self.current_user_last_distance = 0.0

    def update_presence(self, person_detected: bool):
        if person_detected:
            self.frames_since_last_seen = 0
            if self.state == "IDLE":
                self.state = "TRACKING"
                print("👤 Person entered tracking zone.")
        else:
            if self.state != "IDLE":
                self.frames_since_last_seen += 1
                if self.frames_since_last_seen >= self.max_lost_frames:
                    return True # Signifies departure
        return False

    def update_evaluation(self, decision: str, confidence: int, estimated_ft, pet_engine_reference):
        if self.state in ["TRACKING", "EVALUATING", "LOCKED"]:
            
            is_currently_violating = "NO" in self.current_user_final_decision.upper() or self.current_user_final_decision == "UNKNOWN"
            is_new_read_compliant = "NO" not in decision.upper() and "DETECTED" in decision.upper()

            # The Upgrade Block
            if self.state == "LOCKED" and is_currently_violating and is_new_read_compliant and confidence >= 60:
                print(f"🔄 Compliance status upgraded live: Shifted to {decision} at {confidence}%.")
                self.current_user_max_confidence = confidence
                self.current_user_final_decision = decision
                pet_engine_reference.register_successful_feeding() # Ghost Diet fix applied
            
            # Initial Locking Block
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

    def trigger_departure_event(self):
        global event_queue, pet, telemetry_data, connected_clients, main_event_loop

        profile_string = self.current_user_final_decision.title()
        saved_dist = self.current_user_last_distance
        
        payload = {
            "is_entry_event": True,
            "time": datetime.now().strftime("%H:%M:%S"),
            "profile": profile_string,
            "confidence": f"{self.current_user_max_confidence}%",
            "proximity": f"{saved_dist} ft"
        }
        event_queue.put(payload)
                    
        print(f"🚶 Person departed. Event sent: {profile_string} at {saved_dist} ft")
        
        if pet is not None:
            pet.reset_user()
            telemetry_data["daily_goal"] = pet.DAILY_GOAL
            telemetry_data["successful_feedings"] = pet.successful_feedings
            telemetry_data["pet_status"] = pet.get_status()
            telemetry_data["streak"] = getattr(pet, "streak", 0)
            telemetry_data["health"] = getattr(pet, "health", 100)
            
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

        self.state = "IDLE"
        self.current_user_max_confidence = 0
        self.current_user_final_decision = "UNKNOWN"
        self.frames_since_last_seen = 0

telemetry_data = {
    "badge_status": "INITIALIZING...",
    "distance_status": "SEARCHING",
    "estimated_ft": 0.0,
    "is_entry_event": False,
    "pet_status": "UNKNOWN",
    "successful_feedings": 0,
    "daily_goal": 5,
    "streak": 0,
    "health": 100
}

connected_clients = set()
app = FastAPI()
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/")
async def get_dashboard(): return FileResponse("web/Index.html")

@app.get("/cat")
async def get_cat_dashboard(): return FileResponse("web/Cat.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global main_event_loop, telemetry_data, pet
    await websocket.accept()
    connected_clients.add(websocket)
    main_event_loop = asyncio.get_running_loop()

    try:
        while True:
            try:
                while not event_queue.empty():
                    live_alert = event_queue.get_nowait()
                    await websocket.send_json(live_alert)
                    event_queue.task_done()
            except queue.Empty:
                pass
            
            if pet is not None:
                telemetry_data["daily_goal"] = pet.DAILY_GOAL
                telemetry_data["successful_feedings"] = pet.successful_feedings
                telemetry_data["pet_status"] = pet.get_status()
                telemetry_data["streak"] = getattr(pet, "streak", 0)
                telemetry_data["health"] = getattr(pet, "health", 100)

            await websocket.send_json(dict(telemetry_data))
            await asyncio.sleep(0.05)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        connected_clients.discard(websocket)


def run_mock_pipeline():
    global telemetry_data, pet
    tracker = BadgeTrackerStateMachine()
    pet = TamagotchiEngine(daily_goal=5)
    
    # Mock states
    mock_person_in_frame = False
    mock_decision = "CALCULATING..."
    mock_confidence = 0
    
    print("\n🚀 Starting Keyboard Mock Pipeline.")
    print("Click the OpenCV window and press keys to test.")

    while True:
        # Create a blank black frame for the GUI
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # 1. Listen for keypresses
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            mock_person_in_frame = not mock_person_in_frame
            mock_decision = "CALCULATING..." # Reset when someone new walks in
            mock_confidence = 0
            print(f"Mock Input: Person in frame = {mock_person_in_frame}")
        elif key == ord('b') and mock_person_in_frame:
            mock_decision = "BADGE DETECTED"
            mock_confidence = 95
            print("Mock Input: Classifier says BADGE DETECTED")
        elif key == ord('n') and mock_person_in_frame:
            mock_decision = "NO BADGE"
            mock_confidence = 95
            print("Mock Input: Classifier says NO BADGE")
            
        # 2. Feed mock data into the tracker
        has_departed = tracker.update_presence(mock_person_in_frame)
        if has_departed:
            tracker.trigger_departure_event()
            
        distance_status = "OK" if mock_person_in_frame else "SEARCHING"
        estimated_ft = 5.0 if mock_person_in_frame else 0.0

        if mock_person_in_frame and mock_decision != "CALCULATING...":
            tracker.update_evaluation(mock_decision, mock_confidence, estimated_ft, pet)

        # 3. Handle Telemetry Updates
        local_badge_status = tracker.current_user_final_decision if tracker.state == "LOCKED" else mock_decision
        HUD_badge_string = f"{local_badge_status} ({mock_confidence}%)" if tracker.state != "IDLE" else "SCANNING..."
        
        telemetry_data.update({
            "badge_status": HUD_badge_string,
            "distance_status": distance_status,
            "estimated_ft": estimated_ft,
            "pet_status": pet.get_status(),
            "successful_feedings": pet.successful_feedings,
            "daily_goal": pet.DAILY_GOAL,
            "streak": getattr(pet, "streak", 0),
            "health": getattr(pet, "health", 100)
        })
        
        # 4. Draw UI on the blank frame
        cv2.putText(frame, "[p] Toggle Person", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "[b] Send 'BADGE DETECTED'", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "[n] Send 'NO BADGE'", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "[q] Quit", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        
        state_color = (0, 255, 255) if tracker.state != "LOCKED" else (0, 255, 0)
        cv2.putText(frame, f"Machine State: {tracker.state}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        cv2.putText(frame, f"Current Read: {HUD_badge_string}", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Person Present: {mock_person_in_frame}", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Mock Pipeline Input", frame)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        vision_thread = threading.Thread(target=run_mock_pipeline, daemon=True)
        vision_thread.start()

        print("Starting FastAPI Mock Server...")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
        
    except KeyboardInterrupt:
        print("\nStopping...")
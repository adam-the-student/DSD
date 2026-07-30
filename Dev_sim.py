# Dev_sim.py
import asyncio
import json
import queue
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import the exact Tamagotchi engine used by the final pipeline
from tamagotchi import TamagotchiEngine

app = FastAPI()

# ==========================================
# 🌐 WEB SERVING ROUTES
# ==========================================

app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/")
async def get_dashboard():
    return FileResponse("web/Index.html")

@app.get("/cat")
async def get_cat_dashboard():
    return FileResponse("web/Cat.html")


# ==========================================
# ⚙️ TAMAGOTCHI ENGINE SETUP
# ==========================================
pet = TamagotchiEngine(daily_goal=5)
event_queue = queue.Queue()
connected_clients = set()

# Added the 'streak' key. Defaults to 3 if pet.streak doesn't exist yet.
telemetry_data = {
    "badge_status": "SCANNING...",
    "distance_status": "SEARCHING",
    "estimated_ft": 0.0,
    "is_entry_event": False,
    "pet_status": pet.get_status(),
    "successful_feedings": pet.successful_feedings,
    "daily_goal": pet.DAILY_GOAL,
    "streak": getattr(pet, "streak", 3), 
    "fps": 60.0  
}


# ==========================================
# 📡 WEBSOCKET ENDPOINT
# ==========================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    
    await websocket.send_json({
        "is_startup_history": True,
        "raw_text": "--- DEV SIMULATOR HISTORY BOOT ---"
    })
    
    try:
        while True:
            # Sync pet state continuously
            telemetry_data["pet_status"] = pet.get_status()
            telemetry_data["successful_feedings"] = pet.successful_feedings
            telemetry_data["daily_goal"] = pet.DAILY_GOAL
            telemetry_data["streak"] = getattr(pet, "streak", 3)
            
            try:
                while not event_queue.empty():
                    live_alert = event_queue.get_nowait()
                    await websocket.send_json(live_alert)
                    event_queue.task_done()
                    await asyncio.sleep(0.01)
            except queue.Empty:
                pass
            
            await websocket.send_json(telemetry_data)
            await asyncio.sleep(0.05)
            
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def reset_badge_status(delay: float):
    await asyncio.sleep(delay)
    telemetry_data["badge_status"] = "SCANNING..."
    pet.reset_user()


# ==========================================
# 🛠️ DEV CONSOLE ENDPOINTS
# ==========================================
@app.get("/dev/add")
async def dev_add_checkin():
    pet.reset_user() 
    success = pet.register_successful_feeding()
    print(f"🔧 DEV: Added check-in manually. Total: {pet.successful_feedings}")
    return {"status": "success", "total": pet.successful_feedings, "registered": success}

@app.get("/dev/remove")
async def dev_remove_checkin():
    if pet.successful_feedings > 0:
        pet.successful_feedings -= 1
        pet.save_state_to_disk() 
        print(f"🔧 DEV: Removed check-in manually. Total: {pet.successful_feedings}")
    return {"status": "success", "total": pet.successful_feedings}

@app.get("/dev/simulate_person")
async def dev_simulate_person():
    pet.reset_user()
    pet.register_successful_feeding()
    
    telemetry_data["badge_status"] = "DOS DETECTED (99%)"
    
    payload = {
        "is_entry_event": True,
        "time": datetime.now().strftime("%H:%M:%S"),
        "profile": "MOCK USER DETECTED",
        "confidence": "99%",
        "proximity": "3.5 ft"
    }
    event_queue.put(payload)
    asyncio.create_task(reset_badge_status(2.0))
    print(f"🔧 DEV: Simulated POSITIVE person event. Total check-ins: {pet.successful_feedings}")
    return {"status": "event_triggered", "total": pet.successful_feedings}

@app.get("/dev/simulate_negative")
async def dev_simulate_negative():
    pet.reset_user()
    telemetry_data["badge_status"] = "NO BADGE DETECTED (85%)"
    
    payload = {
        "is_entry_event": True,
        "time": datetime.now().strftime("%H:%M:%S"),
        "profile": "NO BADGE DETECTED",
        "confidence": "85%",
        "proximity": "4.2 ft"
    }
    event_queue.put(payload)
    asyncio.create_task(reset_badge_status(5.0))
    print(f"🔧 DEV: Simulated NEGATIVE person event (No increment). Total check-ins: {pet.successful_feedings}")
    return {"status": "event_triggered", "total": pet.successful_feedings}

@app.get("/dev")
async def get_dev_dashboard():
    html = """
    <html>
        <head><title>Dev Simulator Dashboard</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h2>🛠️ Simulator Control Panel</h2>
            <button onclick="fetch('/dev/add')" style="padding: 10px; font-size: 16px;">➕ Add Check-in Manually</button>
            <button onclick="fetch('/dev/remove')" style="padding: 10px; font-size: 16px;">➖ Remove Check-in Manually</button>
            <br><br>
            <button onclick="fetch('/dev/simulate_person')" style="padding: 10px; font-size: 16px; background-color: lightgreen;">🚶 Simulate POSITIVE Event (+1 Check-in)</button>
            <button onclick="fetch('/dev/simulate_negative')" style="padding: 10px; font-size: 16px; background-color: lightcoral;">🚨 Simulate NEGATIVE Event (Does Nothing)</button>
        </body>
    </html>
    """
    return HTMLResponse(content=html)


if __name__ == "__main__":
    print("Starting Development Simulator Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
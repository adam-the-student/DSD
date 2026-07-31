# tamagotchi.py
import json
import os
import time

STATE_FILE = "tamagotchi_state.json"

class TamagotchiEngine:
    def __init__(self, daily_goal=5):
        self.DAILY_GOAL = daily_goal     
        self.successful_feedings = 0     
        self.shift_ended = False         
        self.feeding_locked_for_current_user = False
        
        # --- NEW LONG-TERM METRICS ---
        self.health = 100
        self.streak = 0
        
        # Load any existing progress from today on boot
        self.load_state_from_disk()

    def register_successful_feeding(self):
        """Awards 1 point per unique tracking session and flushes to disk."""
        if not self.feeding_locked_for_current_user:
            self.successful_feedings += 1
            self.feeding_locked_for_current_user = True  
            self.save_state_to_disk() # 💾 Save instantly so crashes don't lose data
            return True
        return False

    def reset_user(self):
        self.feeding_locked_for_current_user = False

    def get_status(self):
        """Updates the frontend router string directly based on health and feedings."""
        if self.health <= 0:
            return "DEAD"
        elif self.health <= 50:
            return "SICK"
        elif self.successful_feedings >= self.DAILY_GOAL:
            return "SATISFIED"
        return "ALIVE"

    def save_state_to_disk(self):
        """Dumps current progress and today's date string into a local JSON text file."""
        state_payload = {
            "date": time.strftime('%Y-%m-%d'),
            "successful_feedings": self.successful_feedings,
            "daily_goal": self.DAILY_GOAL,
            "health": self.health,    # Include health
            "streak": self.streak     # Include streak
        }
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state_payload, f)
        except Exception as e:
            print(f"⚠️ Failed to save pet state to disk: {e}")

    def load_state_from_disk(self):
        """Attempts to recover data. Wipes progress if the file belongs to a previous day."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    state_payload = json.load(f)
                
                # --- LOAD PERSISTENT STATS ---
                # These load regardless of what day it is
                self.health = state_payload.get("health", 100)
                self.streak = state_payload.get("streak", 0)
                
                # Check if the saved file matches today's calendar date
                if state_payload.get("date") == time.strftime('%Y-%m-%d'):
                    self.successful_feedings = state_payload.get("successful_feedings", 0)
                    self.DAILY_GOAL = state_payload.get("daily_goal", self.DAILY_GOAL)
                    print(f"💾 RECOVERY ACTIVE: Restored {self.successful_feedings} check-ins, {self.streak} streak, {self.health} HP.")
                else:
                    print(f"📆 Old state file detected from a previous shift. Starting fresh with {self.streak} streak and {self.health} HP.")
                    self.save_state_to_disk()
            except Exception as e:
                print(f"⚠️ Error reading pet state backup: {e}")

    def end_shift_and_reset(self):
        """Logs the final day pass/fail status, then revives the cat for a new 24hr loop."""
        self.shift_ended = True
        
        # --- NEW END OF DAY EVALUATION MATH ---
        if self.successful_feedings >= self.DAILY_GOAL:
            self.streak += 1
            self.health = min(100, self.health + 25)  # Heal up to 100
            final_outcome = "TARGET MET - HEALED"
        else:
            self.streak = 0
            self.health = max(0, self.health - 34)    # Take damage, floor at 0
            final_outcome = "TARGET FAILED - DAMAGED"
        
        # Reset engine variables for the next 24-hour cycle
        self.successful_feedings = 0
        self.shift_ended = False
        self.feeding_locked_for_current_user = False
        self.save_state_to_disk() # Update the backup file for the new day
        
        return final_outcome

    def get_telemetry_string(self, HUD_badge_string):
        return f"{HUD_badge_string} | PET: {self.get_status()} ({self.successful_feedings}/{self.DAILY_GOAL}) | HP: {self.health} | STREAK: {self.streak}"
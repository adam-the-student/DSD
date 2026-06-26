# tamagotchi.py
import time

class TamagotchiEngine:
    def __init__(self, daily_goal=5):
        self.DAILY_GOAL = daily_goal     # Number of compliant check-ins required
        self.successful_feedings = 0     # Checked-ins tracked today
        self.last_feeding_time = 0.0

    def register_successful_feeding(self):
        """Ticks up the feeding counter with a built-in anti-spam threshold."""
        current_time = time.time()
        # 10-second cool down cushion per validation block
        if current_time - self.last_feeding_time > 10:
            self.successful_feedings += 1
            self.last_feeding_time = current_time
            return True
        return False

    def get_status(self):
        """Returns the absolute lifecycle state readout."""
        if self.successful_feedings >= self.DAILY_GOAL:
            return "ALIVE"
        return "DEAD"

    def get_telemetry_string(self, HUD_badge_string):
        """Formats the data layout string directly for your master dashboard payload."""
        status = self.get_status()
        return f"{HUD_badge_string} | PET: {status} ({self.successful_feedings}/{self.DAILY_GOAL})"
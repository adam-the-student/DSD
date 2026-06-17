# telemetry.py
import os
import csv
from datetime import datetime

def get_daily_csv_path():
    """Generates a file path unique to the current calendar date inside a 'logs' folder."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(log_dir, f"telemetry_{date_str}.csv")

def initialize_universal_logger():
    """Establishes the base data matrix infrastructure if not already present on disk."""
    target_csv = get_daily_csv_path()
    if not os.path.exists(target_csv):
        with open(target_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Log_Level", "Metric_Name", "Data_Value"])
        print(f"📁 Initialized universal systems logger matrix at: {target_csv}")


def log_system_telemetry(metric_name: str, data_value, log_level: str = "INFO"):
    """Appends tracking properties to the daily rotated disk file."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target_csv = get_daily_csv_path()
    
    try:
        file_exists = os.path.exists(target_csv) and os.path.getsize(target_csv) > 0
        with open(target_csv, mode='a', newline='', errors='ignore') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Log_Level", "Metric_Name", "Data_Value"])
            writer.writerow([current_time, log_level.upper(), metric_name, str(data_value)])
        return current_time
    except Exception as e:
        print(f"⚠️ Telemetry Disk Write Failure: {e}")
        return current_time
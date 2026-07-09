import os
import csv
import re
import cv2
from datetime import datetime

# --- CONFIGURATION PATHS ---
CSV_INPUT_PATH = "logs/telemetry_2026-07-09.csv"          # Target log file
FRAMES_DIRECTORY = "harvested_edge_cases"          # Folder holding frame_*.jpg assets
OUTPUT_VERIFIED_CSV = "logs/verified_dataset_2026-07-09.csv"  # Target spreadsheet output
MAX_ALLOWED_DRIFT_SECONDS = 5.0                    # Captures frames within X seconds of the log

def human_to_unix(timestamp_str):
    """Converts '2026-07-08 12:56:54' into an integer Unix Epoch timestamp."""
    try:
        clean_str = timestamp_str.strip().replace('"', '').replace("'", "")
        dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    except ValueError:
        return None

def extract_unix_from_filename(filename):
    """Extracts the integer timestamp out of 'frame_1783537014_PATIENT.jpg'."""
    match = re.search(r"frame_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None

def build_frame_index(directory):
    """Creates a sorted list of (unix_timestamp, filename) pairs from the folder."""
    index = []
    if not os.path.exists(directory):
        print(f"⚠️ Directory not found: {directory}")
        return index
        
    for fname in os.listdir(directory):
        # Explicitly isolate validation timeline assets
        if fname.startswith("frame_") and fname.endswith(".jpg"):
            ts = extract_unix_from_filename(fname)
            if ts is not None:
                index.append((ts, fname))
    
    index.sort(key=lambda x: x[0])
    return index

def find_all_matching_frames(log_unix, frame_index, max_drift):
    """Gathers ALL frames that fell within the time window of this log entry."""
    if not frame_index or log_unix is None:
        return []
        
    matches = []
    for frame_unix, filename in frame_index:
        if abs(log_unix - frame_unix) <= max_drift:
            matches.append(filename)
    return matches

def main():
    print("📦 Indexing harvested image frames...")
    frame_index = build_frame_index(FRAMES_DIRECTORY)
    print(f"Loaded {len(frame_index)} total validation frames from '{FRAMES_DIRECTORY}'.")
    
    if not os.path.exists(CSV_INPUT_PATH):
        print(f"❌ Error: Input CSV file does not exist at {CSV_INPUT_PATH}")
        return

    file_exists = os.path.exists(OUTPUT_VERIFIED_CSV)
    outfile = open(OUTPUT_VERIFIED_CSV, mode='a', newline='', encoding='utf-8')
    csv_writer = csv.writer(outfile)
    
    if not file_exists:
        csv_writer.writerow(["Timestamp", "Log_Level", "Metric", "Data_Payload", "Matched_Frames", "Accuracy_Status"])

    print(f"📖 Reading logs from: {CSV_INPUT_PATH}...")
    print("\n🎮 CONTROLS:")
    print("  [Spacebar] = Next frame for this log row")
    print("  [v]        = Mark whole event as ACCURATE")
    print("  [i]        = Mark whole event as INCORRECT")
    print("  [s]        = Skip row")
    print("  [q]        = Save and Quit\n")

    cv2.namedWindow("Verification Station", cv2.WINDOW_NORMAL)

    with open(CSV_INPUT_PATH, mode='r', encoding='utf-8', errors='ignore') as infile:
        reader = csv.reader(infile)
        
        for row in reader:
            if not row or len(row) < 4:
                continue
                
            log_time_str, log_level, metric_name, data_value = row[0], row[1], row[2], row[3]
            
            if "wearer_departure_summary" not in metric_name:
                continue

            log_unix = human_to_unix(log_time_str)
            matching_frames = find_all_matching_frames(log_unix, frame_index, MAX_ALLOWED_DRIFT_SECONDS)
            
            if not matching_frames:
                csv_writer.writerow([log_time_str, log_level, metric_name, data_value, "NO_MATCHING_FRAMES", "N/A"])
                continue
                
            # Loop interactive UI window for current log row
            current_idx = 0
            accuracy_label = None
            
            while True:
                current_frame = matching_frames[current_idx]
                img_path = os.path.join(FRAMES_DIRECTORY, current_frame)
                frame_img = cv2.imread(img_path)
                
                if frame_img is None:
                    current_idx = (current_idx + 1) % len(matching_frames)
                    continue

                # UI Overlay details
                display_canvas = frame_img.copy()
                total_f = len(matching_frames)
                overlay_text = f"Frame {current_idx + 1}/{total_f}: {data_value}"
                
                cv2.putText(display_canvas, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(display_canvas, "[Space] Next Frame | [v] Valid | [i] Incorrect | [s] Skip", (10, display_canvas.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                cv2.imshow("Verification Station", display_canvas)
                key = cv2.waitKey(0) & 0xFF

                if key == ord(' '):  # Cycle forward through matching images
                    current_idx = (current_idx + 1) % len(matching_frames)
                elif key == ord('v'):
                    accuracy_label = "ACCURATE"
                    print(f"✅ Log Entry {log_time_str} verified using {total_f} frames.")
                    break
                elif key == ord('i'):
                    accuracy_label = "INCORRECT"
                    print(f"❌ Log Entry {log_time_str} flagged as INCORRECT.")
                    break
                elif key == ord('s'):
                    accuracy_label = "SKIPPED"
                    break
                elif key == ord('q'):
                    print("Stopping verification process...")
                    cv2.destroyAllWindows()
                    outfile.close()
                    return

            if accuracy_label and accuracy_label != "SKIPPED":
                # Join all looked-at frames with a pipe separator inside the cell row entry 
                frames_list_str = "|".join(matching_frames)
                csv_writer.writerow([log_time_str, log_level, metric_name, data_value, frames_list_str, accuracy_label])
                outfile.flush()

    cv2.destroyAllWindows()
    outfile.close()
    print(f"🏁 Update saved to: {OUTPUT_VERIFIED_CSV}")

if __name__ == "__main__":
    main()
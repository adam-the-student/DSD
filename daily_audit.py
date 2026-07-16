import os
import glob
import pandas as pd
import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "runs/classify/train-4/weights/best.pt"  # Your Stage 2 model
CSV_DIRECTORY = "logs"                                # Directory where your daily CSV files are stored
IMAGE_DIRECTORY = "harvested_edge_cases"                   # Root directory where the crop stills are stored

def run_daily_audit():
    # 1. Sanity checks
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Stage 2 model not found at '{MODEL_PATH}'")
        return

    # 2. Get the target date from user
    print("📋 --- DAILY MODEL AUDIT TOOL ---")
    target_date = input("Enter the date to audit (format e.g., 2026-07-16 or matching your CSV filename): ").strip()
    
    # Locate the CSV file
    csv_pattern = os.path.join(CSV_DIRECTORY, f"*{target_date}*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"❌ Error: No CSV file found matching '{target_date}' inside '{CSV_DIRECTORY}/'.")
        return
    
    csv_path = csv_files[0]
    print(f"📖 Loading log data from: {csv_path}")
    
    try:
        # Load the telemetry log dataset
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    if df.empty:
        print("⚠️ The selected CSV file is empty.")
        return

    # 3. Load the Stage 2 model
    print(f"🧠 Loading Stage 2 Model...")
    model = YOLO(MODEL_PATH)
    
    # Pre-count the rows we actually care about to give a precise event tracker
    harvest_rows = df[df['Metric_Name'].isin(['frame_harvest', 'random_harvest'])]
    total_records = len(harvest_rows)
    
    if total_records == 0:
        print("⚠️ No 'frame_harvest' or 'random_harvest' lines found in this log file.")
        return
        
    print(f"\n🚀 Found {total_records} logged crop events for this day. Opening viewer...")
    print("👉 Press ANY KEY to go to the next logged frame.")
    print("👉 Press 'q' or 'ESC' to exit the auditor.")

    current_match_idx = 0

    # 4. Step through the CSV entries
    for idx, row in df.iterrows():
        metric = str(row.get('Metric_Name', ''))
        data_val = str(row.get('Data_Value', ''))
        
        # We only care about rows where an actual crop image was saved
        if metric not in ['frame_harvest', 'random_harvest']:
            continue
            
        current_match_idx += 1
            
        # Parse the filename out of text like: "Saved validation audit frame: frame_1784199755_PATIENT.jpg"
        if ":" in data_val:
            img_name = data_val.split(":")[-1].strip()
        else:
            continue
            
        # Clean up the numeric timestamp from the filename (e.g., 1784199755)
        timestamp_str = "".join(filter(str.isdigit, img_name))
        
        if not timestamp_str:
            continue

        # Look for the historical decision log. 
        # Since it lives on a separate line in your telemetry file (e.g., wearer_departure_summary),
        # we scan a few rows down to see if a decision was recorded right after this frame.
        historical_log = "No exit decision recorded at this timestamp"
        search_window = df.iloc[idx : min(idx + 5, len(df))]
        
        for _, sub_row in search_window.iterrows():
            if sub_row.get('Metric_Name') == 'wearer_departure_summary':
                historical_log = str(sub_row.get('Data_Value', 'UNKNOWN'))
                break

        # FUZZY SEARCH: Match the unique numeric ID against files on disk
        search_pattern = os.path.join(IMAGE_DIRECTORY, f"*{timestamp_str}*")
        matching_files = glob.glob(search_pattern)
        
        if not matching_files:
            print(f"⚠️ Item {current_match_idx}/{total_records} (Row {idx+1}): No crop file found on disk for timestamp ID *{timestamp_str}*")
            continue
            
        img_path = matching_files[0]

        # Load image for processing and display
        display_img = cv2.imread(img_path)
        if display_img is None:
            print(f"❌ Error: Found '{os.path.basename(img_path)}' but OpenCV failed to open it.")
            continue
            
        # Run the current Stage 2 model on the file
        results = model(img_path, verbose=False)
        result = results[0]
        
        # Extract live prediction data
        names = result.names
        probs = result.probs.data.tolist()
        top_class_id = result.probs.top1
        live_pred = names[top_class_id].upper()
        live_conf = probs[top_class_id]

        # Resize image for clarity if it's a tight crop
        h, w, _ = display_img.shape
        if w < 500 or h < 500:
            display_img = cv2.resize(display_img, (500, 500), interpolation=cv2.INTER_LINEAR)
            h, w, _ = display_img.shape

        # Make space at the top for information text overlay (solid dark canvas bar)
        cv2.rectangle(display_img, (0, 0), (w, 85), (20, 20, 20), -1)

        # Draw current model prediction metrics
        color = (0, 255, 0) if live_pred == "BADGE" and live_conf >= 0.60 else (0, 0, 255)
        live_info = f"LIVE MODEL: {live_pred} ({live_conf*100:.1f}%)"
        csv_info = f"HISTORICAL LOG: {historical_log}"
        counter_info = f"Record [{current_match_idx}/{total_records}] | File: {os.path.basename(img_path)}"

        cv2.putText(display_img, live_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(display_img, csv_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(display_img, counter_info, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1)

        # Draw an outer window border indicating pass/fail status
        cv2.rectangle(display_img, (0, 0), (w, h), color, 4)

        # Display window
        window_title = "Daily Stage 2 Model Audit Tool"
        cv2.imshow(window_title, display_img)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 1)

        # Keyboard event capture
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:  # 'q' key or ESC key
            print("\n👋 Exited audit tool early.")
            break

    cv2.destroyAllWindows()
    print("✅ Audit complete for this batch.")

if __name__ == "__main__":
    # Ensure folder paths exist cleanly
    os.makedirs(CSV_DIRECTORY, exist_ok=True)
    os.makedirs(IMAGE_DIRECTORY, exist_ok=True)
    run_daily_audit()
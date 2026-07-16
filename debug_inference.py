import os
import glob
import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "models/dosClassifier3.pt"
IMAGE_FOLDER = "testframes"

def run_interactive_debugger():
    # 1. Ensure testframe folder exists
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    # 2. Verify model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at '{MODEL_PATH}'")
        return

    # 3. Find all images in the folder
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in supported_extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))

    if not image_files:
        print(f"📁 Created folder '{IMAGE_FOLDER}'. Drop your test crops inside and run the script again!")
        return

    # 4. Load the trained YOLO model
    print(f"🧠 Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    print(f"\n🚀 Found {len(image_files)} image(s). Opening viewer...")
    print("👉 Press ANY KEY to go to the next image.")
    print("👉 Press 'q' or 'ESC' to exit.")

    # 5. Loop through images one by one
    for idx, img_path in enumerate(image_files):
        # Load image for displaying
        display_img = cv2.imread(img_path)
        if display_img is None:
            continue
            
        # Run classification
        results = model(img_path, verbose=False)
        result = results[0]
        
        # Get predictions
        names = result.names
        probs = result.probs.data.tolist()
        top_class_id = result.probs.top1
        top_class_name = names[top_class_id].upper()
        top_confidence = probs[top_class_id]
        
        # Determine color for the text overlay (Green for secure, Orange/Red for bad/warning)
        if top_class_name == "BADGE" and top_confidence >= 0.60:
            color = (0, 255, 0)      # Bright Green (BGR format)
            status_text = "PASSED"
        else:
            color = (0, 0, 255)      # Red
            status_text = "NO BADGE" if top_class_name == "NO_BADGE" else "LOW CONFIDENCE"

        # Resize image slightly if it is too small to read the text overlay clearly
        h, w, _ = display_img.shape
        if w < 400 or h < 400:
            display_img = cv2.resize(display_img, (400, 400), interpolation=cv2.INTER_LINEAR)
            h, w, _ = display_img.shape

        # Draw a solid background bar at the top for clear text readability
        cv2.rectangle(display_img, (0, 0), (w, 65), (20, 20, 20), -1)

        # Draw prediction and confidence on the frame
        info_str = f"{top_class_name}: {top_confidence*100:.1f}% ({status_text})"
        file_counter = f"[{idx + 1}/{len(image_files)}]"
        
        cv2.putText(display_img, info_str, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display_img, file_counter, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Draw a colored border around the entire window to show status at a glance
        border_thickness = 4
        cv2.rectangle(display_img, (0, 0), (w, h), color, border_thickness)

        # Display the window
        cv2.imshow("Stage 2 Model Debugger", display_img)
        
        # Force the window to stay on top
        cv2.setWindowProperty("Stage 2 Model Debugger", cv2.WND_PROP_TOPMOST, 1)

        # Wait for keypress
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:  # 'q' key or ESC key
            print("\n👋 Exited debugger early.")
            break

    cv2.destroyAllWindows()
    print("✅ Finished checking all frames.")

if __name__ == "__main__":
    run_interactive_debugger()
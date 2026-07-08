import cv2
import os
import shutil

# --- DIRECTORY PATH CONFIGURATIONS ---
SOURCE_DIR = "harvested_edge_cases"
TARGET_BADGE_DIR = os.path.join("dataset", "badge")
TARGET_NO_BADGE_DIR = os.path.join("dataset", "no_badge")

def initialize_dataset_folders():
    """Ensures target classification directories exist before sorting begins."""
    os.makedirs(TARGET_BADGE_DIR, exist_ok=True)
    os.makedirs(TARGET_NO_BADGE_DIR, exist_ok=True)
    print("📁 Dataset training directories verified and active.")

def run_sorting_utility():
    initialize_dataset_folders()
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist yet.")
        return
        
    # Gather all harvested image files matching common extensions
    harvested_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not harvested_files:
        print("\n🎉 No harvested edge cases to sort! The directory is empty.")
        return
        
    print(f"\n🚀 Found {len(harvested_files)} edge-case images ready for processing.")
    print("==================================================")
    print("INSTRUCTIONS:")
    print("  Press [b] -> Move to BADGE folder")
    print("  Press [n] -> Move to NO_BADGE folder")
    print("  Press [s] -> SKIP image (keep in source folder)")
    print("  Press [q] -> QUIT sorting engine")
    print("==================================================")

    for idx, file_name in enumerate(harvested_files):
        source_path = os.path.join(SOURCE_DIR, file_name)
        
        # Read frame file array matrix from disk
        img = cv2.imread(source_path)
        if img is None:
            print(f"⚠️ Warning: Could not read image file {file_name}. Skipping...")
            continue
            
        # Upscale frame preview strictly inside display window so it is easy to inspect
        display_preview = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
        
        # Inject standard navigation UI text into image array window header
        progress_text = f"Image {idx + 1}/{len(harvested_files)}"
        cv2.putText(display_preview, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_preview, "[B]=Badge | [N]=No Badge | [S]=Skip", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display rendering layout window hook
        window_title = "DSD Active Learning - Dataset Sorting Utility"
        cv2.imshow(window_title, display_preview)
        
        # Wait for structural keycode mapping inputs
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('b'):
            # Move to verified badge category layout
            dest_path = os.path.join(TARGET_BADGE_DIR, file_name)
            shutil.move(source_path, dest_path)
            print(f"✅ [{idx+1}] Moved {file_name} -> dataset/badge/")
            
        elif key == ord('n'):
            # Move to empty chest/graphic baseline category layout
            dest_path = os.path.join(TARGET_NO_BADGE_DIR, file_name)
            shutil.move(source_path, dest_path)
            print(f"❌ [{idx+1}] Moved {file_name} -> dataset/no_badge/")
            
        elif key == ord('s'):
            # Keep image file anchored right inside source tracking cache
            print(f"➡️ [{idx+1}] Skipped {file_name} (left in harvested folder).")
            continue
            
        elif key == ord('x'):
        # 🟢 Permanently erase the current file from the disk
            try:
                os.remove(source_path)
                print(f"🗑️  [{idx+1}] Deleted {file_name}")
            except Exception as e:
                print(f"⚠️  Failed to delete {file_name}: {e}")

            
        elif key == ord('q'):
            print("\nExiting dataset utility engine. Sorting paused.")
            break
        else:
            # Handle invalid keys safely without skipping current item bounds
            print("⚠️ Invalid key pressed. Use 'b', 'n', 's', or 'q'.")
            # Clear display windows and decrement index loop back to retry item
            cv2.destroyAllWindows()
            return run_sorting_utility()

    cv2.destroyAllWindows()
    print("\nProcessing complete.")

if __name__ == "__main__":
    run_sorting_utility()
import tensorflow as tf
import os
import cv2
import numpy as np

# --- CONFIGURATION ---
RAW_DIR = os.path.join("dataset", "raw")
AUG_DIR = os.path.join("dataset", "augmented")
MULTIPLIER = 20  # Number of augmented variations to generate per raw image

# --- TWEAK YOUR PIPELINE HERE ---
# This setup intentionally degrades and shifts full-image colors to smash 
# any dependency the model has on your specific shirt textures or bedroom lighting.
augmentation_pipeline = tf.keras.Sequential([
    # Slight rotations and horizontal flips keep the model position-agnostic
    tf.keras.layers.RandomRotation(factor=0.08, fill_mode="reflect"),
    tf.keras.layers.RandomFlip("horizontal"),
    
    # 1. MAXIMUM HUE ROTATION: Spin the full color wheel violently.
    # A factor of 0.5 completely rotates all colors (Blue shirt becomes Red/Green/Yellow)
    tf.keras.layers.RandomHue(factor=0.5),          
    
    # 2. INTENSE SATURATION AND BRIGHTNESS VARIANCE:
    # Mimics harsh hospital fluorescent lights vs dim hallway settings
    tf.keras.layers.RandomSaturation(factor=(0.2, 1.8)),
    tf.keras.layers.RandomBrightness(factor=0.3),
    tf.keras.layers.RandomContrast(factor=(0.6, 1.4))
])

# Ensure base directory exists
if not os.path.exists(RAW_DIR):
    print(f"Error: Base directory '{RAW_DIR}' not found. Please run collect_data.py first.")
    exit()

# Scan for available category subdirectories
all_categories = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]

if not all_categories:
    print(f"No subdirectories found inside '{RAW_DIR}'.")
    exit()

# --- TERMINAL INTERACTION INTERFACE ---
print("\n==========================================")
print("       DSD DATA AUGMENTATION UTILITY       ")
print("==========================================")
print("Available folders found:")
print("  [0] -- AUGMENT ENTIRE RAW DIRECTORY --")
for idx, cat in enumerate(all_categories, start=1):
    print(f"  [{idx}] {cat}")
print("==========================================")

try:
    choice = int(input("Select an option number to augment: ").strip())
except ValueError:
    print("Invalid input. Please enter a valid number.")
    exit()

# Determine targeted categories based on choice
if choice == 0:
    categories_to_process = all_categories
    print("\nTarget selected: Processing all raw folders.\n")
elif 1 <= choice <= len(all_categories):
    categories_to_process = [all_categories[choice - 1]]
    print(f"\nTarget selected: Processing only '{categories_to_process[0]}'\n")
else:
    print("Option out of range. Aborting.")
    exit()

# --- PIPELINE ENGINE ---
for category in categories_to_process:
    input_folder = os.path.join(RAW_DIR, category)
    output_folder = os.path.join(AUG_DIR, category)
    
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Skipping '{category}': No images found inside.")
        continue
        
    print(f"Augmenting '{category}': Generating {len(image_files) * MULTIPLIER} images...")
    
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_batch = np.expand_dims(img_rgb, axis=0)
        
        for i in range(MULTIPLIER):
            augmented_batch = augmentation_pipeline(img_batch, training=True)
            augmented_img = augmented_batch[0].numpy().astype(np.uint8)
            
            save_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
            name_parts = os.path.splitext(filename)
            new_name = f"{name_parts[0]}_aug_{i}.jpg"
            cv2.imwrite(os.path.join(output_folder, new_name), save_img)

print(f"\nSuccess! Operations complete. Output stored in: {AUG_DIR}")
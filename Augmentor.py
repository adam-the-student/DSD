import tensorflow as tf
import os
import cv2
import numpy as np

# --- CONFIGURATION ---
RAW_DIR = os.path.join("dataset", "raw")
AUG_DIR = os.path.join("dataset", "augmented")
MULTIPLIER = 20  # How many augmented images to make per 1 raw image

# --- TWEAK YOUR AUGMENTATION PIPELINE HERE ---
augmentation_pipeline = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(factor=0.05),
    tf.keras.layers.RandomFlip("horizontal"),
    
    # Color-shifting layers (Crucial for blocking room/shirt memorization)
    tf.keras.layers.RandomHue(factor=0.5),          # 0.5 allows full 180° color wheel flips
    tf.keras.layers.RandomSaturation(factor=(0.3, 1.7)),
    tf.keras.layers.RandomBrightness(factor=0.2),
    tf.keras.layers.RandomContrast(factor=(0.8, 1.2))
])

print("Starting data augmentation...")

# Process both 'badge' and 'no_badge' subfolders
for category in ["badge", "no_badge"]:
    input_folder = os.path.join(RAW_DIR, category)
    output_folder = os.path.join(AUG_DIR, category)
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(input_folder):
        print(f"Skipping {category}: Raw folder not found.")
        continue
        
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} raw images in '{category}'. Generating {len(image_files) * MULTIPLIER} variations...")
    
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        
        # Load image via OpenCV and convert to RGB format for TensorFlow
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Add a batch dimension: shape becomes (1, 224, 224, 3)
        img_batch = np.expand_dims(img_rgb, axis=0)
        
        for i in range(MULTIPLIER):
            # Pass image through the pipeline
            augmented_batch = augmentation_pipeline(img_batch, training=True)
            augmented_img = augmented_batch[0].numpy().astype(np.uint8)
            
            # Convert back to BGR so OpenCV can save it correctly
            save_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
            
            # Save file with a distinct name
            name_parts = os.path.splitext(filename)
            new_name = f"{name_parts[0]}_aug_{i}.jpg"
            cv2.imwrite(os.path.join(output_folder, new_name), save_img)

print(f"Success! Check out your new images in: {AUG_DIR}")
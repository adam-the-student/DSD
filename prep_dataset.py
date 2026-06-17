# prepare_dataset.py
import os
import cv2
import numpy as np
from cryptography.fernet import Fernet

# 1. Initialize the master key (Must match the key generated on the Pi)
with open(".master_secret.key", "rb") as key_file:
    KEY = key_file.read()
fernet = Fernet(KEY)

ENCRYPTED_DIR = "harvested_edge_cases"
TRAINING_INPUT_DIR = "dataset/retrain_images" # Where YOLO or your classifier expects images
os.makedirs(TRAINING_INPUT_DIR, exist_ok=True)

print("🔓 Starting dataset decryption sweep...")

for file_name in os.listdir(ENCRYPTED_DIR):
    if file_name.endswith(".enc"):
        enc_path = os.path.join(ENCRYPTED_DIR, file_name)
        
        # Read the encrypted bytes
        with open(enc_path, "rb") as enc_file:
            encrypted_data = enc_file.read()
            
        try:
            # Decrypt back into raw JPG bytes in RAM
            decrypted_bytes = fernet.decrypt(encrypted_data)
            
            # Convert bytes directly into an OpenCV image matrix object
            np_array = np.frombuffer(decrypted_bytes, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            # Save out to your temporary local staging directory for the training loop
            output_file_name = file_name.replace(".enc", ".jpg")
            cv2.imwrite(os.path.join(TRAINING_INPUT_DIR, output_file_name), img)
            
        except Exception as e:
            # 🟢 This will tell us the exact cryptographic reason it's failing (e.g., InvalidToken)
            print(f"⚠️ Failed to decrypt {file_name}. Error details: {type(e).__name__} - {e}")
            
print("✨ Dataset prepped and ready for training pass!")
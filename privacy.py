# privacy.py
import os
import cv2
import numpy as np
import time
from datetime import datetime
# --- UNCOMMENT TO REACTIVATE ENCRYPTION ---
# from cryptography.fernet import Fernet
from telemetry import log_system_telemetry

# =====================================================================
# CRYPTOGRAPHIC INITIALIZATION LAYER (COMMENTED OUT)
# =====================================================================
# try:
#     # Attempt to load the permanent master token from the hidden disk file
#     with open(".master_secret.key", "rb") as key_file:
#         ENCRYPTION_KEY = key_file.read()
#     fernet_engine = Fernet(ENCRYPTION_KEY)
# except Exception:
#     # Runtime fallback: generates a session-only key if the file is missing
#     ENCRYPTION_KEY = Fernet.generate_key()
#     fernet_engine = Fernet(ENCRYPTION_KEY)
#     print("⚠️ WARNING: Operating with temporary encryption keys. Captured files cannot be decrypted after reboot!")

# Ensure the storage directory exists safely
EDGE_CASE_DIR = "harvested_edge_cases"
os.makedirs(EDGE_CASE_DIR, exist_ok=True)


# =====================================================================
# CORE PRIVACY PROCESSING ENGINE
# =====================================================================
def save_anonymized_and_encrypted_frame(frame_or_crop, pose_results, prefix="sample", extra_suffix="", crop_offsets=None, is_rad_tech=False):
    """
    Saves a chest crop. Raw passthrough is currently active.
    To reactivate blurring or encryption, uncomment their respective sections below.
    """
    processed_img = frame_or_crop.copy()
    h, w, _ = processed_img.shape
    offset_x, offset_y = crop_offsets if crop_offsets else (0, 0)

    # =====================================================================
    # 📑 OPTIONAL: FACE BLURRING LAYER (COMMENTED OUT)
    # =====================================================================
    # # 🟢 RULE 1: If Identified as a Rad Tech, SKIP BLURRING COMPLETELY
    # if is_rad_tech:
    #     pass 
    #     
    # # Otherwise, run the face presence evaluation for patients / visitors
    # elif pose_results and len(pose_results) > 0:
    #     for result in pose_results:
    #         if result.keypoints is not None and len(result.keypoints.xy) > 0:
    #             keypoints = result.keypoints.xy[0].cpu().numpy()
    #             
    #             # Check the 5 facial landmarks relative to our crop box boundaries
    #             face_pts_in_crop = []
    #             upper_face_count = 0
    #             
    #             # Keypoint index mapping: 0=Nose, 1=Left Eye, 2=Right Eye, 3=Left Ear, 4=Right Ear
    #             for idx, kp in enumerate(keypoints[:5]):
    #                 if kp[0] > 0 and kp[1] > 0:
    #                     rel_x = kp[0] - offset_x
    #                     rel_y = kp[1] - offset_y
    #                     
    #                     # Verify if this specific keypoint lands inside the saved crop canvas
    #                     if 0 <= rel_x <= w and 0 <= rel_y <= h:
    #                         face_pts_in_crop.append([rel_x, rel_y])
    #                         if idx in [0, 1, 2]:  # Track upper features (Nose and Eyes)
    #                             upper_face_count += 1
    #             
    #             # 🟢 RULE 2: Only blur if a substantial portion of the upper face leaks into the crop
    #             if upper_face_count >= 2 and len(face_pts_in_crop) >= 3:
    #                 face_pts_in_crop = np.array(face_pts_in_crop)
    #                 min_xy = np.min(face_pts_in_crop, axis=0)
    #                 max_xy = np.max(face_pts_in_crop, axis=0)
    #                 
    #                 pad_x = int((max_xy[0] - min_xy[0]) * 0.5) + 15
    #                 pad_y = int((max_xy[1] - min_xy[1]) * 0.5) + 15
    #                 
    #                 x1, y1 = max(0, int(min_xy[0] - pad_x)), max(0, int(min_xy[1] - pad_y))
    #                 x2, y2 = min(w, int(max_xy[0] + pad_x)), min(h, int(max_xy[1] + pad_y))
    #                 
    #                 face_roi = processed_img[y1:y2, x1:x2]
    #                 if face_roi.size > 0:
    #                     processed_img[y1:y2, x1:x2] = cv2.GaussianBlur(face_roi, (51, 51), 15)

    timestamp_id = int(time.time())
    suffix_str = f"_{extra_suffix}" if extra_suffix else ""
    tech_status = "_TECH" if is_rad_tech else "_PATIENT"

    # =====================================================================
    # 🔒 OPTIONAL: ENCRYPTION WRITER LAYER (COMMENTED OUT)
    # =====================================================================
    # success, encoded_image = cv2.imencode('.jpg', processed_img)
    # if not success:
    #     return None
    #     
    # encrypted_data = fernet_engine.encrypt(encoded_image.tobytes())
    # file_name = f"{prefix}_{timestamp_id}{suffix_str}{tech_status}.enc"
    # file_path = os.path.join(EDGE_CASE_DIR, file_name)
    # 
    # try:
    #     with open(file_path, "wb") as enc_file:
    #         enc_file.write(encrypted_data)
    #     return file_name
    # except Exception as e:
    #     print(f"⚠️ Failed to write encrypted sample: {e}")
    #     return None

    # =====================================================================
    # 🟢 CURRENT: RAW PASSTHROUGH WRITER LAYER
    # =====================================================================
    file_name = f"{prefix}_{timestamp_id}{suffix_str}{tech_status}.jpg"
    file_path = os.path.join(EDGE_CASE_DIR, file_name)
    
    try:
        cv2.imwrite(file_path, processed_img)
        return file_name
    except Exception as e:
        print(f"⚠️ Failed to write raw sample image: {e}")
        return None
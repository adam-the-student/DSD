import cv2
import os
import time
import numpy as np
from picamera2 import Picamera2
import libcamera
from hailo_platform import VDevice, FormatType, HailoSchedulingAlgorithm

# --- SET UP DIRECTORIES ---
folder_choice = input("Enter folder name to save images to (e.g., 'badge' or 'no_badge'): ").strip()
save_dir = os.path.join("dataset", "raw", folder_choice)
os.makedirs(save_dir, exist_ok=True)

# ==============================================================================
#                         CAMERA CALIBRATION CONFIGURATION
# ==============================================================================
REAL_SHOULDER_WIDTH_INCHES = 17.0
FOCAL_LENGTH_FACTOR = 318 
# ==============================================================================

# --- INITIALIZE NATIVE HAILO-10H ACCELERATOR ENGINE ---
model_path = "yolov8s-pose-h10.hef"
print(f"Loading hardware network: {model_path} onto AI HAT+ 2...")

vdevice_params = VDevice.create_params()
vdevice_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

target_device = VDevice(vdevice_params)
infer_model = target_device.create_infer_model(model_path)

infer_model.input().set_format_type(FormatType.UINT8)
for output_name in infer_model.output_names:
    infer_model.output(output_name).set_format_type(FormatType.FLOAT32)

input_name = infer_model.input_names[0]

# --- INITIALIZE NATIVE RASPBERRY PI CAMERA BACKEND ---
print("Initializing Picamera2 pipeline...")
picam = Picamera2()

config = picam.create_video_configuration(main={"size": (640, 480)})
config["transform"] = libcamera.Transform(hflip=True, vflip=True)
picam.configure(config)
picam.start()

Offset = -60

# --- INITIALIZE SMART COUNTER ---
existing_files = [f for f in os.listdir(save_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
img_counter = len(existing_files)

# Map out multi-scale layers alongside their mathematical downsampling pixel strides
layer_pairs = [
    {"score": "yolov8s_pose/conv71", "keypoint": "yolov8s_pose/conv72", "stride": 32},  # 20x20 -> 640/20 = 32
    {"score": "yolov8s_pose/conv58", "keypoint": "yolov8s_pose/conv59", "stride": 16},  # 40x40 -> 640/40 = 16
    {"score": "yolov8s_pose/conv44", "keypoint": "yolov8s_pose/conv45", "stride": 8}    # 80x80 -> 640/80 = 8
]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

try:
    with infer_model.configure() as configured_infer_model:
        bindings = configured_infer_model.create_bindings()
        
        output_buffers = {}
        for name in infer_model.output_names:
            output_buffers[name] = np.empty(infer_model.output(name).shape, dtype=np.float32)
            bindings.output(name).set_buffer(output_buffers[name])

        while True:
            raw_frame = picam.capture_array()
            frame = np.ascontiguousarray(raw_frame)
            
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            clean_frame = frame.copy()
            
            npu_input_frame = cv2.resize(frame, (640, 640))
            input_array = np.expand_dims(npu_input_frame, axis=0).astype(np.uint8)
            bindings.input(input_name).set_buffer(input_array)
            
            configured_infer_model.run([bindings], timeout=1000)
            
            current_distance = None
            chest_crop_square = None
            
            best_score = -999.0
            best_ls_x, best_ls_y = 0, 0
            best_rs_x, best_rs_y = 0, 0

            # --- DECODE MULTI-SCALE GRIDS WITH SPATIAL ANCHORING ---
            for pair in layer_pairs:
                score_tensor = output_buffers[pair["score"]]      
                keypoint_tensor = output_buffers[pair["keypoint"]]  
                stride = pair["stride"]
                
                H, W, _ = score_tensor.shape
                
                for h in range(H):
                    for w in range(W):
                        raw_score = score_tensor[h, w, 0]
                        
                        if raw_score > best_score:
                            kp_data = keypoint_tensor[h, w]
                            
                            # Extract localized grid anchor offsets
                            ls_x_offset = kp_data[15]
                            ls_y_offset = kp_data[16]
                            rs_x_offset = kp_data[18]
                            rs_y_offset = kp_data[19]
                            
                            # Reconstruct global 640x640 input resolution landmarks mathematically:
                            # Center-point location = ((center_offset * 2) + grid_index) * stride
                            global_ls_x = ((ls_x_offset * 2.0) + w) * stride
                            global_ls_y = ((ls_y_offset * 2.0) + h) * stride
                            global_rs_x = ((rs_x_offset * 2.0) + w) * stride
                            global_rs_y = ((rs_y_offset * 2.0) + h) * stride
                            
                            best_score = raw_score
                            
                            # Map coordinates back into our 640x480 video canvas smoothly
                            best_ls_x = int(global_ls_x)
                            best_ls_y = int(global_ls_y * (480 / 640))
                            best_rs_x = int(global_rs_x)
                            best_rs_y = int(global_rs_y * (480 / 640))

            final_conf = sigmoid(best_score)

            # --- DRAW STABLE BOUNDING BOX ---
            if final_conf > 0.35 and best_ls_x > 0 and best_rs_x > 0:
                print(f"Tracking Shoulders! Conf: {final_conf:.2f} | LS: ({best_ls_x},{best_ls_y}) RS: ({best_rs_x},{best_rs_y})     ", end="\r")
                
                pixel_width = np.sqrt((best_ls_x - best_rs_x)**2 + (best_ls_y - best_rs_y)**2)
                if pixel_width > 0:
                    current_distance = (REAL_SHOULDER_WIDTH_INCHES * FOCAL_LENGTH_FACTOR) / pixel_width
                
                box_width = int(pixel_width * 0.65)
                box_height = box_width  
                
                shoulder_center_x = min(best_ls_x, best_rs_x) + (abs(best_ls_x - best_rs_x) // 2)
                x_min = shoulder_center_x - (box_width // 2)
                x_max = x_min + box_width
                
                y_min = min(best_ls_y, best_rs_y) + Offset
                y_max = y_min + box_height
                
                is_in_range = current_distance is not None and (current_distance / 12.0) <= 3.5
                box_color = (0, 0, 255) if is_in_range else (255, 0, 0)
                
                cv2.rectangle(frame, (max(0, x_min), max(0, y_min)), (min(640, x_max), min(480, y_max)), box_color, 2)
                cv2.circle(frame, (best_ls_x, best_ls_y), 5, (0, 255, 0), -1)
                cv2.circle(frame, (best_rs_x, best_rs_y), 5, (0, 255, 0), -1)
                
                if is_in_range:
                    crop = clean_frame[max(0, y_min):min(480, y_max), max(0, x_min):min(640, x_max)]
                    if crop.size > 0:
                        chest_crop_square = cv2.resize(crop, (224, 224))
                        cv2.imshow("Live Target Crop Preview", chest_crop_square)
            else:
                print("Status: Searching for shoulders...                                                               ", end="\r")

            cv2.imshow("Data Collector Sandbox", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32:  # SPACEBAR
                if chest_crop_square is not None:
                    img_name = f"crop_{int(time.time())}_{img_counter}.jpg"
                    cv2.imwrite(os.path.join(save_dir, img_name), chest_crop_square)
                    print(f"\n[SAVED] Captured and Saved: {img_name}")
                    img_counter += 1
                else:
                    print("\n[ERROR] Capture failed: Target box must be active and RED!")

finally:
    print("\nShutting down camera channels cleanly...")
    picam.stop()
    cv2.destroyAllWindows()
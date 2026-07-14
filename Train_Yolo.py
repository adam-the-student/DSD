from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 classification baseline
model = YOLO('yolov8n-cls.pt')

# # 2. Run the native training engine with aggressive minority class balancing
print("Starting YOLOv8 training routine with heavy target augmentation...")
model.train(
    data='dataset',          # Point here so YOLO finds the required /train folder
    epochs=50,               # 💡 BUMPED TO 50: Gives the network time to learn from the synthetic variations
    imgsz=224,
    batch=4,                 # Small batch size keeps memory clean on the Pi 5
    device='cpu',
    
    # --- 🟢 MINORITY OVERSAMPLING HYPERPARAMETERS ---
    scale=0.7,               # Randomly zooms badges in/out from 30% to 100% scale bounds
    hsv_v=0.6,               # Shifts brightness drastically (simulates shadows vs flashlight flares)
    hsv_s=0.5,               # Adjusts color saturation (handles different shirt dye/fabric colors)
    fliplr=0.5,              # Horizontally mirrors the crop on half the training loops
    auto_augment='randaugment', # Forces the automated RandAugment policy stack for classification
    
    # --- 🧠 TRAINING STABILITY ---
    cache=True               # Caches images in RAM to offset slow disk reading speeds on edge devices
)
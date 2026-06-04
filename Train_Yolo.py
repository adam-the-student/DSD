from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 classification baseline
model = YOLO('yolov8n-cls.pt')

# 2. Run the native training engine on your relative data path
model.train(
    data='dataset/raw',  # Clean relative path instead of C:/Users/...
    epochs=10,
    imgsz=224
)
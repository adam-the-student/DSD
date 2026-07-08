from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 classification baseline
model = YOLO('yolov8n-cls.pt')

# 2. Run the native training engine on your structured data path
print("Starting YOLOv8 training routine...")
model.train(
    data='dataset',  # Point here so YOLO finds the required /train folder
    epochs=10,
    imgsz=224,
    batch=4,                  # Small batch size keeps memory clean on the Pi 5
    device='cpu'
)

# 3. Export out to ONNX format right away for the Hailo Dataflow Compiler
print("\nExporting trained network to ONNX...")
onnx_path = model.export(format='onnx', imgsz=224)

print(f"\n[SUCCESS] Model compiled to: {onnx_path}")
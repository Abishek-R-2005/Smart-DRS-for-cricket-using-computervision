from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # You can choose different models like yolov8s, yolov8m, yolov8l, yolov8x

# Train the model
results = model.train(data="yolov8-obb/data.yaml", epochs=100, imgsz=640, batch=16, name="yolov8_custom")

model.export(format="onnx")
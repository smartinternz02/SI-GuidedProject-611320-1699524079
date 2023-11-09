from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='pistolYolo\data.yaml',epochs = 10)

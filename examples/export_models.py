from ultralytics import YOLO

model = YOLO('./models/yolov8n.pt')

model.export(format='tflite', int8=True)
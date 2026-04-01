from ultralytics import YOLO
import cv2

class FaceDetector:
    def __init__(self, model_path="yolov8n-face.onnx"):
        # Load YOLOv8 ONNX model for optimized inference
        self.model = YOLO(model_path)

    def detect_faces(self, frame):
        results = self.model(frame)
        faces = []
        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            faces.append((x1, y1, x2, y2))
        return faces

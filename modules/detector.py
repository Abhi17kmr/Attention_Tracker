from ultralytics import YOLO
import cv2
import mediapipe as mp

class HybridDetector:
    def __init__(self, yolo_model="yolov8n.pt"):
        # Load YOLOv8
        self.yolo = YOLO(yolo_model)
        # Initialize MediaPipe FaceMesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    def detect(self, frame):
        """
        Run YOLOv8 to get bounding boxes, then refine with MediaPipe landmarks.
        Returns bounding boxes + landmarks.
        """
        results = self.yolo(frame)
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            # Run MediaPipe inside ROI
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            mp_results = self.face_mesh.process(rgb)

            if mp_results.multi_face_landmarks:
                for face_landmarks in mp_results.multi_face_landmarks:
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "landmarks": face_landmarks
                    })

        return detections

import cv2
import numpy as np
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self, model_path="models/emotion_model.h5"):
        """
        Initialize emotion detector by loading Advisor-ai's CNN model.
        """
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Could not load emotion model: {e}")
            self.model = None

        # Standard FER2013 emotion labels
        self.labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def detect(self, frame):
        """
        Detect emotion from the given frame using Advisor-ai's model.
        """
        if not self.model:
            return "Neutral"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(gray, (48, 48)) / 255.0
        face_roi = np.expand_dims(face_roi, axis=(0, -1))  # shape (1,48,48,1)

        pred = self.model.predict(face_roi)
        emotion_label = np.argmax(pred)
        return self.labels[emotion_label]

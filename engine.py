from .features import extract_features
from .model import AttentionModel

class AttentionEngine:
    def __init__(self):
        self.model = AttentionModel()

    def process_frame(self, frame):
        features, landmarks = extract_features(frame)

        if not features:
            return frame, 0, "NO FACE", None

        score = self.model.predict(features)

        if score > 70:
            status = "FOCUSED"
        elif score < 40:
            status = "DISTRACTED"
        else:
            status = "MODERATE"

        return frame, score, status, landmarks
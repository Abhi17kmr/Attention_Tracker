import joblib

class AttentionScorer:
    def __init__(self, alpha=0.2, model_path="attention.pkl"):
        self.value = 0
        self.alpha = alpha
        try:
            self.model = joblib.load(model_path)
        except:
            self.model = None

    def update(self, ear, head_status, gaze):
        ear_norm = 1 if ear > 0.2 else 0
        pose_stability = 1 if head_status == "CENTER" else 0
        gaze_center = 1 if gaze == "CENTER" else 0

        raw_attention = 0.5*ear_norm + 0.3*pose_stability + 0.2*gaze_center

        # If ML model exists, use it
        if self.model:
            features = [[ear_norm, pose_stability, gaze_center]]
            pred = self.model.predict(features)[0]
            self.value = pred
        else:
            self.value = self.alpha * raw_attention + (1 - self.alpha) * self.value

        return self.value

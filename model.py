import joblib
import numpy as np
import os

class AttentionModel:
    def __init__(self, path="models/attention.pkl"):
        self.use_ml = os.path.exists(path)

        if self.use_ml:
            self.model = joblib.load(path)
        else:
            print("⚠️ No ML model found, using fallback")

    def predict(self, f):
        if self.use_ml:
            X = np.array([
                f["blink_rate"],
                f["ear"],
                f["yaw"],
                f["eye_closure"]
            ]).reshape(1, -1)

            prob = self.model.predict_proba(X)[0][1]
            return int(prob * 100)

        # fallback scoring
        score = 100
        score -= f["blink_rate"] * 1.2
        score -= abs(f["yaw"]) * 1.5
        score -= f["eye_closure"] * 10

        return max(0, min(100, int(score)))
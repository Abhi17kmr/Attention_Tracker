import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class AttentionModel:
    def __init__(self, path="models/attention.pkl"):
        self.use_ml = os.path.exists(path)
        self.scaler = StandardScaler()

        if self.use_ml:
            self.model = joblib.load(path)
        else:
            print("⚠️ No ML model found, using fallback")

    def predict(self, f):
        X = np.array([
            f["ear"],
            f["yaw"],
            f["pitch"],
            f["roll"],
            f["eye_closure"],
            f.get("blink_frequency", 0)
        ]).reshape(1, -1)

        if self.use_ml:
            X_scaled = self.scaler.fit_transform(X)
            prob = self.model.predict_proba(X_scaled)[0][1]
            return int(prob * 100)

        # Fallback scoring
        score = 100
        score -= abs(f["yaw"]) * 0.8
        score -= abs(f["pitch"]) * 0.8
        score -= abs(f["roll"]) * 0.5
        score -= f["eye_closure"] * 20
        score -= f.get("blink_frequency", 0) * 0.5
        return max(0, min(100, int(score)))

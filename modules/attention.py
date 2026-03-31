import joblib

class AttentionScorer:
    def __init__(self, alpha=0.2, model_path="attention.pkl"):
        self.value = 0
        self.alpha = alpha
        try:
            self.model = joblib.load(model_path)
        except:
            self.model = None

    def update(self, ear, head_status, gaze, emotion, posture):
        """
        Update attention score using fatigue (EAR), head pose, gaze,
        plus emotion and posture signals.
        """

        # Normalize basic signals
        ear_norm = 1 if ear > 0.2 else 0
        pose_stability = 1 if head_status == "CENTER" else 0
        gaze_center = 1 if gaze == "CENTER" else 0

        # Emotion weighting (example heuristic)
        emotion_weights = {
            "Happy": 1.0,
            "Neutral": 0.8,
            "Surprise": 0.7,
            "Sad": 0.5,
            "Fear": 0.4,
            "Angry": 0.3,
            "Disgust": 0.3
        }
        emotion_score = emotion_weights.get(emotion, 0.7)

        # Posture weighting (example heuristic)
        posture_weights = {
            "Upright": 1.0,
            "Lean": 0.7,
            "Slouch": 0.5,
            "Unknown": 0.6
        }
        posture_score = posture_weights.get(posture, 0.7)

        # Fusion model: combine all signals
        raw_attention = (
            0.4 * ear_norm +
            0.2 * pose_stability +
            0.2 * gaze_center +
            0.1 * emotion_score +
            0.1 * posture_score
        )

        # If ML model exists, use it
        if self.model:
            features = [[ear_norm, pose_stability, gaze_center, emotion_score, posture_score]]
            pred = self.model.predict(features)[0]
            self.value = pred
        else:
            # Exponential smoothing
            self.value = self.alpha * raw_attention + (1 - self.alpha) * self.value

        return self.value

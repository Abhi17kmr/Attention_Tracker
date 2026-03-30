import time
import csv
import os
from .features import extract_features
from .model import AttentionModel

class AttentionEngine:
    def __init__(self):
        self.model = AttentionModel()
        self.baseline_ear = None
        self.frame_count = 0
        self.blink_count = 0
        self.start_time = time.time()
        self.last_eye_state = 0  # 0=open, 1=closed

        # CSV logging setup
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "attention_log.csv")

        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Frame", "EAR", "Yaw", "Pitch", "Roll",
                    "Eye Closure", "Blink Frequency", "Score", "Status"
                ])

    def process_frame(self, frame):
        features, landmarks = extract_features(frame, self.baseline_ear)

        if not features:
            return frame, 0, "NO FACE", None

        if self.frame_count < 50:
            self.baseline_ear = features["ear"] if not self.baseline_ear else \
                                (self.baseline_ear + features["ear"]) / 2
        self.frame_count += 1

        if self.last_eye_state == 0 and features["eye_closure"] == 1:
            self.last_eye_state = 1
        elif self.last_eye_state == 1 and features["eye_closure"] == 0:
            self.blink_count += 1
            self.last_eye_state = 0

        elapsed_minutes = (time.time() - self.start_time) / 60.0
        blink_frequency = self.blink_count / elapsed_minutes if elapsed_minutes > 0 else 0
        features["blink_frequency"] = blink_frequency

        score = self.model.predict(features)

        if score > 70:
            status = "FOCUSED"
        elif score < 40:
            status = "DISTRACTED"
        else:
            status = "MODERATE"

        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.frame_count,
                round(features["ear"], 4),
                round(features["yaw"], 2),
                round(features["pitch"], 2),
                round(features["roll"], 2),
                features["eye_closure"],
                round(features["blink_frequency"], 2),
                score,
                status
            ])

        return frame, score, status, landmarks

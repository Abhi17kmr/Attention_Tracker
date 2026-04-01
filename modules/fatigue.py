import numpy as np
from collections import deque

class FatigueDetector:
    def __init__(self, window_size=10):
        self.blink_history = deque(maxlen=window_size)
        self.baseline_ear = None

    def calibrate(self, ear_values):
        # Establish baseline EAR per user
        self.baseline_ear = np.mean(ear_values)

    def detect_blink(self, ear):
        if self.baseline_ear is None:
            return False
        threshold = self.baseline_ear * 0.75  # adaptive threshold
        blink = ear < threshold
        self.blink_history.append(int(blink))
        return blink

    def smoothed_blink_rate(self):
        return np.mean(self.blink_history) * 60  # blinks per minute

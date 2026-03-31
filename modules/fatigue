from collections import deque
import time
from scipy.spatial import distance as dist

class FatigueDetector:
    def __init__(self, ear_thresh=0.21, mar_thresh=0.6, window_size=60):
        self.blink_flag = False
        self.eye_closure_time = 0
        self.ear_thresh = ear_thresh
        self.mar_thresh = mar_thresh
        self.yawn_count = 0
        self.blink_times = deque()
        self.window_size = window_size

    def compute_ear(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def compute_mar(self, mouth):
        A = dist.euclidean(mouth[1], mouth[5])
        B = dist.euclidean(mouth[2], mouth[4])
        C = dist.euclidean(mouth[0], mouth[3])
        return (A + B) / (2.0 * C)

    def update(self, left_eye, right_eye, mouth):
        ear = (self.compute_ear(left_eye) + self.compute_ear(right_eye)) / 2.0
        mar = self.compute_mar(mouth)

        if ear < self.ear_thresh:
            if not self.blink_flag:
                self.blink_flag = True
                self.blink_times.append(time.time())
            self.eye_closure_time += 1
        else:
            self.blink_flag = False
            self.eye_closure_time = 0

        if mar > self.mar_thresh:
            self.yawn_count += 1

        now = time.time()
        while self.blink_times and now - self.blink_times[0] > self.window_size:
            self.blink_times.popleft()

        blink_rate = len(self.blink_times) * (60 / self.window_size)
        return ear, blink_rate, self.eye_closure_time, self.yawn_count

    def compute_fatigue_index(self, blink_rate, eye_closure_time, yawn_count, yaw):
        score = 0.4 * blink_rate + 0.3 * eye_closure_time + 0.2 * yawn_count + 0.1 * abs(yaw)
        if score > 20: return "HIGH"
        elif score > 10: return "MEDIUM"
        else: return "LOW"

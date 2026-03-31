import cv2
import mediapipe as mp

class PostureTracker:
    def __init__(self):
        """
        Initialize posture tracker using MediaPipe Pose (Advisor-ai style).
        """
        self.pose = mp.solutions.pose.Pose()

    def track(self, frame):
        """
        Track posture from the given frame using skeleton landmarks.
        """
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return "Unknown"

        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[11]
        left_hip = landmarks[23]

        # Simple heuristic: shoulder vs hip vertical distance
        if (left_shoulder.y - left_hip.y) < 0.1:
            return "Slouch"
        else:
            return "Upright"

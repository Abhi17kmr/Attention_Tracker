import cv2
import numpy as np
import time

# Import hybrid detector
from modules.detector import HybridDetector

# Import modules
from modules.fatigue import FatigueDetector
from modules.headpose import HeadPoseEstimator
from modules.gaze import GazeEstimator
from modules.attention import AttentionScorer
from modules.analytics import SessionAnalytics
from modules.emotion import EmotionDetector
from modules.posture import PostureTracker

# Import utils
from utils.drawing import draw_eye_pupils, draw_overlay
from utils.io import CSVLogger, VideoRecorder

# -----------------------------
# Setup
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

recording = False

# Initialize modules
detector = HybridDetector()  # YOLOv8 + MediaPipe
fatigue_detector = FatigueDetector()
head_pose_estimator = HeadPoseEstimator()
gaze_estimator = GazeEstimator()
attention_scorer = AttentionScorer()
session_analytics = SessionAnalytics()
emotion_detector = EmotionDetector()
posture_tracker = PostureTracker()

# Initialize IO
csv_logger = CSVLogger("data/dataset.csv")
video_recorder = VideoRecorder("data/output.avi", (640, 480))

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    attention = 0
    fatigue = "LOW"
    head_status = "CENTER"
    gaze = "CENTER"
    blink_rate = 0
    ear = 0
    yaw = 0
    yawn_count = 0
    emotion = "Neutral"
    posture = "Upright"

    # Hybrid detection
    detections = detector.detect(frame)

    for det in detections:
        face_landmarks = det["landmarks"]
        h, w, _ = frame.shape

        # Extract eye/mouth landmarks
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        MOUTH = [61, 81, 311, 291, 13, 14]

        left_eye = [(int(face_landmarks.landmark[idx].x * w),
                     int(face_landmarks.landmark[idx].y * h)) for idx in LEFT_EYE]
        right_eye = [(int(face_landmarks.landmark[idx].x * w),
                      int(face_landmarks.landmark[idx].y * h)) for idx in RIGHT_EYE]
        mouth = [(int(face_landmarks.landmark[idx].x * w),
                  int(face_landmarks.landmark[idx].y * h)) for idx in MOUTH]

        # Fatigue detection
        ear, blink_rate, eye_closure_time, yawn_count = fatigue_detector.update(left_eye, right_eye, mouth)

        # Head pose
        yaw, pitch, roll = head_pose_estimator.estimate_pose(face_landmarks.landmark, frame)
        if yaw < -10:
            head_status = "LEFT"
        elif yaw > 10:
            head_status = "RIGHT"
        elif pitch > 15:
            head_status = "DOWN"
        elif roll > 15:
            head_status = "TILT"
        else:
            head_status = "CENTER"

        # Gaze estimation
        eye_img = gaze_estimator.extract_eye_region(frame, left_eye)
        if eye_img is not None:
            gaze, pupil_coords = gaze_estimator.estimate_gaze(eye_img)
        else:
            pupil_coords = (0, 0)

        # Attention scoring (now includes emotion + posture)
        emotion = emotion_detector.detect(frame)
        posture = posture_tracker.track(frame)
        attention = attention_scorer.update(ear, head_status, gaze, emotion, posture)

        # Fatigue index
        fatigue = fatigue_detector.compute_fatigue_index(blink_rate, eye_closure_time, yawn_count, yaw)

        # Draw pupils
        frame = draw_eye_pupils(frame, left_eye, right_eye)

        # Update session analytics
        session_analytics.update(attention, fatigue, blink_rate, yawn_count, pupil_coords, emotion, posture)

        # Save data if recording
        if recording:
            csv_logger.log([blink_rate, ear, yaw, eye_closure_time, yawn_count,
                            attention, fatigue, pupil_coords[0], pupil_coords[1],
                            emotion, posture])

    # Overlay UI
    frame = draw_overlay(frame, attention, fatigue, blink_rate, yawn_count, head_status, gaze, emotion, posture)

    # Write video
    video_recorder.write(frame)
    cv2.imshow("Attention Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        recording = not recording
    if key == 27:  # ESC
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
video_recorder.release()
csv_logger.close()
cv2.destroyAllWindows()

# Print session summary
session_analytics.summary()

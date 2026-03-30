import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

mp_mesh = mp.solutions.face_mesh
mesh = mp_mesh.FaceMesh(refine_landmarks=True)

mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

LEFT = [33,160,158,133,153,144]
RIGHT = [362,385,387,263,373,380]

def ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A+B)/(2*C)

def extract_features(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)

    if not res.multi_face_landmarks:
        return None, None

    face = res.multi_face_landmarks[0]
    h, w, _ = frame.shape

    # -------- Landmarks for drawing --------
    landmarks = [(int(lm.x*w), int(lm.y*h)) for lm in face.landmark]

    # -------- Eyes --------
    left_eye = [(int(face.landmark[i].x*w),
                 int(face.landmark[i].y*h)) for i in LEFT]

    right_eye = [(int(face.landmark[i].x*w),
                  int(face.landmark[i].y*h)) for i in RIGHT]

    ear_val = (ear(left_eye)+ear(right_eye))/2

    # Head pose proxy
    nose = face.landmark[1]
    yaw = (nose.x - 0.5) * 100

    blink_rate = max(0, (0.3 - ear_val) * 100)

    features = {
        "blink_rate": blink_rate,
        "ear": ear_val,
        "yaw": yaw,
        "eye_closure": 1 if ear_val < 0.2 else 0
    }

    return features, landmarks
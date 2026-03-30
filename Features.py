import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

mp_mesh = mp.solutions.face_mesh
mesh = mp_mesh.FaceMesh(refine_landmarks=True)

LEFT = [33,160,158,133,153,144]
RIGHT = [362,385,387,263,373,380]

# 3D model points for head pose estimation
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0), # Left eye corner
    (225.0, 170.0, -135.0),  # Right eye corner
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

def ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A+B)/(2*C)

def get_head_pose(face, h, w):
    image_points = np.array([
        (face.landmark[i].x * w, face.landmark[i].y * h) for i in LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4,1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rmat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rmat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [float(angle) for angle in euler_angles]
    return pitch, yaw, roll

def extract_features(frame, baseline_ear=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)

    if not res.multi_face_landmarks:
        return None, None

    face = res.multi_face_landmarks[0]
    h, w, _ = frame.shape

    landmarks = [(int(lm.x*w), int(lm.y*h)) for lm in face.landmark]

    left_eye = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in LEFT]
    right_eye = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in RIGHT]

    ear_val = (ear(left_eye)+ear(right_eye))/2

    if baseline_ear:
        eye_closure = 1 if ear_val < baseline_ear * 0.75 else 0
    else:
        eye_closure = 1 if ear_val < 0.2 else 0

    pitch, yaw, roll = get_head_pose(face, h, w)

    features = {
        "ear": ear_val,
        "eye_closure": eye_closure,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll
    }

    return features, landmarks

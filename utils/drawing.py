import cv2
import numpy as np

def draw_attention_bar(frame, score):
    h, w = frame.shape[:2]
    bar_length = int((score / 100) * w)
    color = (0, 255, 0) if score > 70 else (0, 255, 255) if score > 40 else (0, 0, 255)
    cv2.rectangle(frame, (0, h-20), (bar_length, h), color, -1)
    return frame

def draw_gaze_heatmap(frame, gaze_points):
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    for (x, y) in gaze_points:
        cv2.circle(heatmap, (x, y), 15, 1, -1)
    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

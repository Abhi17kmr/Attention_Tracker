import cv2

class GazeEstimator:
    def extract_eye_region(self, frame, eye_points):
        x_coords = [p[0] for p in eye_points]
        y_coords = [p[1] for p in eye_points]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        eye_img = frame[y1:y2, x1:x2]
        if eye_img.size == 0: return None
        return cv2.resize(eye_img, (60, 30))

    def estimate_gaze(self, eye_img):
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        min_val, _, min_loc, _ = cv2.minMaxLoc(gray)
        px, py = min_loc
        h, w = gray.shape

        if px < w/3: horiz = "LEFT"
        elif px > 2*w/3: horiz = "RIGHT"
        else: horiz = "CENTER"

        if py < h/3: vert = "UP"
        elif py > 2*h/3: vert = "DOWN"
        else: vert = "CENTER"

        if horiz == "CENTER" and vert == "CENTER":
            return "CENTER", (px, py)
        return f"{vert}-{horiz}", (px, py)

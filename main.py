import cv2
from modules.detector import FaceDetector
from modules.fatigue import FatigueDetector
from modules.analytics import AnalyticsLogger
from utils.drawing import draw_attention_bar, draw_gaze_heatmap

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    fatigue = FatigueDetector()
    logger = AnalyticsLogger()
    gaze_points = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        # Example: process first face for demo
        if faces:
            x1, y1, x2, y2 = faces[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Dummy values for demo
        attention_score = 75
        blink_rate = fatigue.smoothed_blink_rate()
        gaze = "center"
        head_pose = {"yaw": 0, "pitch": 0, "roll": 0}
        posture = "upright"

        # Log session data
        logger.log(attention_score, blink_rate, gaze, head_pose, posture)

        # Visualization
        frame = draw_attention_bar(frame, attention_score)
        frame = draw_gaze_heatmap(frame, gaze_points)

        cv2.imshow("Attention Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.export_csv()
    logger.export_json()

if __name__ == "__main__":
    main()

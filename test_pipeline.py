import cv2
from modules.detector import FaceDetector
from modules.fatigue import FatigueDetector
from modules.analytics import AnalyticsLogger

def run_test():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    fatigue = FatigueDetector()
    logger = AnalyticsLogger()

    print("Starting YOLOv8 + Analytics test... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detector.detect_faces(frame)
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Dummy EAR values for calibration (replace with real landmark EAR later)
        ear = 0.25
        fatigue.detect_blink(ear)
        blink_rate = fatigue.smoothed_blink_rate()

        # Log session data
        logger.log(
            attention_score=80, 
            blink_rate=blink_rate, 
            gaze="center", 
            head_pose={"yaw": 0, "pitch": 0, "roll": 0}, 
            posture="upright"
        )

        cv2.imshow("Test Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Export logs
    logger.export_csv()
    logger.export_json()
    print("Test complete. Logs saved to session_log.csv and session_log.json.")

if __name__ == "__main__":
    run_test()

import cv2
from attention.engine import AttentionEngine

engine = AttentionEngine()
path = input("Enter video path: ")
cap = cv2.VideoCapture(path)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, score, status, landmarks = engine.process_frame(frame)

    if landmarks:
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.rectangle(frame, (10,10), (300,140), (30,30,30), -1)
    cv2.putText(frame, f"Score: {score}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Status: {status}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

    cv2.imshow("Attention AI - Video", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

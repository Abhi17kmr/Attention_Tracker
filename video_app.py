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

    frame, score, status = engine.process_frame(frame)

    cv2.putText(frame, f"Score: {score}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Attention AI - Video", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2

def draw_eye_pupils(frame, left_eye, right_eye):
    """
    Draw subtle highlights on both eyes.
    """
    lx = int(sum([p[0] for p in left_eye]) / len(left_eye))
    ly = int(sum([p[1] for p in left_eye]) / len(left_eye))
    rx = int(sum([p[0] for p in right_eye]) / len(right_eye))
    ry = int(sum([p[1] for p in right_eye]) / len(right_eye))

    for (x, y) in [(lx, ly), (rx, ry)]:
        cv2.circle(frame, (x, y), 2, (0, 255, 200), -1)   # tiny filled circle
        cv2.circle(frame, (x, y), 4, (0, 255, 200), 1)   # thin outline

    return frame


def draw_overlay(frame, attention, fatigue, blink_rate, yawn_count,
                 head_status, gaze, emotion, posture):
    """
    Draw overlay with metrics, attention bar, emotion and posture indicators.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 340), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"Attention: {attention:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Fatigue: {fatigue}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Blink Rate: {blink_rate:.1f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Yawns: {yawn_count}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
    cv2.putText(frame, f"Head: {head_status}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Gaze: {gaze}", (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(frame, f"Emotion: {emotion}", (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
    cv2.putText(frame, f"Posture: {posture}", (20, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    # Attention bar
    bar_width = int(attention * 200)
    cv2.rectangle(frame, (20, 350), (220, 370), (255, 255, 255), 2)
    cv2.rectangle(frame, (20, 350), (20 + bar_width, 370), (0, 255, 0), -1)

    return frame

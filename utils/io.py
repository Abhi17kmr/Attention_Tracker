import csv
import cv2

class CSVLogger:
    def __init__(self, filename):
        self.file = open(filename, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "BlinkRate", "EAR", "Yaw", "EyeClosureTime", "YawnCount",
            "Attention", "Fatigue", "PupilX", "PupilY", "Emotion", "Posture"
        ])

    def log(self, row):
        self.writer.writerow(row)

    def close(self):
        self.file.close()


class VideoRecorder:
    def __init__(self, filename, frame_size, fps=20.0):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

    def write(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()

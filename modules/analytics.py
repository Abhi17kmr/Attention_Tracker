import csv, json, datetime

class AnalyticsLogger:
    def __init__(self, csv_path="session_log.csv", json_path="session_log.json"):
        self.csv_path = csv_path
        self.json_path = json_path
        self.records = []

    def log(self, attention_score, blink_rate, gaze, head_pose, posture):
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "attention_score": attention_score,
            "blink_rate": blink_rate,
            "gaze_direction": gaze,
            "head_pose": head_pose,
            "posture": posture
        }
        self.records.append(record)

    def export_csv(self):
        if not self.records: return
        keys = self.records[0].keys()
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)

    def export_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.records, f, indent=4)

import numpy as np

class SessionAnalytics:
    def __init__(self):
        self.attention_scores = []
        self.fatigue_levels = []
        self.blink_rates = []
        self.yawn_counts = []
        self.pupil_positions = []
        self.emotions = []
        self.postures = []

    def update(self, attention, fatigue, blink_rate, yawn_count, pupil_coords, emotion, posture):
        self.attention_scores.append(attention)
        self.fatigue_levels.append(fatigue)
        self.blink_rates.append(blink_rate)
        self.yawn_counts.append(yawn_count)
        self.pupil_positions.append(pupil_coords)
        self.emotions.append(emotion)
        self.postures.append(posture)

    def summary(self):
        avg_attention = np.mean(self.attention_scores) if self.attention_scores else 0
        avg_blink_rate = np.mean(self.blink_rates) if self.blink_rates else 0
        total_yawns = sum(self.yawn_counts)
        fatigue_distribution = {lvl: self.fatigue_levels.count(lvl) for lvl in set(self.fatigue_levels)}
        emotion_distribution = {e: self.emotions.count(e) for e in set(self.emotions)}
        posture_distribution = {p: self.postures.count(p) for p in set(self.postures)}

        print("\n--- Session Summary ---")
        print(f"Average Attention: {avg_attention:.2f}")
        print(f"Average Blink Rate: {avg_blink_rate:.1f} per min")
        print(f"Total Yawns: {total_yawns}")
        print(f"Fatigue Distribution: {fatigue_distribution}")
        print(f"Emotion Distribution: {emotion_distribution}")
        print(f"Posture Distribution: {posture_distribution}")
        print(f"Total Pupil Samples: {len(self.pupil_positions)}")

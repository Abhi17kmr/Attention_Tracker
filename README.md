# 🧠 Attention AI — Real-Time Attention Monitoring System

A **production-ready, real-time attention analysis system** built using **Computer Vision + Machine Learning**, designed to evaluate user focus through behavioral signals such as eye movement, blink patterns, and head orientation.

---

# 🚀 Overview

**Attention AI** analyzes live webcam input or recorded video to generate a unified **Attention Score (0–100)**.
It combines multiple human behavioral cues into a single interpretable metric, making it suitable for:

* 🎓 Online learning analytics
* 💼 Workplace productivity monitoring
* 🧪 Behavioral research
* 🤖 Human-computer interaction systems

---

# 🎯 Core Features

### 👁️ Eye Tracking (EAR)

* Detects eye openness using **Eye Aspect Ratio**
* Helps identify drowsiness and fatigue

### 👀 Blink Rate Detection

* Tracks blink frequency over time
* High blink rate → fatigue or distraction indicator

### 🧭 Head Pose Estimation

* Uses facial landmarks to estimate **yaw (left/right movement)**
* Detects off-screen attention

### 😴 Eye Closure Detection

* Identifies prolonged eye closure (fatigue signal)

### 🧠 ML-Based Attention Score

* Combines all features into a **single score (0–100)**
* Uses:

  * Trained model (if available)
  * Intelligent fallback scoring (if model not present)

---

# 🧩 System Architecture

```bash
attention-ai/
│
├── attention/            # Core AI Engine
│   ├── engine.py         # Main processing pipeline
│   ├── features.py       # Feature extraction (CV)
│   ├── model.py          # ML + fallback scoring
│
├── app/                  # Applications
│   ├── webcam_app.py     # Real-time webcam analysis
│   ├── video_app.py      # Video file processing
│
├── models/               # Trained ML models
│   └── attention.pkl
│
├── demo/                 # Sample inputs
│   └── sample.mp4
│
├── requirements.txt
├── README.md
```

---

# ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/attention-ai.git
cd attention-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

## 🎥 Real-Time Webcam Mode

```bash
python app/webcam_app.py
```

### Output:

* Live facial landmarks
* Attention Score (0–100)
* Status: Focused / Moderate / Distracted

---

## 📂 Video File Analysis

```bash
python app/video_app.py
```

Then enter:

```bash
path/to/video.mp4
```

---

# 🧠 Attention Score Logic

The system computes attention using:

* Blink Rate
* Eye Aspect Ratio (EAR)
* Head Pose (Yaw)
* Eye Closure Duration

### ML Mode:

If `models/attention.pkl` exists:

* Uses trained classifier
* Outputs probability-based score

### Fallback Mode:

If no model is present:

* Uses weighted heuristic scoring
* Ensures system always runs

---

# 🎨 Visual Output

* 🟢 Facial landmark overlay (468 points)
* 📊 Attention score bar
* 📍 Real-time tracking feedback

---

# 🔬 Tech Stack

* **OpenCV** — video processing
* **MediaPipe** — facial landmark detection
* **NumPy / SciPy** — numerical computation
* **Scikit-learn** — machine learning
* **Joblib** — model loading

---

# ⚡ Performance

* Real-time processing (~20–30 FPS)
* Lightweight and CPU-friendly
* No GPU required

---

# 🧪 Future Improvements

* 📈 Temporal models (LSTM / Transformers)
* 🖥️ Advanced GUI (PyQt Dashboard)
* 👥 Multi-face tracking
* ☁️ Cloud-based API deployment
* 📊 Attention analytics dashboard

---

# 🤝 Contribution

Contributions are welcome!

Steps:

1. Fork the repo
2. Create a new branch
3. Make changes
4. Submit a pull request

---

# 📜 License

This project is open-source and available under the MIT License.

---

# 🙌 Acknowledgements

* MediaPipe by Google
* OpenCV Community
* Scikit-learn

---

# 💡 Final Note

This project demonstrates how **simple behavioral signals + ML** can create powerful, real-time attention systems without heavy deep learning models.

---

# 🔥 Demo Ready

✔ Clean architecture
✔ Real-time + video support
✔ ML-integrated pipeline
✔ Visual feedback

---

👉 Built for performance, clarity, and rapid deployment.

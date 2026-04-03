# 🧠 Attention AI — Real-Time Behavioral Intelligence System

> A production-ready, real-time attention analysis system built with **Computer Vision** and **Machine Learning**.

---

## 📌 Overview

Attention AI evaluates user focus using multiple behavioral signals:

- 👁️ Eye movement
- 💤 Blink patterns
- 🧭 Head pose
- 🗺️ Facial landmarks

…and generates a unified **Attention Score (0–100)**.

> Designed as a **scalable AI system**, not just a demo.

---

## 🎥 Live Demo

<!-- Add your demo GIF or video link here -->
<!-- ![Demo](assets/demo.gif) -->

---

## 🏗️ Architecture

### System Pipeline

```
Video Input (Webcam / Video)
        ↓
Face Detection (YOLOv8 / MediaPipe)
        ↓
Facial Landmark Extraction
        ↓
Feature Extraction
 ├── Eye Aspect Ratio (EAR)
 ├── Blink Rate
 ├── Head Pose (Yaw / Pitch / Roll)
 └── Eye Closure Duration
        ↓
Attention Engine
        ↓
ML Model / Fallback Scoring
        ↓
Analytics & Logging
        ↓
Visualization Overlay
        ↓
Final Output (Score + UI)
```

---

## 📁 Project Structure

```
attention-ai/
│
├── attention/                 # Core AI Engine
│   ├── engine.py              # Main pipeline controller
│   ├── features.py            # EAR, blink, pose extraction
│   ├── model.py               # ML model + fallback scoring
│   ├── detector.py            # YOLOv8 / MediaPipe integration
│   ├── gaze.py                # Gaze tracking (optional extension)
│   ├── fatigue.py             # Fatigue logic
│   └── headpose.py            # Head pose estimation
│
├── utils/
│   ├── drawing.py             # UI overlays
│   └── io.py                  # Input/output handling
│
├── models/
│   └── attention.pkl          # Trained ML model
│
├── logs/
│   └── attention_log.csv      # Session logs
├── main.py                    # Optional unified runner
├── requirements.txt
└── README.md
```

---

## 🎯 Core Features

### 👁️ Eye Tracking (EAR)

- Computes **Eye Aspect Ratio** to measure openness
- Detects drowsiness and fatigue

### 👀 Blink Rate Detection

- Tracks blink frequency over time
- Identifies fatigue & distraction patterns

### 🧭 Head Pose Estimation

- Calculates **Yaw**, **Pitch**, and **Roll**
- Detects off-screen attention

### 😴 Eye Closure Detection

- Detects prolonged eye closure
- Strong fatigue signal

### 🧠 ML-Based Attention Score

| Mode | Description |
|------|-------------|
| **ML Mode** | Uses trained model (`models/attention.pkl`) — outputs probability-based attention score |
| **Fallback Mode** | Weighted heuristic scoring — ensures the system always runs even without a trained model |

### 📊 Analytics System

- Logs session data to `attention_log.csv`
- Enables dataset creation, model improvement, and behavioral insights

### 🎨 Visualization Layer

- Facial landmarks overlay
- Attention score display
- Real-time feedback: **Focused** / **Distracted**

---

## ▶️ Usage

### 🎥 Webcam Mode

```bash
python -m app.webcam_app
```

### 📂 Video Mode

```bash
python -m app.video_app
```

### 🔁 Optional Unified Run

```bash
python main.py
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Abhi17kmr/Attention_Tracker.git
cd Attention_Tracker
pip install -r requirements.txt
```

---

## 🔬 Tech Stack

| Technology | Purpose |
|------------|---------|
| **YOLOv8** (Ultralytics) | Face detection |
| **MediaPipe** | Facial landmarks |
| **OpenCV** | Video processing |
| **NumPy / SciPy** | Computation |
| **Scikit-learn** | ML model |
| **Joblib** | Model persistence |

---

## ⚡ Performance

- ~20–30 FPS on CPU
- Real-time processing
- Lightweight and efficient

---

### 🎯 Real-Time Tracking

<<img width="1138" height="552" alt="image" src="https://github.com/user-attachments/assets/14ace0dd-54f0-452d-b2bb-4e87c5d472d4" />
-->

## 🚀 Future Improvements

### 🔬 AI Enhancements

- LSTM / Transformer-based temporal models
- Personalized attention scoring

### 🖥️ Product Enhancements

- PyQt GUI dashboard
- Multi-face tracking
- Web-based deployment

### ☁️ System Scaling

- REST API backend
- Cloud deployment
- Analytics dashboard

---

## 📊 Use Cases

- 🎓 Online learning platforms
- 🏢 Workplace monitoring
- 🚗 Driver attention systems
- 🔬 Behavioral research

---

## 🤝 Contribution

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

**MIT License** — see [LICENSE](LICENSE) for details.

---

## 🔥 Why This Project Stands Out

| | |
|---|---|
| ✅ | Real-time AI system |
| ✅ | ML + fallback hybrid design |
| ✅ | Modular & scalable architecture |
| ✅ | Production-ready structure |
| ✅ | Strong portfolio + resume impact |

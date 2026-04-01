🧠 Attention AI-Real-Time Behavioral Intelligence System
📌 Overview

Attention AI is a production-ready, real-time attention analysis system built using Computer Vision and Machine Learning.

It evaluates user focus using multiple behavioral signals:

Eye movement
Blink patterns
Head pose
Facial landmarks

and generates a unified Attention Score (0–100).

Designed as a scalable AI system, not just a demo.

🎥 Live Demo

🏗️ Architecture

🧠 System Pipeline
Video Input (Webcam / Video)
        ↓
Face Detection (YOLOv8 / MediaPipe)
        ↓
Facial Landmark Extraction
        ↓
Feature Extraction
 ├── Eye Aspect Ratio (EAR)
 ├── Blink Rate
 ├── Head Pose (Yaw/Pitch/Roll)
 ├── Eye Closure Duration
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
🏗️ Project Structure (UPDATED)
attention-ai/
│
├── attention/                 # Core AI Engine
│   ├── engine.py              # Main pipeline controller
│   ├── features.py            # EAR, blink, pose extraction
│   ├── model.py               # ML model + fallback scoring
│   ├── detector.py            # (YOLOv8 / MediaPipe integration)
│   ├── gaze.py                # Gaze tracking (optional extension)
│   ├── fatigue.py             # Fatigue logic
│   ├── headpose.py            # Head pose estimation
│
├── app/                       # Application layer
│   ├── webcam_app.py          # Real-time webcam app
│   ├── video_app.py           # Video processing app
│
├── utils/
│   ├── drawing.py             # UI overlays
│   ├── io.py                  # Input/output handling
│
├── models/
│   └── attention.pkl          # Trained ML model
│
├── logs/
│   └── attention_log.csv      # Session logs
│
├── assets/                    # README visuals
│   ├── demo.gif
│   ├── architecture.png
│   ├── output_main.png
│
├── main.py                    # Optional unified runner
├── requirements.txt
├── README.md

🎯 Core Features
👁️ Eye Tracking (EAR)
Computes Eye Aspect Ratio
Detects drowsiness and fatigue
👀 Blink Rate Detection
Tracks blink frequency over time
Identifies fatigue & distraction
🧭 Head Pose Estimation
Calculates:
Yaw
Pitch
Roll
Detects off-screen attention
😴 Eye Closure Detection
Detects prolonged eye closure
Strong fatigue signal
🧠 ML-Based Attention Score
🔹 ML Mode
Uses trained model: models/attention.pkl
Outputs probability-based attention score
🔹 Fallback Mode
Weighted heuristic scoring
Ensures system always runs
📊 Analytics System
Logs session data (attention_log.csv)
Enables:
Dataset creation
Model improvement
Behavioral insights
🎨 Visualization Layer
Facial landmarks overlay
Attention score display
Real-time feedback (Focused / Distracted)
▶️ Usage
🎥 Webcam Mode
python -m app.webcam_app
📂 Video Mode
python -m app.video_app
🔁 Optional Unified Run
python main.py
⚙️ Installation
git clone https://github.com/Abhi17kmr/Attention_Tracker.git
cd Attention_Tracker

pip install -r requirements.txt
🔬 Tech Stack
YOLOv8 (Ultralytics) — Face detection
MediaPipe — Facial landmarks
OpenCV — Video processing
NumPy / SciPy — Computation
Scikit-learn — ML model
Joblib — Model persistence
⚡ Performance
~20–30 FPS (CPU)
Real-time processing
Lightweight and efficient
📸 Sample Outputs
🎯 Real-Time Tracking

🚀 Future Improvements
🔬 AI Enhancements
LSTM / Transformer-based temporal models
Personalized attention scoring
🖥️ Product Enhancements
PyQt GUI dashboard
Multi-face tracking
Web-based deployment
☁️ System Scaling
REST API backend
Cloud deployment
Analytics dashboard
📊 Use Cases
Online learning platforms
Workplace monitoring
Driver attention systems
Behavioral research
🤝 Contribution

Pull requests are welcome.

📜 License

MIT License

🔥 Why This Project Stands Out
✅ Real-time AI system
✅ ML + fallback hybrid design
✅ Modular & scalable architecture
✅ Production-ready structure
✅ Strong portfolio + resume impact

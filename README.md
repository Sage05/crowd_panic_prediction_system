# AI-Based Crowd Panic Prediction System

## Project Overview

The AI-Based Crowd Panic Prediction System is an intelligent crowd monitoring platform designed to analyze crowd movement, detect abnormal behavior, and assist in predicting panic situations before they escalate into dangerous incidents.

The system combines Computer Vision, Machine Learning, Backend APIs, and a Web Dashboard to create an end-to-end crowd safety solution.

---

# System Pipeline

```
CCTV / Video Feed
        |
        ↓
YOLOv8 Person Detection
        |
        ↓
ByteTrack Person Tracking
        |
        ↓
Trajectory Extraction
        |
        ↓
Behavior Analysis
        |
        ↓
Panic Prediction Model
        |
        ↓
Alert & Monitoring Dashboard
```

---

# Project Structure

```
panic_prediction_system/
│
├── cv/                         # Computer Vision Module
│   ├── __init__.py             # Makes cv a Python package
│   ├── config.py               # Centralized model and system settings
│   ├── detector.py             # YOLOv8 person detection and crowd counting
│   ├── tracker.py              # ByteTrack multi-person tracking
│   ├── trajectory_visualizer.py # Displays movement trails for tracked people
│   ├── export_data.py          # Exports person trajectories to CSV
│   └── evaluate_images.py      # Evaluates YOLO on ShanghaiTech Part B
│
├── ml/                         # Machine Learning Module
│   ├── anomaly.py              # Panic/anomaly detection models
│   ├── features.py             # Feature extraction for ML models
│   └── predictor.py            # Final panic prediction logic
│
├── backend/                    # FastAPI backend services
│
├── frontend/                   # React dashboard
│
├── data/
│   ├── samples/                # Sample videos for testing
│   ├── datasets/               # ShanghaiTech and other datasets
│   └── results/                # Evaluation output images
│
├── docs/                       # Documentation and reports
│
├── requirements.txt            # Python dependencies
│
└── README.md                   # Project documentation
```

---

# Installation Guide

## 1. Clone the Repository

```bash
git clone <repository-url>
cd panic_prediction_system
```

---

## 2. Create Virtual Environment

### Windows CMD

```cmd
python -m venv venv
```

Activate the environment:

```cmd
venv\Scripts\activate
```

Successful activation:

```
(venv) M:\panic_prediction_system>
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Verify CUDA Support

```bash
yolo checks
```

Expected output:

```
GPU: NVIDIA RTX GPU
CUDA: Enabled
```

---

# Running the Computer Vision Module

Make sure the terminal is opened in the project root:

```
panic_prediction_system/
```

---

## Person Detection

Runs YOLOv8 detection and displays crowd count.

```bash
python -m cv.detector
```

---

## Person Tracking

Tracks individuals with persistent IDs using ByteTrack.

```bash
python -m cv.tracker
```

---

## Trajectory Visualization

Displays movement paths of tracked people.

```bash
python -m cv.trajectory_visualizer
```

---

## Export Tracking Data

Creates CSV files containing person IDs and coordinates.

```bash
python -m cv.export_data
```

Output:

```
data/tracking_data.csv
```

---

## Evaluate on ShanghaiTech Part B

Runs YOLO on the crowd counting dataset and saves annotated results.

```bash
python -m cv.evaluate_images
```

Output:

```
data/results/
```

---

# Model Evaluation

The Computer Vision module was evaluated using the ShanghaiTech Part B dataset.

## Models Tested

- YOLOv8n (Nano)
- YOLOv8s (Small)

### Evaluation Summary

| Model | Speed | Detection Quality | Final Decision |
|-------|-------|-------------------|----------------|
| YOLOv8n | Very Fast | Comparable to YOLOv8s | ✅ Selected |
| YOLOv8s | Slower | Similar detection results | ❌ Not selected |

YOLOv8n was selected because it provided similar detection performance while maintaining better real-time speed, making it more suitable for a live CCTV-based panic monitoring system.

---

# Technologies Used

## Computer Vision

- YOLOv8
- ByteTrack
- OpenCV

## Machine Learning

- PyTorch
- ConvLSTM (planned)
- Anomaly Detection Models

## Backend

- FastAPI
- Python

## Frontend

- React.js

## Hardware Acceleration

- NVIDIA CUDA
- RTX GPU support

---

# Current Progress

## Completed

- [x] CUDA-enabled YOLOv8 setup
- [x] Real-time person detection
- [x] Crowd counting
- [x] ByteTrack person tracking
- [x] Trajectory visualization
- [x] CSV trajectory export
- [x] ShanghaiTech Part B evaluation
- [x] YOLOv8 model comparison

## In Progress

- [ ] ML anomaly detection
- [ ] ConvLSTM panic prediction
- [ ] Backend API integration
- [ ] Frontend dashboard

---

# Future Improvements

- Improve dense crowd detection
- Integrate live CCTV streams
- Add real-time panic risk scores
- Deploy as a complete web-based monitoring system

---

# Team Responsibilities

| Member | Responsibility |
|---------|---------------|
| Ayuj | Computer Vision (YOLO, tracking, trajectories, evaluation) |
| Neeraj | Machine Learning and panic prediction |
| Darshit | Backend APIs and frontend dashboard |

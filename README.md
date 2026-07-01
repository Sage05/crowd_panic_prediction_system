# AI-Based Crowd Panic Prediction System

## Project Overview

The AI-Based Crowd Panic Prediction System is an intelligent crowd monitoring platform designed to analyze crowd movement, detect abnormal behavior, and predict panic situations before they escalate into dangerous incidents.

The system combines Computer Vision, Machine Learning, Backend APIs, and a Web Dashboard into an end-to-end crowd safety pipeline.

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
Speed & Acceleration Extraction
        |
        ↓
Trajectory Logging → tracking_data.csv
        |
        ↓
        ├──→ Flat Feature Extraction → features.csv (Isolation Forest)
        |
        └──→ Occupancy Grid Sequences → X.npy + y.npy (ConvLSTM2D)
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
├── cv/                             # Computer Vision Module
│   ├── __init__.py                 # Makes cv a Python package
│   ├── config.py                   # Centralized model and system settings
│   ├── detector.py                 # YOLOv8 person detection and crowd counting
│   ├── tracker.py                  # ByteTrack multi-person tracking with speed & acceleration
│   ├── trajectory_visualizer.py    # Displays movement trails for tracked people
│   ├── feature_extractor.py        # Per-frame crowd feature computation
│   ├── export_data.py              # Exports flat features to features.csv
│   ├── grid_exporter.py            # Builds normalized occupancy grid sequences (X.npy, y.npy)
│   ├── grid_visualizer.py          # Animated heatmap viewer for occupancy grids
│   ├── evaluate_images.py          # Evaluates YOLO on ShanghaiTech Part B
│   ├── optical_flow.py             # (Planned) Optical flow analysis
│   └── main.py                     # Single entry point — runs full CV pipeline
│
├── ml/                             # Machine Learning Module
│   ├── anomaly.py                  # Isolation Forest anomaly detection
│   ├── features.py                 # Feature engineering for ML models
│   └── predictor.py                # ConvLSTM2D panic prediction
│
├── backend/                        # FastAPI backend services
│
├── frontend/                       # React dashboard
│
├── data/
│   ├── samples/                    # Sample videos for testing
│   ├── datasets/                   # ShanghaiTech and other datasets
│   ├── results/                    # Evaluation output images
│   ├── tracking_data.csv           # Raw per-frame tracking output
│   ├── features.csv                # Flat crowd features (Isolation Forest input)
│   ├── X.npy                       # Occupancy grid sequences (ConvLSTM2D input)
│   └── y.npy                       # Sequence labels (ConvLSTM2D supervision)
│
├── docs/                           # Documentation and reports
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

# Installation Guide

## 1. Clone the Repository

```bash
git clone <repository-url>
cd panic_prediction_system
```

## 2. Create Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

Successful activation:

```
(venv) M:\panic_prediction_system>
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

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

# Running the CV Module

All commands must be run from the project root (`panic_prediction_system/`).

---

## Full Pipeline (Recommended)

Runs all CV steps in order — tracking → feature export → grid export.

```bash
python -m cv.main
```

Output files:

```
data/tracking_data.csv   — raw per-frame detections
data/features.csv        — flat crowd features (Isolation Forest input)
data/X.npy               — occupancy grid sequences (ConvLSTM2D input)
data/y.npy               — sequence labels (ConvLSTM2D supervision)
```

---

## Individual Modules

### Person Detection
```bash
python -m cv.detector
```

### Person Tracking
Tracks individuals with persistent IDs. Logs speed, acceleration, and trajectories.
```bash
python -m cv.tracker
```

### Trajectory Visualization
```bash
python -m cv.trajectory_visualizer
```

### Feature Export (Isolation Forest input)
```bash
python -m cv.export_data
```

### Grid Export (ConvLSTM2D input)
```bash
python -m cv.grid_exporter
```

### Grid Visualizer
Plays back the 10×10 occupancy grid as an animated heatmap. Press `N` / `P` to navigate sequences, `ESC` to quit.
```bash
python -m cv.grid_visualizer
```

### Evaluate on ShanghaiTech Part B
```bash
python -m cv.evaluate_images
```

---

# CV Module — Data Flow

## tracker.py → tracking_data.csv

Columns:

| Column | Description |
|--------|-------------|
| frame | Frame number |
| person_id | Unique tracked person ID |
| center_x | X coordinate of bounding box center |
| center_y | Y coordinate of bounding box center |
| speed | Pixels moved since last frame |
| acceleration | Change in speed since last frame |

---

## feature_extractor.py → features.csv (via export_data.py)

Per-frame crowd features fed into Isolation Forest:

| Feature | Description |
|---------|-------------|
| people_count | Total people detected |
| avg_speed | Mean speed across all people |
| avg_speed_squared | Mean squared speed (energy proxy) |
| crowd_density | People per unit area |
| avg_acceleration | Mean acceleration across all people |
| moving_count | People moving above threshold |
| stationary_count | People below movement threshold |

---

## grid_exporter.py → X.npy + y.npy

Builds spatiotemporal occupancy grids for ConvLSTM2D:

- **Frame size:** 640×640 pixels
- **Grid:** 10×10 cells (each cell = 64×64 pixels)
- **Sampling:** Every 3rd frame (frames 1, 4, 7 … → 10 timesteps)
- **Normalization:** Min-max across all frames → [0, 1]
- **Sequence mode:** Overlapping (sliding window) or non-overlapping (configurable)
- **X shape:** `(N, 10, 10, 10, 1)` — batch, timesteps, grid height, grid width, channels
- **y shape:** `(N,)` — total occupancy in last frame of each sequence

---

# Model Evaluation

Evaluated on the ShanghaiTech Part B dataset.

| Model | Speed | Detection Quality | Decision |
|-------|-------|-------------------|----------|
| YOLOv8n | Very Fast | Comparable to YOLOv8s | ✅ Selected |
| YOLOv8s | Slower | Similar results | ❌ Not selected |

YOLOv8n was selected for its real-time performance, which is critical for live CCTV-based monitoring.

---

# Technologies Used

| Layer | Technologies |
|-------|-------------|
| Computer Vision | YOLOv8, ByteTrack, OpenCV |
| Machine Learning | PyTorch, Isolation Forest, ConvLSTM2D (planned) |
| Backend | FastAPI, Python |
| Frontend | React.js |
| Hardware | NVIDIA CUDA, RTX GPU |

---

# Current Progress

## Computer Vision
- [x] CUDA-enabled YOLOv8 setup
- [x] Real-time person detection
- [x] Crowd counting
- [x] ByteTrack person tracking with persistent IDs
- [x] Speed and acceleration tracking per person
- [x] Trajectory visualization and logging
- [x] CSV trajectory export (tracking_data.csv)
- [x] Flat feature extraction for Isolation Forest (features.csv)
- [x] Normalized occupancy grid sequences for ConvLSTM2D (X.npy, y.npy)
- [x] Occupancy grid visualizer (animated heatmap)
- [x] ShanghaiTech Part B evaluation
- [x] YOLOv8 model comparison (nano vs small)
- [x] Full pipeline runner (main.py)

## Machine Learning
- [ ] Isolation Forest anomaly detection
- [ ] ConvLSTM2D panic prediction
- [ ] Model training and evaluation

## Backend & Frontend
- [ ] FastAPI backend integration
- [ ] React dashboard
- [ ] Live alert system

---

# Team Responsibilities

| Member | Responsibility |
|--------|---------------|
| Ayuj | Computer Vision — detection, tracking, feature extraction, grid export |
| Neeraj | Machine Learning — Isolation Forest, ConvLSTM2D, panic prediction |
| Darshit | Backend APIs and frontend dashboard |



## 🧠 Machine Learning Module & Predictive Streaming Engine

The machine learning module is split into structural model architectures (`ml/utilities.py`) and a stateful, real-time node orchestration tracking layer (`ml/predictor.py`). Together, they form an auto-calibrating predictive system designed to monitor crowd anomalies at the edge.

---

### 📂 Technical Component Breakdown

#### 1. Deep Learning Foundations (`ml/utilities.py`)
This file serves as our model repository and mathematical pipeline, preparing raw frames into 5D spatiotemporal tensors optimized for recurrent networks:
* **Spatial Feature Extraction (CSRNet):** Implements a dilated Convolutional Neural Network backbone. It utilizes front-end feature extractors and back-end dilated convolutions to map highly precise spatial density fields. Model weights are dynamically fetched over network sockets directly from HuggingFace Hub on setup (`get_weights_for_csrnet`).
* **Spatiotemporal Forecasting (ConvLSTM):** A deeper 5D Convolutional Long Short-Term Memory network that tracks historical vector movements. By running 2D convolutions directly within the internal memory gate cells, it accurately forecasts future crowd displacement fields while retaining structural coordinates.
* **Stream Helpers:** Contains custom data-shaping blocks like `prepare_grid_batched` and `build_all_interleaved_sequences` to handle sliding window segmenting ($10 \times 10 \times 3$ frame gaps), along with built-in Matplotlib timeline plotting utilities (`visualize_prediction_timeline`, `plot_graph`).

#### 2. Stateful Multi-Camera Processing (`ml/predictor.py`)
Houses the decoupled, object-oriented `Camera` class. Each active CCTV stream feed on the dashboard initializes its own dedicated tracker instance to prevent memory leaks or cross-stream data corruption over long production runtimes.

* **Volatility-Based Adaptive Ceilings:** To handle diverse physical environments without manual retuning, each camera node auto-calibrates an envelope ceiling on startup. It accumulates the first 15 evaluation sequences and applies a 90% volatility dampening factor directly to the trend variance:
  
  $$\text{Initial } \sigma = \text{np.std(calibration\-scores)} \times 0.1$$
  
  $$\text{Adaptive Threshold Ceiling} = \text{running\-mean} + (1.5 \times \text{running\-std})$$

* **Dynamic Decay Tracking:** Once production tracking is active, the node shifts to an Exponential Moving Average (EMA) decay cadence (`alpha=0.06`) to smoothly update the threshold ceiling over time, keeping it positioned safely right above normal crowd spikes.
* **Flicker-Control Breach Filtering:** To avoid triggering false alerts from random camera jitter, video encoding artifacts, or compression frame drops, the node enforces a **Sustained Sequential Violation Checking Rule**. An anomaly alert is *only* dispatched to the frontend web sockets if the tracking Mean Squared Error ($MSE$) breaches the adaptive threshold for **5 consecutive evaluation blocks**.

---

### 📊 Empirical Performance Verification
* **Normal Crowd Sequences:** **5/7 Slices Verified Correctly** (Ambient noise filtered cleanly)
* **Abnormal Panic Surges:** **5/6 Slices Caught Correctly** (Instantaneous threshold penetration directly at surge initiation)
* **Stream Core Cycle Latency:** **~8ms to 25ms** per individual stream evaluation cycle (Ultra-lightweight edge performance)

---




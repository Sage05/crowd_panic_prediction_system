# 🚨 Crowd Panic Prediction System

> **Real-Time Crowd Anomaly Detection and Forecasting using CSRNet,
> ConvLSTM, FastAPI and React**

------------------------------------------------------------------------

# 📖 Overview

The Crowd Panic Prediction System is an end-to-end AI surveillance
platform designed to detect abnormal crowd behaviour before visible
panic develops.

Unlike conventional surveillance systems that only detect events after
they occur, this system forecasts the short-term evolution of crowd
density and compares expected behaviour with observed behaviour.

The pipeline combines deep learning models with a real-time backend and
an interactive dashboard.

------------------------------------------------------------------------

# ✨ Features

-   Multi-camera architecture
-   Real-time video processing
-   MP4 and future RTSP/IP camera support
-   CSRNet density estimation
-   ConvLSTM crowd forecasting
-   Adaptive statistical anomaly detection
-   Camera-specific calibration
-   FastAPI backend
-   Live WebSocket updates
-   React monitoring dashboard
-   Crowd density heatmaps
-   Panic risk scoring
-   Modular architecture

------------------------------------------------------------------------

# 🏗 System Architecture

``` text
               CCTV / RTSP / MP4
                       │
                       ▼
               VideoStream Module
                       │
                       ▼
            Camera.receive_frames()
                       │
                       ▼
        CSRNet Density Estimation Model
                       │
                32×60 Density Grid
                       │
                       ▼
          ConvLSTM Forecasting Network
                       │
                Future Density Maps
                       │
                       ▼
              Mean Squared Error (MSE)
                       │
                       ▼
          Adaptive Statistical Threshold
                       │
                       ▼
              Panic / Normal Decision
                       │
                       ▼
               DashboardState Builder
                       │
                       ▼
           FastAPI WebSocket Backend
                       │
                       ▼
              React Monitoring Dashboard
```

------------------------------------------------------------------------

# 🧠 Machine Learning Pipeline

## Stage 1 -- Frame Acquisition

Frames are collected from each camera independently and buffered into
fixed-length temporal windows.

## Stage 2 -- Density Estimation

CSRNet converts RGB frames into crowd density maps.

Output:

-   32 × 60 density matrix
-   Estimated crowd count
-   Spatial crowd distribution

## Stage 3 -- Forecasting

ConvLSTM receives historical density maps and predicts future crowd
evolution.

Input: - Previous 10 density maps

Output: - Forecasted 10 density maps

## Stage 4 -- Error Computation

Prediction error is computed using Mean Squared Error between predicted
and observed density maps.

## Stage 5 -- Adaptive Thresholding

Each camera maintains:

-   Running Mean
-   Running Standard Deviation
-   Calibration history
-   Consecutive anomaly counter

This allows every camera to learn its own normal behaviour.

------------------------------------------------------------------------

# ⚙ Backend Architecture

## main.py

-   FastAPI entry point
-   WebSocket server
-   REST endpoints
-   Background update loop

## dashboard_state.py

Responsible for translating ML outputs into frontend payloads.

## camera_manager.py

Maintains one Camera object per feed and coordinates processing.

## video_stream.py

Abstracts MP4 today and RTSP/IP cameras in future.

------------------------------------------------------------------------

# 🖥 Frontend

The React dashboard displays:

-   Live camera cards
-   Crowd count
-   Density heatmaps
-   Risk indicator
-   Alerts
-   Panic history
-   Zone status
-   System health

Communication occurs exclusively through a WebSocket connection.

------------------------------------------------------------------------

# 📁 Repository Structure

``` text
panic_prediction_system/
│
├── backend/
│   ├── main.py
│   ├── dashboard_state.py
│   ├── camera/
│   ├── video/
│   └── demo/
│
├── ml/
│   ├── predictor.py
│   ├── utilities.py
│   └── ...
│
├── Frontend/
│
├── cv/
│
├── docs/
│
└── README.md
```

------------------------------------------------------------------------

# 🛠 Technology Stack

## AI / ML

-   CSRNet
-   ConvLSTM
-   TensorFlow
-   PyTorch
-   NumPy

## Backend

-   FastAPI
-   Uvicorn
-   OpenCV
-   asyncio

## Frontend

-   React
-   JavaScript
-   WebSocket API

------------------------------------------------------------------------

# 🚀 Installation

## Clone

``` bash
git clone <repository-url>
cd panic_prediction_system
```

## Backend

``` bash
cd backend
python -m venv venv
pip install -r ../requirements.txt
```

Run:

``` bash
python -m uvicorn backend.main:app --reload
```

## Frontend

``` bash
cd Frontend
npm install
npm run dev
```

------------------------------------------------------------------------

# 📡 API

## REST

  Method   Endpoint     Description
  -------- ------------ ----------------
  GET      /            Health
  GET      /api/hello   Backend status

## WebSocket

    ws://localhost:8000/ws/live

Payload contains:

-   site
-   timestamp
-   cameras
-   zones
-   alerts
-   peopleCount
-   densityGrid
-   panicHistory
-   panicRisk
-   systemStatus

------------------------------------------------------------------------

# 📈 Future Improvements

-   RTSP camera support
-   YOLO person tracking
-   Multi-camera fusion
-   Database logging
-   Docker
-   Kubernetes
-   Alert notifications
-   Mobile dashboard
-   Grafana integration
-   Model retraining pipeline

------------------------------------------------------------------------

# 👥 Team

  Member         Responsibility
  -------------- -------------------------------------------------
  Ayuj Shingte   Computer Vision, Backend Integration, Dashboard
  Neeraj         CSRNet + ConvLSTM ML Pipeline
  Darshit        Backend Services & Integration

------------------------------------------------------------------------

# 📚 References

-   CSRNet: Congested Scene Recognition
-   ConvLSTM: Precipitation Nowcasting
-   FastAPI Documentation
-   React Documentation

------------------------------------------------------------------------

# 📄 License

This project is intended for academic research, experimentation and
educational purposes.

Contributions and improvements are welcome.

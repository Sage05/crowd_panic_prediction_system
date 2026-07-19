# System Interfaces

# Overview

This document describes the interfaces between the major components of the Crowd Panic Prediction System.

---

# System Architecture

```
Camera Feed
      │
      ▼
YOLO Person Detection
      │
      ▼
CSRNet Density Estimation
      │
      ▼
ConvLSTM Forecasting
      │
      ▼
Adaptive Anomaly Detection
      │
      ▼
FastAPI Backend
      │
      ▼
WebSocket API
      │
      ▼
React Dashboard
```

---

# Frontend ↔ Backend

## REST APIs

### GET /api/cameras

Returns all registered cameras.

Response

```json
[
  {
    "camera_id": "Camera_001",
    "zone": "North Stand",
    "source": "demo/videos/demo1.mp4"
  }
]
```

---

### POST /api/cameras

Registers a new camera.

Request

```json
{
  "zone": "Gate A",
  "source": "demo/videos/demo2.mp4"
}
```

---

### DELETE /api/cameras/{camera_id}

Removes a registered camera.

---

# WebSocket Interface

Endpoint

```
ws://localhost:8000/ws/live
```

The backend streams live dashboard updates approximately every processing cycle.

Example payload

```json
{
  "site": "Live Deployment",
  "panicRisk": 67,
  "peopleCount": 29,
  "riskClass": "Moderate",
  "systemStatus": "Elevated",
  "cameras": [],
  "zones": [],
  "alerts": []
}
```

---

# Camera Manager Interface

The CameraManager is responsible for:

- Registering cameras
- Removing cameras
- Persisting camera configuration
- Updating all active camera streams
- Providing processed ML results to DashboardState

Output

```python
[
    {
        "camera_id": "Camera_001",
        "people_count": 15,
        "window_mse": 0.00048,
        "adaptive_threshold": 0.00084,
        "is_anomaly": False,
        "forecasted_grids": [...],
        "current_density_map": [...]
    }
]
```

---

# ML Interface

Input

- Video frames
- Previous density history

Output

- Crowd density map
- Predicted density maps
- People count
- Adaptive threshold
- Panic anomaly flag

---

# Dashboard Interface

The React dashboard consumes the WebSocket payload and displays:

- Camera feeds
- Density heatmaps
- Panic risk
- Crowd count
- Alert history
- Camera lifecycle status
- Zone statistics

---

# Camera Lifecycle

Each registered camera progresses through the following states:

```
GATHERING_CONTEXT
        ↓
CALIBRATING
        ↓
LIVE
```

Predictions are only generated after calibration is complete.

---

# Configuration Files

camera_config.json

Stores persistent camera registration.

requirements.txt

Lists backend dependencies.

README.md

Project documentation.

---

# Future Interfaces

- RTSP/IP camera support
- Cloud storage
- Authentication service
- Database-backed camera registry
- Notification service

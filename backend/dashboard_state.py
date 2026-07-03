from datetime import datetime, timezone
from collections import deque

from backend.camera.camera_manager import CameraManager


class DashboardState:

    def __init__(self):

        self.capacity = 8000
        self.confidence = 94
        self.site = "Live Deployment"

        self.camera_manager = CameraManager()

        self.payload = {}

        self.panic_history = deque([0.0] * 30, maxlen=30)

    def update(self):

        ml_results = self.camera_manager.update()

        cameras = []
        zones = []
        alerts = []

        total_people = 0
        max_risk = 0.0

        for result in ml_results:

            zone = result.get("zone", "Unknown")

            # -----------------------------
            # Calibration / Gathering state
            # -----------------------------
            if "status" in result:

                cameras.append({
                    "id": result["camera_id"],
                    "zone": zone,
                    "status": result["status"],
                    "riskScore": 0.0,
                    "load": 0,
                    "count": 0,
                    "density": 0.0,
                    "isAnomaly": False,
                    "grid": [[0.0] * 60 for _ in range(32)],
                    "forecasted_grids": [], # Added empty fallback for calibration state
                })

                zones.append({
                    "name": zone,
                    "load": 0
                })

                alerts.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "level": "INFO",
                    "zone": zone,
                    "msg": result["status"]
                })

                continue

            # -----------------------------
            # Normal ML prediction
            # -----------------------------
            people = result["people_count"]

            load = min(int((people / 400) * 100), 100)

            density = round(people / 100.0, 1)

            risk_score = min(
                result["window_mse"] /
                max(result["adaptive_threshold"], 1e-6),
                1.0,
            )

            cameras.append({

                "id": result["camera_id"],

                "zone": zone,

                "status": "LIVE",

                "riskScore": risk_score,

                "load": load,

                "count": people,

                "density": density,

                "isAnomaly": result["is_anomaly"],

                "grid": result["current_density_map"],

                "forecasted_grids": result.get("forecasted_grids", []), # Added forecast sequence mapping
            })

            zones.append({
                "name": zone,
                "load": load
            })

            total_people += people

            max_risk = max(max_risk, risk_score)

            if result["is_anomaly"]:

                alerts.append({

                    "time": datetime.now().strftime("%H:%M:%S"),

                    "level": "WARNING",

                    "zone": zone,

                    "msg": "Panic anomaly detected"

                })

        # -----------------------------
        # System status
        # -----------------------------
        panic = int(max_risk * 100)

        self.panic_history.append(max_risk)

        if panic >= 75:

            risk_class = "Critical"
            system_status = "Critical"

        elif panic >= 40:

            risk_class = "Moderate"
            system_status = "Elevated"

        else:

            risk_class = "Low"
            system_status = "Normal"

        if not alerts:

            alerts.append({

                "time": datetime.now().strftime("%H:%M:%S"),

                "level": "OK",

                "zone": "System",

                "msg": "No active anomalies"

                })

        density_grid = (
            cameras[0]["grid"]
            if cameras
            else [[0.0] * 60 for _ in range(32)]
        )

        trend = panic - int(self.panic_history[0] * 100)

        self.payload = {

            "site": self.site,

            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "fps": 30,

            "confidence": self.confidence,

            "systemStatus": system_status,

            "riskClass": risk_class,

            "panicRisk": panic,

            "peopleCount": total_people,

            "capacity": self.capacity,

            "trend": trend,

            "cameras": cameras,

            "zones": zones,

            "alerts": alerts,

            "panicHistory": list(self.panic_history),

            "densityGrid": density_grid,

        }

    def to_dict(self):

        return self.payload
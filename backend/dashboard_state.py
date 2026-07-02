from datetime import datetime, timezone
from collections import deque

from camera_manager import CameraManager


class DashboardState:

    def __init__(self):

        self.capacity = 8000
        self.confidence = 94
        self.site = "Live deployment"
        self.systemStatus = "Elevated"

        self.camera_manager = CameraManager()

        self.payload = {}

        # Persistent history
        self.panic_history = deque([0.5] * 30, maxlen=30)
    def update(self):

        cams = self.camera_manager.update()

        people = sum(c["count"] for c in cams)

        avg_load = sum(c["load"] for c in cams) / len(cams)

        panic = int(avg_load)

        self.panic_history.append(panic / 100)

        # Risk Class
        if panic >= 75:
            risk = "Critical"
            status = "Critical"

        elif panic >= 60:
            risk = "Moderate"
            status = "Elevated"

        else:
            risk = "Low"
            status = "Normal"

        # Trend
        trend = panic - self.panic_history[0]

        self.payload = {
            "site": self.site,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fps": 30,
            "confidence": self.confidence,
            "systemStatus": status,
            "riskClass": risk,
            "panicRisk": panic,
            "peopleCount": people,
            "capacity": self.capacity,
            "trend": trend,
            "cameras": cams,
            "zones": [
                {
                    "name": c["zone"],
                    "load": c["load"]
                }
                for c in cams
            ],
            "alerts": [
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "level": "OK" if panic < 70 else "WARNING",
                    "zone": "System",
                    "msg": "Backend connected",
                }
            ],
            "panicHistory": list(self.panic_history),
            "densityGrid": [
                [0.0 for _ in range(22)]
                for _ in range(6)
            ],
        }

    def to_dict(self):
        return self.payload
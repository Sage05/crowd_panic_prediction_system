from datetime import datetime, timezone
import random

from camera_manager import CameraManager


class DashboardState:

    def __init__(self):

        self.capacity = 8000
        self.confidence = 94
        self.site = "Live deployment"
        self.systemStatus = "Elevated"

        self.camera_manager = CameraManager()

        self.payload = {}

    def update(self):

        cams = self.camera_manager.update()

        panic = random.randint(45, 70)

        self.payload = {
            "site": self.site,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fps": 30,
            "confidence": self.confidence,
            "systemStatus": self.systemStatus,
            "riskClass": "Moderate",
            "panicRisk": panic,
            "peopleCount": sum(c["count"] for c in cams),
            "capacity": self.capacity,
            "trend": 16,
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
                    "level": "OK",
                    "zone": "System",
                    "msg": "Backend connected",
                }
            ],
            "panicHistory": [
                random.uniform(0.3, 0.6)
                for _ in range(30)
            ],
            "densityGrid": [
                [random.random() for _ in range(22)]
                for _ in range(6)
            ],
        }

    def to_dict(self):
        return self.payload
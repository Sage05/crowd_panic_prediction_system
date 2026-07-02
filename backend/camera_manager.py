import random


class CameraManager:

    def __init__(self):

        self.cameras = [
            {
                "id": "Camera_001",
                "zone": "North Stand",
            },
            {
                "id": "Camera_002",
                "zone": "Gate A",
            },
            {
                "id": "Camera_003",
                "zone": "Concourse",
            },
            {
                "id": "Camera_004",
                "zone": "South Ramp",
            },
            {
                "id": "Camera_005",
                "zone": "East Wing",
            },
            {
                "id": "Camera_006",
                "zone": "Plaza",
            },
        ]

    def update(self):

        output = []

        for cam in self.cameras:

            load = random.randint(20, 80)

            output.append({
                "id": cam["id"],
                "zone": cam["zone"],
                "status": "LIVE",
                "riskScore": load / 100,
                "load": load,
                "count": random.randint(80, 400),
                "density": round(random.uniform(0.8, 3.5), 1),
                "isAnomaly": load > 72,
                "grid": [
                    [random.random() for _ in range(12)]
                    for _ in range(7)
                ],
            })

        return output
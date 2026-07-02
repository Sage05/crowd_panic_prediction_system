import random

from .base_camera import BaseCamera


class SimulatedCamera(BaseCamera):

    def __init__(self, camera_id: str, zone: str):

        super().__init__(camera_id, zone)

        self.load = random.randint(35, 60)
        self.count = random.randint(150, 250)
        self.density = round(random.uniform(1.5, 2.5), 1)

        # Persistent density map
        self.grid = [
            [random.uniform(0.2, 0.8) for _ in range(12)]
            for _ in range(7)
        ]

        # Moving hotspot
        self.hotspot_x = random.randint(2, 9)
        self.hotspot_y = random.randint(2, 4)

        self.dx = random.choice([-1, 1])
        self.dy = random.choice([-1, 1])

    def update(self):

        # Crowd load
        self.load += random.randint(-3, 3)
        self.load = max(0, min(100, self.load))

        # People count
        self.count += random.randint(-10, 10)
        self.count = max(0, self.count)

        # Density
        self.density += random.uniform(-0.1, 0.1)
        self.density = round(max(0.0, self.density), 1)

        # Random direction changes
        if random.random() < 0.25:
            self.dx = random.choice([-1, 0, 1])

        if random.random() < 0.25:
            self.dy = random.choice([-1, 0, 1])

        self.hotspot_x += self.dx
        self.hotspot_y += self.dy

        self.hotspot_x = max(1, min(10, self.hotspot_x))
        self.hotspot_y = max(1, min(5, self.hotspot_y))

        # Fade grid
        for r in range(7):
            for c in range(12):
                self.grid[r][c] *= 0.92

        # Hotspot
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):

                rr = self.hotspot_y + dy
                cc = self.hotspot_x + dx

                if 0 <= rr < 7 and 0 <= cc < 12:
                    self.grid[rr][cc] = min(
                        1.0,
                        self.grid[rr][cc] + random.uniform(0.2, 0.4)
                    )

        # Sensor noise
        for r in range(7):
            for c in range(12):
                self.grid[r][c] += random.uniform(-0.01, 0.01)
                self.grid[r][c] = max(0.0, min(1.0, self.grid[r][c]))

    def get_dashboard_data(self):

        return {
            "id": self.camera_id,
            "zone": self.zone,
            "status": "LIVE",
            "riskScore": self.load / 100,
            "load": self.load,
            "count": self.count,
            "density": self.density,
            "isAnomaly": self.load >= 75,
            "grid": self.grid,
        }
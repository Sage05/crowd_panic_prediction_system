from .simulated_camera import SimulatedCamera


class CameraManager:

    def __init__(self):

        self.debug = True

        self.cameras = [

            SimulatedCamera("Camera_001", "North Stand"),
            SimulatedCamera("Camera_002", "Gate A"),
            SimulatedCamera("Camera_003", "Concourse"),
            SimulatedCamera("Camera_004", "South Ramp"),
            SimulatedCamera("Camera_005", "East Wing"),
            SimulatedCamera("Camera_006", "Plaza"),

        ]

    def update(self):

        output = []

        for camera in self.cameras:

            camera.update()

            if self.debug:
                print(
                    f"{camera.camera_id:10} | "
                    f"Load {camera.load:3}% | "
                    f"Count {camera.count:3} | "
                    f"Density {camera.density:.1f}"
                )

            output.append(
                camera.get_dashboard_data()
            )

        if self.debug:
            print("-" * 80)

        return output
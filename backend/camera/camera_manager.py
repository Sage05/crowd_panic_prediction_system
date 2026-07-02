from ml.predictor import Camera
from ml.utilities import density_model, forecasting_model

from backend.video.video_stream import VideoStream


CAMERA_CONFIG = {
    "Camera_001": {
        "zone": "North Stand",
        "source": "demo/videos/demo1.mp4",
    },
    # Later:
    # "Camera_002": {
    #     "zone": "Gate A",
    #     "source": "demo/videos/demo2.mp4",
    # },
}


class CameraManager:

    def __init__(self):

        self.cameras = {}

        for camera_id, config in CAMERA_CONFIG.items():

            self.cameras[camera_id] = {

                "camera": Camera(camera_id),

                "stream": VideoStream(config["source"]),

                "zone": config["zone"],

            }

    def update(self):

        payload = []

        for camera_id, obj in self.cameras.items():

            frames = obj["stream"].get_next_chunk()

            if len(frames) == 0:
                continue

            camera = obj["camera"]

            camera.receive_frames(frames)

            result = camera.process_and_evaluate_stream(
                density_model,
                forecasting_model,
            )

            result["zone"] = obj["zone"]

            payload.append(result)

        return payload

    def shutdown(self):

        for obj in self.cameras.values():

            obj["stream"].release()
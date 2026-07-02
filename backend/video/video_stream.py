import cv2
from pathlib import Path


class VideoStream:

    def __init__(self, source, chunk_size=60):

        self.chunk_size = chunk_size

        # Resolve path relative to backend/
        base_dir = Path(__file__).resolve().parents[1]
        video_path = base_dir / source

        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {video_path}"
            )

    def get_next_chunk(self):

        frames = []

        while len(frames) < self.chunk_size:

            success, frame = self.cap.read()

            if not success:
                break

            frames.append(frame)

        return frames

    def has_frames(self):

        return self.cap.isOpened()

    def release(self):

        if self.cap is not None:
            self.cap.release()
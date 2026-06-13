from ultralytics import YOLO
import cv2
from collections import defaultdict
import math

from cv.config import (
    MODEL_PATH,
    DEVICE,
    CONFIDENCE,
    VIDEO_PATH,
    TRACK_HISTORY,
    MIN_MOVEMENT
)


model = YOLO(MODEL_PATH)

tracks = defaultdict(list)


def visualize_trajectory(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONFIDENCE,
            device=DEVICE
        )

        boxes = results[0].boxes

        if boxes.id is not None:

            ids = boxes.id.cpu().numpy()
            coords = boxes.xywh.cpu().numpy()

            for pid, box in zip(ids, coords):

                pid = int(pid)

                x, y, w, h = box

                center = (int(x), int(y))

                if (
                    len(tracks[pid]) == 0
                    or math.dist(center, tracks[pid][-1]) > MIN_MOVEMENT
                ):
                    tracks[pid].append(center)

                if len(tracks[pid]) > TRACK_HISTORY:
                    tracks[pid].pop(0)

                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID {pid}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                for i in range(1, len(tracks[pid])):
                    cv2.line(
                        frame,
                        tracks[pid][i-1],
                        tracks[pid][i],
                        (0, 0, 255),
                        2
                    )

        cv2.imshow("Trajectory Tracking", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_trajectory(VIDEO_PATH)
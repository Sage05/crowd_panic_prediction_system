from ultralytics import YOLO
import cv2
import math
import csv
import os
from collections import defaultdict

from cv.config import (
    MODEL_PATH,
    DEVICE,
    CONFIDENCE,
    VIDEO_PATH
)

model = YOLO(MODEL_PATH)

# -----------------------------
# Tracking Storage
# -----------------------------

track_history = defaultdict(list)

previous_centers = {}

previous_speeds = {}

tracking_data = {}

csv_file = "data/tracking_data.csv"

fps = 0


def initialize_csv():

    os.makedirs("data", exist_ok=True)

    with open(csv_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "frame",
            "person_id",
            "center_x",
            "center_y",
            "speed",
            "acceleration"
        ])


def calculate_speed(previous_center, current_center):

    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]

    return math.sqrt(dx * dx + dy * dy)


def calculate_acceleration(previous_speed, current_speed):

    return current_speed - previous_speed


def track_crowd(video_path):

    initialize_csv()

    cap = cv2.VideoCapture(video_path)

    frame_number = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONFIDENCE,
            device=DEVICE
        )

        output = results[0].plot()

        count = len(results[0].boxes)

        if results[0].boxes.id is not None:

            ids = results[0].boxes.id.int().cpu().tolist()

            boxes = results[0].boxes.xyxy.cpu().tolist()

            for person_id, box in zip(ids, boxes):

                x1, y1, x2, y2 = box

                center = (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2)
                )

                speed = 0

                acceleration = 0

                if person_id in previous_centers:

                    speed = calculate_speed(
                        previous_centers[person_id],
                        center
                    )

                if person_id in previous_speeds:

                    acceleration = calculate_acceleration(
                        previous_speeds[person_id],
                        speed
                    )

                previous_centers[person_id] = center
                previous_speeds[person_id] = speed

                track_history[person_id].append(center)

                if len(track_history[person_id]) > 100:

                    track_history[person_id].pop(0)

                history = track_history[person_id]

                for i in range(1, len(history)):

                    cv2.line(
                        output,
                        history[i - 1],
                        history[i],
                        (255, 0, 255),
                        2
                    )

                tracking_data[person_id] = {

                    "center": center,

                    "speed": speed,

                    "acceleration": acceleration

                }

                with open(csv_file, "a", newline="") as file:

                    writer = csv.writer(file)

                    writer.writerow([

                        frame_number,

                        person_id,

                        center[0],

                        center[1],

                        round(speed, 2),

                        round(acceleration, 2)

                    ])

        cv2.putText(

            output,

            f"Tracked People : {count}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2

        )

        cv2.putText(

            output,

            f"FPS : {fps:.2f}",

            (20, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 255),

            2

        )

        cv2.imshow("Crowd Tracking", output)

        if cv2.waitKey(1) == 27:
            break

    cap.release()

    cv2.destroyAllWindows()

    return tracking_data


if __name__ == "__main__":

    track_crowd(VIDEO_PATH)

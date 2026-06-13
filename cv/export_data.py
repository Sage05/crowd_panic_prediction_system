from ultralytics import YOLO
import cv2
import csv

from cv.config import MODEL_PATH, DEVICE, CONFIDENCE, VIDEO_PATH


model = YOLO(MODEL_PATH)


def export_data(video_path, output_file):

    cap = cv2.VideoCapture(video_path)

    with open(output_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "frame",
            "person_id",
            "x_center",
            "y_center",
            "width",
            "height"
        ])

        frame_no = 0

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            frame_no += 1

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

                    x, y, w, h = box

                    writer.writerow([
                        frame_no,
                        int(pid),
                        x,
                        y,
                        w,
                        h
                    ])

    cap.release()

    print(
        f"Export complete. Saved to {output_file}"
    )


if __name__ == "__main__":
    export_data(
        VIDEO_PATH,
        "data/tracking_data.csv"
    )
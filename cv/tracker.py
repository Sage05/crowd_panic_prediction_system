from ultralytics import YOLO
import cv2

from cv.config import MODEL_PATH, DEVICE, CONFIDENCE, VIDEO_PATH


model = YOLO(MODEL_PATH)


def track_crowd(video_path):
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

        count = len(results[0].boxes)

        output = results[0].plot()

        cv2.putText(
            output,
            f"Tracked People: {count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        cv2.imshow("YOLO Tracking", output)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track_crowd(VIDEO_PATH)
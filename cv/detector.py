from ultralytics import YOLO
import cv2

from cv.config import (
    MODEL_PATH,
    DEVICE,
    CONFIDENCE,
    VIDEO_PATH
)

model = YOLO(MODEL_PATH)


def detect_crowd(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    while True:
        success, frame = cap.read()

        if not success:
            print("Video ended.")
            break

        # Detect only people (COCO class 0)
        results = model(
        frame,
        device=DEVICE,
        classes=[0],
        conf=CONFIDENCE
        )

        # Number of people detected
        people_count = len(results[0].boxes)

        # Draw YOLO detections
        annotated_frame = results[0].plot()

        # Display people count
        cv2.putText(
            annotated_frame,
            f"People Count: {people_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        # Show video
        cv2.imshow(
            "Crowd Panic Detection - YOLOv8",
            annotated_frame
        )

        # Press ESC to exit
        if cv2.waitKey(1) == 27:
            print("Stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    detect_crowd(VIDEO_PATH)
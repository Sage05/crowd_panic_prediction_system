from ultralytics import YOLO
import cv2
import os

from cv.config import (
    MODEL_PATH,
    DEVICE,
    CONFIDENCE
)


# Load YOLO model
model = YOLO(MODEL_PATH)


def evaluate_dataset(input_folder, output_folder):
    """
    Runs YOLO person detection on a folder of images,
    saves annotated images, and prints crowd counts.
    """

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Supported image formats
    extensions = (".jpg", ".jpeg", ".png")

    # Get all images
    images = [
        img for img in os.listdir(input_folder)
        if img.lower().endswith(extensions)
    ]

    print(f"Found {len(images)} images")
    print("Starting evaluation...\n")


    for image_name in images:

        image_path = os.path.join(
            input_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read {image_name}")
            continue


        # Run YOLO detection
        results = model(
            image,
            device=DEVICE,
            classes=[0],
            conf=CONFIDENCE
        )


        # Count people
        people_count = len(results[0].boxes)


        # Draw detections
        output_image = results[0].plot()


        # Add count text
        cv2.putText(
            output_image,
            f"People Detected: {people_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )


        # Save result image
        save_path = os.path.join(
            output_folder,
            image_name
        )

        cv2.imwrite(
            save_path,
            output_image
        )


        print(
            f"{image_name} --> {people_count} people detected"
        )


    print("\n================================")
    print("Evaluation Complete")
    print(f"Results saved to: {output_folder}")
    print("================================")


if __name__ == "__main__":

    INPUT_FOLDER = (
        "data/datasets/"
        "ShanghaiTech_Crowd_Counting_Dataset/"
        "part_B_final/test_data/images"
    )


    OUTPUT_FOLDER = (
        "data/results/yolov8n_partB"
    )


    evaluate_dataset(
        INPUT_FOLDER,
        OUTPUT_FOLDER
    )
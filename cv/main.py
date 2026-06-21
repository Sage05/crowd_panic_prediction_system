from cv.config import VIDEO_PATH
from cv.tracker import track_crowd
from cv.export_data import export_features
from cv.grid_exporter import export_grid
from cv.grid_visualizer import visualize_grid


def main():

    print("=" * 50)
    print("  Crowd Panic Prediction — CV Pipeline")
    print("=" * 50 + "\n")

    # Step 1 — Track people, write tracking_data.csv
    print("[1/3] Running YOLO tracker...")
    track_crowd(VIDEO_PATH)
    print("Done.\n")

    # Step 2 — Extract flat features, write features.csv (Isolation Forest input)
    print("[2/3] Exporting flat features for Isolation Forest...")
    export_features()
    print("Done.\n")

    # Step 3 — Build occupancy grid sequences, write X.npy + y.npy (ConvLSTM2D input)
    print("[3/3] Building occupancy grid sequences for ConvLSTM2D...")
    export_grid()
    print("Done.\n")

    print("=" * 50)
    print("Output files:")
    print("  data/tracking_data.csv  — raw per-frame detections")
    print("  data/features.csv       — flat features  (Isolation Forest)")
    print("  data/X.npy              — grid sequences (ConvLSTM2D input)")
    print("  data/y.npy              — labels         (ConvLSTM2D supervision)")
    print("=" * 50)
    print("\nTo verify grids visually, run:")
    print("  python -m cv.grid_visualizer")


if __name__ == "__main__":

    main()

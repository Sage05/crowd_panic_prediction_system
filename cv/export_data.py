import csv
import os

from cv.feature_extractor import extract_features

output_file = "data/features.csv"


def export_features(tracking_csv="data/tracking_data.csv", out_file=output_file):
    """
    Reads tracking_data.csv → extracts features → writes features.csv.
    This is the handoff file for the ML teammate (Isolation Forest input).
    """

    features = extract_features(tracking_csv)

    if not features:
        print("No features to export. Run tracker.py first.")
        return

    os.makedirs("data", exist_ok=True)

    with open(out_file, "w", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=features[0].keys())

        writer.writeheader()

        writer.writerows(features)

    print(f"Exported {len(features)} frames → {out_file}")


if __name__ == "__main__":

    export_features()

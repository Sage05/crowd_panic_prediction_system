import csv
import os
from collections import defaultdict

csv_file = "data/tracking_data.csv"

# Speed threshold — below this a person is considered stationary (pixels/frame)
STATIONARY_THRESHOLD = 2.0


def extract_features(tracking_csv=csv_file):
    """
    Reads tracking_data.csv and computes per-frame crowd features.

    Returns a list of dicts, one per frame:
        frame
        people_count
        avg_speed
        avg_speed_squared
        crowd_density         (people per 1000x1000 pixel area — placeholder)
        avg_acceleration
        moving_count
        stationary_count
    """

    if not os.path.exists(tracking_csv):
        print(f"CSV not found: {tracking_csv}")
        return []

    # Group rows by frame
    frames = defaultdict(list)

    with open(tracking_csv, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame = int(row["frame"])
            frames[frame].append({
                "person_id":   int(row["person_id"]),
                "center_x":    float(row["center_x"]),
                "center_y":    float(row["center_y"]),
                "speed":       float(row["speed"]),
                "acceleration": float(row["acceleration"])
            })

    feature_rows = []

    for frame_number in sorted(frames.keys()):

        people = frames[frame_number]

        people_count = len(people)

        speeds       = [p["speed"] for p in people]
        accelerations = [p["acceleration"] for p in people]

        avg_speed         = sum(speeds) / people_count if people_count else 0
        avg_speed_squared = sum(s ** 2 for s in speeds) / people_count if people_count else 0
        avg_acceleration  = sum(accelerations) / people_count if people_count else 0

        moving_count     = sum(1 for s in speeds if s > STATIONARY_THRESHOLD)
        stationary_count = people_count - moving_count

        # Crowd density: people per 1000x1000 block
        # (a rough spatial proxy — can be refined with actual frame dimensions later)
        crowd_density = people_count / (1000 * 1000) * 1e6  # scales to readable number

        feature_rows.append({
            "frame":            frame_number,
            "people_count":     people_count,
            "avg_speed":        round(avg_speed, 4),
            "avg_speed_squared": round(avg_speed_squared, 4),
            "crowd_density":    round(crowd_density, 4),
            "avg_acceleration": round(avg_acceleration, 4),
            "moving_count":     moving_count,
            "stationary_count": stationary_count
        })

    return feature_rows


if __name__ == "__main__":

    features = extract_features()

    if features:
        # Print first 5 frames as a sanity check
        for row in features[:5]:
            print(row)
        print(f"\nTotal frames extracted: {len(features)}")

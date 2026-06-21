import csv
import os
import numpy as np
from collections import defaultdict

# -----------------------------
# Grid Configuration
# -----------------------------

FRAME_WIDTH  = 640
FRAME_HEIGHT = 640

GRID_ROWS = 10
GRID_COLS = 10

CELL_W = FRAME_WIDTH  // GRID_COLS   # 64 pixels per cell
CELL_H = FRAME_HEIGHT // GRID_ROWS   # 64 pixels per cell

TIMESTEPS  = 10
FRAME_STEP = 3    # sample every 3rd frame

tracking_csv  = "data/tracking_data.csv"
output_X_file = "data/X.npy"
output_y_file = "data/y.npy"

# Set to True for sliding window (overlapping), False for non-overlapping chunks
OVERLAPPING = True


# -----------------------------
# Step 1 — Build occupancy grid per frame
# -----------------------------

def build_occupancy_grid(people):
    """
    Returns a 10x10 grid where each cell = number of people in it (raw count).
    Normalization happens later across the full dataset.
    """

    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

    for cx, cy in people:

        col = min(int(cx // CELL_W), GRID_COLS - 1)
        row = min(int(cy // CELL_H), GRID_ROWS - 1)

        grid[row][col] += 1.0    # count, not binary

    return grid


# -----------------------------
# Step 2 — Load tracking CSV
# -----------------------------

def load_frames(csv_path):
    """
    Returns { frame_number: [(cx, cy), ...] }
    """

    frames = defaultdict(list)

    with open(csv_path, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            frame = int(row["frame"])
            cx    = float(row["center_x"])
            cy    = float(row["center_y"])

            frames[frame].append((cx, cy))

    return frames


# -----------------------------
# Step 3 — Normalize grids
# -----------------------------

def normalize_grids(grids):
    """
    Min-max normalizes the full list of grids to [0, 1].
    Uses global max so all grids are on the same scale.
    """

    stacked = np.stack(grids, axis=0)    # (total_frames, 10, 10)

    max_val = stacked.max()

    if max_val == 0:
        return grids    # avoid divide by zero if video is empty

    normalized = [g / max_val for g in grids]

    return normalized


# -----------------------------
# Step 4 — Build sequences
# -----------------------------

def build_sequences(grids, overlapping=OVERLAPPING):
    """
    Groups grids into sequences of length TIMESTEPS.

    overlapping=True  → sliding window (step=1), more data, sequences share frames
    overlapping=False → non-overlapping chunks (step=TIMESTEPS), less data, fully independent

    X shape: (N, TIMESTEPS, GRID_ROWS, GRID_COLS, 1)
    y shape: (N,) — total people count in the last frame of each sequence (proxy label)
    """

    step = 1 if overlapping else TIMESTEPS

    X = []
    y = []

    for i in range(0, len(grids) - TIMESTEPS, step):

        window = grids[i : i + TIMESTEPS]         # list of 10 grids

        sequence = np.stack(window, axis=0)        # (10, 10, 10)
        sequence = sequence[..., np.newaxis]       # (10, 10, 10, 1)

        X.append(sequence)

        # Label = total occupancy in the last frame of the sequence
        label = float(grids[i + TIMESTEPS - 1].sum())
        y.append(label)

    if not X:
        print(f"Not enough frames. Need at least {TIMESTEPS * FRAME_STEP} frames.")
        return None, None

    return np.stack(X, axis=0), np.array(y, dtype=np.float32)


# -----------------------------
# Step 5 — Export
# -----------------------------

def export_grid(
    csv_path=tracking_csv,
    out_X=output_X_file,
    out_y=output_y_file,
    overlapping=OVERLAPPING
):

    if not os.path.exists(csv_path):
        print(f"tracking_data.csv not found at {csv_path}. Run tracker.py first.")
        return

    print("Loading tracking data...")
    frames = load_frames(csv_path)

    all_frame_numbers = sorted(frames.keys())

    # Sample every FRAME_STEP-th frame
    sampled = [f for i, f in enumerate(all_frame_numbers) if i % FRAME_STEP == 0]

    print(f"Sampled {len(sampled)} frames (every {FRAME_STEP}rd frame)")

    # Build raw count grids
    grids = [build_occupancy_grid(frames[f]) for f in sampled]

    # Normalize across all grids
    print("Normalizing grids...")
    grids = normalize_grids(grids)

    # Build sequences
    mode = "overlapping" if overlapping else "non-overlapping"
    print(f"Building {mode} sequences...")
    X, y = build_sequences(grids, overlapping=overlapping)

    if X is None:
        return

    os.makedirs("data", exist_ok=True)

    np.save(out_X, X)
    np.save(out_y, y)

    print(f"\nExported successfully ({mode} mode).")
    print(f"  X shape : {X.shape}   → {out_X}")
    print(f"  y shape : {y.shape}   → {out_y}")
    print(f"\n  → Feed X into ConvLSTM2D, y as supervision labels.")


if __name__ == "__main__":

    export_grid()

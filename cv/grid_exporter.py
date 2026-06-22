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

# True = sliding window (more samples), False = fully disjoint X/Y blocks
OVERLAPPING = True


# -----------------------------
# Step 1 — Build density grid per frame (Neeraj's overlap-ratio formula)
# -----------------------------

def build_density_grid(boxes):
    """
    Each cell value = sum of (overlap area between YOLO box and cell) / (YOLO box area)
    for all boxes that overlap with that cell.

    This avoids a nearly-empty grid when people span multiple cells.

    boxes: list of (x1, y1, x2, y2) in pixel coordinates
    Returns: np.array of shape (GRID_ROWS, GRID_COLS)
    """

    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

    for (x1, y1, x2, y2) in boxes:

        box_area = (x2 - x1) * (y2 - y1)

        if box_area <= 0:
            continue

        # Which cells does this box touch?
        col_start = max(0, int(x1 // CELL_W))
        col_end   = min(GRID_COLS - 1, int(x2 // CELL_W))
        row_start = max(0, int(y1 // CELL_H))
        row_end   = min(GRID_ROWS - 1, int(y2 // CELL_H))

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):

                # Cell boundaries
                cell_x1 = col * CELL_W
                cell_y1 = row * CELL_H
                cell_x2 = cell_x1 + CELL_W
                cell_y2 = cell_y1 + CELL_H

                # Overlap rectangle
                overlap_x1 = max(x1, cell_x1)
                overlap_y1 = max(y1, cell_y1)
                overlap_x2 = min(x2, cell_x2)
                overlap_y2 = min(y2, cell_y2)

                overlap_w = max(0, overlap_x2 - overlap_x1)
                overlap_h = max(0, overlap_y2 - overlap_y1)
                overlap_area = overlap_w * overlap_h

                # Add ratio of box that falls in this cell
                grid[row][col] += overlap_area / box_area

    return grid


# -----------------------------
# Step 2 — Load tracking CSV
# -----------------------------

def load_frames(csv_path):
    """
    Returns { frame_number: [(x1, y1, x2, y2), ...] }
    Reads full bounding box coordinates for overlap-ratio calculation.
    """

    frames = defaultdict(list)

    with open(csv_path, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            frame = int(row["frame"])

            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])

            frames[frame].append((x1, y1, x2, y2))

    return frames


# -----------------------------
# Step 3 — Normalize grids
# -----------------------------

def normalize_grids(grids):
    """
    Min-max normalizes across all grids to [0, 1] using global max.
    """

    stacked = np.stack(grids, axis=0)
    max_val = stacked.max()

    if max_val == 0:
        return grids

    return [g / max_val for g in grids]


# -----------------------------
# Step 4 — Build X → Y sequences (Neeraj's spec)
# -----------------------------

def build_sequences(grids, overlapping=OVERLAPPING):
    """
    X = grids[i : i + TIMESTEPS]              → input (past second)
    Y = grids[i + TIMESTEPS : i + 2*TIMESTEPS] → target (next second to predict)

    overlapping=True  → step=1, sliding window, lots of samples
    overlapping=False → step=2*TIMESTEPS, fully disjoint X/Y blocks

    X shape: (N, TIMESTEPS, GRID_ROWS, GRID_COLS, 1)
    Y shape: (N, TIMESTEPS, GRID_ROWS, GRID_COLS, 1)
    """

    step = 1 if overlapping else 2 * TIMESTEPS

    X = []
    Y = []

    for i in range(0, len(grids) - 2 * TIMESTEPS + 1, step):

        x_window = grids[i : i + TIMESTEPS]
        y_window = grids[i + TIMESTEPS : i + 2 * TIMESTEPS]

        x_seq = np.stack(x_window, axis=0)[..., np.newaxis]   # (10, 10, 10, 1)
        y_seq = np.stack(y_window, axis=0)[..., np.newaxis]   # (10, 10, 10, 1)

        X.append(x_seq)
        Y.append(y_seq)

    if not X:
        min_frames = 2 * TIMESTEPS * FRAME_STEP
        print(f"Not enough frames. Need at least {min_frames} sampled frames.")
        return None, None

    return np.stack(X, axis=0), np.stack(Y, axis=0)


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

    # Build density grids using overlap-ratio formula
    print("Building density grids (overlap-ratio)...")
    grids = [build_density_grid(frames[f]) for f in sampled]

    # Normalize across all grids
    print("Normalizing grids...")
    grids = normalize_grids(grids)

    # Build X → Y sequences
    mode = "overlapping" if overlapping else "non-overlapping"
    print(f"Building {mode} sequences (X predicts Y)...")
    X, Y = build_sequences(grids, overlapping=overlapping)

    if X is None:
        return

    os.makedirs("data", exist_ok=True)

    np.save(out_X, X)
    np.save(out_y, Y)

    print(f"\nExported successfully ({mode} mode).")
    print(f"  X shape : {X.shape}   → {out_X}")
    print(f"  Y shape : {Y.shape}   → {out_y}")
    print(f"\n  X = input sequences (past {TIMESTEPS} sampled frames)")
    print(f"  Y = target sequences (next {TIMESTEPS} sampled frames to predict)")


if __name__ == "__main__":

    export_grid()

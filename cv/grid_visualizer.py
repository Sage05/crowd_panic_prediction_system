import numpy as np
import cv2
import os

X_file = "data/X.npy"

GRID_ROWS  = 10
GRID_COLS  = 10
CELL_SIZE  = 60     # pixels per cell in the visualizer window


def visualize_grid(X_path=X_file, sequence_index=0):
    """
    Opens a window and plays back the 10-timestep occupancy grid
    for a given sequence index as an animated heatmap.

    Controls:
        ESC  → quit
        N    → next sequence
        P    → previous sequence
    """

    if not os.path.exists(X_path):
        print(f"X.npy not found at {X_path}. Run grid_exporter.py first.")
        return

    X = np.load(X_path)    # (N, 10, 10, 10, 1)

    total_sequences = X.shape[0]
    idx = sequence_index

    print(f"Loaded X.npy — {total_sequences} sequences")
    print("Controls: ESC = quit | N = next sequence | P = previous sequence")

    canvas_w = GRID_COLS * CELL_SIZE
    canvas_h = GRID_ROWS * CELL_SIZE

    while True:

        sequence = X[idx]    # (10, 10, 10, 1)

        for t in range(sequence.shape[0]):

            grid = sequence[t, :, :, 0]    # (10, 10)

            # Scale to 0-255 for display
            display = (grid * 255).astype(np.uint8)

            # Resize to canvas size
            heatmap = cv2.resize(
                display,
                (canvas_w, canvas_h),
                interpolation=cv2.INTER_NEAREST
            )

            # Apply colormap
            colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

            # Draw grid lines
            for r in range(GRID_ROWS + 1):
                y = r * CELL_SIZE
                cv2.line(colored, (0, y), (canvas_w, y), (50, 50, 50), 1)

            for c in range(GRID_COLS + 1):
                x = c * CELL_SIZE
                cv2.line(colored, (x, 0), (x, canvas_h), (50, 50, 50), 1)

            # Overlay cell values
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    val = grid[r][c]
                    if val > 0:
                        cx = c * CELL_SIZE + CELL_SIZE // 2 - 10
                        cy = r * CELL_SIZE + CELL_SIZE // 2 + 5
                        cv2.putText(
                            colored,
                            f"{val:.2f}",
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1
                        )

            # Header
            cv2.putText(
                colored,
                f"Seq {idx}/{total_sequences - 1}  |  Timestep {t + 1}/10",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1
            )

            cv2.imshow("Occupancy Grid Visualizer", colored)

            key = cv2.waitKey(200)    # 200ms per timestep

            if key == 27:             # ESC
                cv2.destroyAllWindows()
                return

            if key == ord("n"):
                idx = min(idx + 1, total_sequences - 1)
                break

            if key == ord("p"):
                idx = max(idx - 1, 0)
                break


if __name__ == "__main__":

    visualize_grid()

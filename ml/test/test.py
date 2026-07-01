import numpy as np
import matplotlib.pyplot as plt

X = np.load("data/X.npy")

print("Input Shape (X):", X.shape)  # Expected: (N, 10, 10, 10, 1)

# Define the number of sequences you want to view
num_sequences = 11  # This covers idx 0, 1, 2, ... up to 10

# Create a grid of plots: num_sequences rows, 10 columns (one for each timestep)
fig, axes = plt.subplots(num_sequences, 10, figsize=(20, 2 * num_sequences))
fig.suptitle("Crowd Flow Heatmaps: Sequences 0 through 10", fontsize=22, y=1.02)

for idx in range(num_sequences):
    for t in range(10):
        # Extract the 10x10 matrix for sequence 'idx' at timestep 't'
        grid_frame = X[idx, t, :, :, 0]
        
        # Select the correct subplot in the grid matrix
        ax = axes[idx, t]
        
        # Plot the matrix
        im = ax.imshow(grid_frame, cmap='hot', vmin=0, vmax=1)
        
        # Formatting: Label columns only on the first row to keep things clean
        if idx == 0:
            ax.set_title(f"T={t+1}", fontsize=12)
            
        # Label rows with the Sequence Index on the very first column
        if t == 0:
            ax.set_ylabel(f"Seq {idx}", fontsize=12, rotation=0, labelpad=30, ha='center')
            
        ax.set_xticks([])
        ax.set_yticks([])

# Add a single shared colorbar on the right side of the entire layout
fig.subplots_adjust(right=0.92, hspace=0.3)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()
import torch
import torch.nn as nn
import tensorflow as tf
from keras import layers,models
from pathlib import Path    
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
# 1. Reuse the working architecture setup

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]

        self.frontend = self._make_layers(self.frontend_feat, in_channels=3, dilation=False)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _make_layers(self, cfg, in_channels=3, dilation=False):
        d_rate = 2 if dilation else 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
def get_weights_for_csrnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_alt = CSRNet().to(device)
    weights_url = "https://huggingface.co/muasifk/CSRNet/resolve/main/CSRNet.pth"

    state_dict = torch.hub.load_state_dict_from_url(
        weights_url,
        map_location=device,
        progress=True
    )
    model_alt.load_state_dict(state_dict)
    model_alt.eval()

    return model_alt
def build_deeper_convlstm(timesteps=10, rows=32, cols=60, channels=1):
    model = models.Sequential()

    # Layer 1 (Increased capacity to 64 filters)
    model.add(layers.ConvLSTM2D(
        filters=64, kernel_size=(3, 3), padding='same',
        return_sequences=True, activation='relu',
        input_shape=(timesteps, rows, cols, channels)
    ))
    model.add(layers.BatchNormalization())

    # Layer 2
    model.add(layers.ConvLSTM2D(
        filters=64, kernel_size=(3, 3), padding='same',
        return_sequences=True, activation='relu'
    ))
    model.add(layers.BatchNormalization())

    # Final reconstruction layer
    model.add(layers.Conv3D(filters=channels, kernel_size=(3, 3, 3), padding='same', activation='linear'))
    return model
def get_actual_convlstm():
    convlstm4 = build_deeper_convlstm()

    weights_path = Path(__file__).parent / "sample_model_weights" / "convlstm_crowd_weights_deep_net_4.weights.h5"

    convlstm4.load_weights(str(weights_path))

    return convlstm4


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def prepare_grid_batched(video_path, model, batch_size=4, downscale_factor=1.0):
    X = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return X

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Processing video frames in batches of {batch_size} on device: {device}...")

    # SAFETY CHECK: Set model to device and strict evaluation mode
    model.to(device)
    model.eval()

    # Calculate exact target size once to avoid double-resizing inside the loop
    target_width = int(960 / downscale_factor)
    target_height = int(536 / downscale_factor)
    # Ensure dimensions are still clean multiples of 16 after downscaling
    target_width = (target_width // 16) * 16
    target_height = (target_height // 16) * 16

    frame_batch = []
    num = 0

    while True:
        ret, frame1 = cap.read()
        if not ret:
            break

        num += 1

        # Combined downscale + standard size step
        frame = cv2.resize(frame1, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Convert color and preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(frame_rgb)
        frame_batch.append(input_tensor)

        # Run batch inference
        if len(frame_batch) == batch_size:
            stacked_batch = torch.stack(frame_batch).to(device)

            with torch.no_grad():
                batch_density_maps = model(stacked_batch)

            batch_np = batch_density_maps.detach().cpu().numpy()[:, 0, :, :]

            for i in range(batch_np.shape[0]):
                X.append(batch_np[i].astype('float16'))

            frame_batch = []
            if num % (batch_size * 5) == 0:
                print(f"Processed up to frame {num}...")

    # Handle remaining tail frames
    if len(frame_batch) > 0:
        stacked_batch = torch.stack(frame_batch).to(device)
        with torch.no_grad():
            batch_density_maps = model(stacked_batch)
        batch_np = batch_density_maps.detach().cpu().numpy()[:, 0, :, :]
        for i in range(batch_np.shape[0]):
            X.append(batch_np[i].astype('float16'))

    cap.release()
    print(f"Finished processing! Total frames in density grid: {len(X)}")
    return X
# import numpy as np

def build_convlstm_sequences(grids, timesteps=10, future_steps=10, stride=10):
    """
    Transforms a continuous sequence of density matrices into lookback (X)
    and target lookahead (Y) temporal sequences optimized for ConvLSTM networks.

    Parameters:
    -----------
    grids : list or np.ndarray
        Array of shape (Total_Frames, 10, 10) containing density matrices.
    timesteps : int
        The historical window size (number of input frames to look back).
    future_steps : int
        The forecasting window size (number of future frames to predict).
    stride : int
        The step size for sliding the lookback sequence forward along the timeline.
        - stride = 1: Maximum overlap (sliding window moves forward frame-by-frame)
        - stride = timesteps: Perfect adjacent continuity (no input overlap)

    Returns:
    --------
    X : np.ndarray
        Input tensor of shape (N, timesteps, 10, 10, 1)
    Y : np.ndarray
        Target tensor of shape (N, future_steps, 10, 10, 1)
    """
    # Ensure input is a clean numpy array
    grids = np.array(grids)

    X = []
    Y = []

    # Calculate total window length required for a single sample (Input + Target)
    total_window_len = timesteps + future_steps

    # Slide the starting index forward by the specified stride step size
    for i in range(0, len(grids) - total_window_len + 1, stride):
        # Extract input lookback block (X)
        x_window = grids[i : i + timesteps]

        # Extract future forecasting block (Y) right after X
        y_window = grids[i + timesteps : i + total_window_len]

        X.append(x_window)
        Y.append(y_window)

    if not X:
        print(f" Error: Dataset too short ({len(grids)} frames) "
              f"to form window blocks of length {total_window_len}.")
        return None, None

    # Convert lists to structural numpy arrays
    X_array = np.array(X)  # Shape: (N, timesteps, 10, 10)
    Y_array = np.array(Y)  # Shape: (N, future_steps, 10, 10)

    # Crucial ConvLSTM alignment: Add the 5th dimensional "Channel" block axis
    X_array = X_array[..., np.newaxis]  # Shape: (N, timesteps, 10, 10, 1)
    Y_array = Y_array[..., np.newaxis]  # Shape: (N, future_steps, 10, 10, 1)

    print(f" Preprocessing Complete!")
    print(f" -> Input (X) Shape: {X_array.shape}")
    print(f" -> Target (Y) Shape: {Y_array.shape}")

    return X_array, Y_array
def visualize_from_videogrids(video_grids,model,idx):
  x_test, y_test = build_all_interleaved_sequences(video_grids, timesteps=10, future_steps=10, stride=3)

  if x_test is None:
    print(" Video is too short to generate testing sequence blocks.")
    return None, None

  # 3. Use slicing [idx:idx+1] instead of [idx] to preserve the 5D shape requirement
  sample_input = x_test[idx : idx + 1]   # Shape: (1, 10, 32, 60, 1)
  sample_target = y_test[idx : idx + 1]  # Shape: (1, 10, 32, 60, 1)

  # 4. Generate the future prediction array
  print(f" Generating forecast for sequence index sample {idx}...")
  prediction = model.predict(sample_input, batch_size=1)

  print(f" -> Prediction output shape: {prediction.shape}") # Shape: (1, 10, 32, 60, 1)
  print(f"loss={np.sum((prediction-sample_target)**2)}")
  print(f"pixel-level-mse:{np.sum((prediction-sample_target)**2)/sample_target.size}")
  visualize_prediction_timeline(prediction,sample_target)

def visualize_prediction_timeline(prediction, sample_target):
    """
    Accepts the 5D prediction and target tensors from your test function
    and plots the 10-frame future timeline side-by-side.

    Expected shapes: (1, 10, 32, 60, 1)
    """
    # Remove the batch and channel dimensions to get a clean 3D sequence array: (10, 32, 60)
    pred_frames = prediction[0, :, :, :, 0]
    true_frames = sample_target[0, :, :, :, 0]

    num_frames = pred_frames.shape[0]  # Should be 10

    # Initialize a 2-row grid layout (Row 1: Ground Truth, Row 2: Model Forecast)
    fig, axes = plt.subplots(2, num_frames, figsize=(20, 5))

    # Determine absolute maximum value across frames to fix the colorbar scaling
    max_val = max(true_frames.max(), pred_frames.max())

    for t in range(num_frames):
        # --- Row 1: True Future Distribution ---
        ax_true = axes[0, t]
        im_true = ax_true.imshow(true_frames[t], cmap='jet', vmin=0, vmax=max_val)
        ax_true.axis('off')
        if t == 0:
            ax_true.set_title("GROUND TRUTH\n(Actual Video)", loc='left', fontsize=12, fontweight='bold')
        else:
            ax_true.set_title(f"T + {t+1}")

        # --- Row 2: Predicted Future Distribution ---
        ax_pred = axes[1, t]
        im_pred = ax_pred.imshow(pred_frames[t], cmap='jet', vmin=0, vmax=max_val)
        ax_pred.axis('off')
        if t == 0:
            ax_pred.set_title("CONVLSTM FORECAST\n(Model Prediction)", loc='left', fontsize=12, fontweight='bold')

    # Add a unified color scale bar on the right edge of the figure panel
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im_true, cax=cbar_ax, label='Crowd Density Intensity')

    plt.suptitle("Sequential Crowd Density Tracking: True vs Predicted Timeline", fontsize=14, y=1.02, fontweight='bold')
    plt.show()


def plot_graph(video_grids, model):
    # 1. Build the sequences
    x_test, y_test = build_all_interleaved_sequences(video_grids, timesteps=10, future_steps=10, step=3)
    
    if x_test is None or len(x_test) == 0:
        print("❌ Error: Not enough frames to construct sequence blocks.")
        return
        
    xs = []
    print(f" Computing sequence-by-sequence MSE for {len(x_test)} windows...")
    
    # 2. Loop safely through the timeline
    for i in range(len(x_test)):
        sample_input = x_test[i : i + 1]   # Shape: (1, 10, 32, 60, 1)
        sample_target = y_test[i : i + 1]  # Shape: (1, 10, 32, 60, 1)
        
        # verbose=0 completely disables the progress bars
        pred = model.predict(sample_input, batch_size=1, verbose=0)
        
        # Calculate true normalized pixel-level MSE for this window
        pixel_level_mse = np.sum((pred - sample_target) ** 2) / sample_target.size
        xs.append(pixel_level_mse)
        
    # 3. Generate the visualization
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(x_test)), xs, label="Tracking Error (MSE)", color="blue", linewidth=1.5)
    
    # Add labels and styling
    plt.title("Spatiotemporal Forecasting Deviation Across Timeline", fontsize=12, fontweight='bold')
    plt.xlabel("Sequence Index Timeline", fontsize=10)
    plt.ylabel("Pixel-Level MSE", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.show()


def build_all_interleaved_sequences(video_grids, timesteps=10, future_steps=10, step=3):
    """
    Transforms a continuous stream of density maps into all interleaved phases.
    Processes frame-by-frame (stride=1) to fully exploit the dataset footprint.
    
    Parameters:
    -----------
    video_grids : list or np.ndarray
        List of 2D density grids generated by your CSRNet backbone.
    timesteps : int (default: 10)
        Number of historical lookback frames.
    future_steps : int (default: 10)
        Number of future target forecasting frames.
    step : int (default: 3)
        The internal frame-skipping gap matching your temporal stride cadence.
        
    Returns:
    --------
    X_array : np.ndarray -> Shape: (N, timesteps, H, W, 1)
    Y_array : np.ndarray -> Shape: (N, future_steps, H, W, 1)
    """
    X_list = []
    Y_list = []
    
    # Convert input to a clean numpy array for slicing performance
    video_grids_arr = np.array(video_grids)
    
    # Calculate the total window footprint required for one block
    input_span = timesteps * step       # 10 * 3 = 30 frames
    target_span = future_steps * step   # 10 * 3 = 30 frames
    total_window = input_span + target_span # 60 frames total
    
    if len(video_grids_arr) < total_window:
        print(f" Error: Input length ({len(video_grids_arr)}) is shorter than "
              f"the required total window sequence size ({total_window}).")
        return None, None
        
    # Slide a window across the entire timeline frame-by-frame (stride=1)
    # This automatically captures Phase 0 (1,4,7...), Phase 1 (2,5,8...), and Phase 2 (3,6,9...)
    for i in range(len(video_grids_arr) - total_window + 1):
        # Extract the continuous 60-frame memory segment
        full_block = video_grids_arr[i : i + total_window]
        
        # Interleave slice for input lookback (e.g., indices 0, 3, 6 ... 27)
        sample_input = full_block[0 : input_span : step]
        
        # Interleave slice for prediction target (e.g., indices 30, 33, 36 ... 57)
        sample_target = full_block[input_span : total_window : step]
        
        X_list.append(sample_input)
        Y_list.append(sample_target)
        
    # Convert structural lists into optimized numpy matrices
    X_array = np.array(X_list)  # Shape: (N, timesteps, H, W)
    Y_array = np.array(Y_list)  # Shape: (N, future_steps, H, W)
    
    # Expand axes to add the 5th dimension (channel dimension = 1) required by ConvLSTM2D
    X_array = X_array[..., np.newaxis]  # Shape: (N, timesteps, H, W, 1)
    Y_array = Y_array[..., np.newaxis]  # Shape: (N, future_steps, H, W, 1)
    
    print(f" Interleaved Processing Complete!")
    print(f" -> Total Extracted Sequences: {X_array.shape[0]}")
    print(f" -> Input Shape (X): {X_array.shape}")
    print(f" -> Target Shape (Y): {Y_array.shape}")
    
    return X_array, Y_array







####### MODELS ##########
forecasting_model=get_actual_convlstm()
density_model=get_weights_for_csrnet()
#####################################
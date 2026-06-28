import numpy as np
import torch
import cv2
import torchvision.transforms as transforms

class Camera:
    def __init__(self, camera_id="Camera_001", context_size=60, alpha=0.06, threshold_multiplier=1.5, sustained_window=5):
        """
        Production Multi-Camera Stream Node Object.
        Tracks unique running spatiotemporal statistics to dynamically process streaming buffers.
        """
        self.camera_id = camera_id
        self.alpha = alpha
        self.threshold_multiplier = threshold_multiplier
        self.context_size = context_size         # Total frames needed for 1 full structural evaluation window (60 frames)
        self.sustained_window = sustained_window # Number of consecutive frames needed to trigger a true alert
        
        # Unique Adaptive Statistical Signatures
        self.running_mean = 0.0
        self.running_std = 0.0
        self.is_calibrated = False
        
        # Calibration score cache to capture true trend volatility on startup
        self.calibration_scores = []
        self.calibration_limit = 15              # Accumulate 15 windows to run exact np.std() baseline
        
        # Concurrency/Streaming State Trackers
        self.consecutive_breaches = 0            # Tracks consecutive threshold violations across stream frames
        
        # Buffers
        self.frame_buffer = []                   # Holds incoming raw BGR frames from the camera feed
        self.grid_history = []                   # Holds past raw 2D density maps to support interleaving
        
        # Preprocessing setup matching architecture norms
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def receive_frames(self, raw_frames):
        """
        Appends new frames from a streaming slice to the camera internal execution pipeline.
        """
        self.frame_buffer.extend(raw_frames)

    def extract_density_grids(self, density_model, batch_size=4, downscale_factor=2.0):
        """
        Converts the raw frame buffer into optimized spatiotemporal density fields using the CSRNet backbone.
        """
        if not self.frame_buffer:
            return []

        density_maps = []
        target_width = int((960 / downscale_factor) // 16) * 16
        target_height = int((536 / downscale_factor) // 16) * 16

        frame_batch = []
        for frame in self.frame_buffer:
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_tensor = self.preprocess(rgb_frame)
            frame_batch.append(input_tensor)

            if len(frame_batch) == batch_size:
                stacked_batch = torch.stack(frame_batch).to(self.device)
                with torch.no_grad():
                    outputs = density_model(stacked_batch)
                batch_np = outputs.detach().cpu().numpy()[:, 0, :, :]
                for i in range(batch_np.shape[0]):
                    density_maps.append(batch_np[i].astype('float16'))
                frame_batch = []

        if frame_batch:
            stacked_batch = torch.stack(frame_batch).to(self.device)
            with torch.no_grad():
                outputs = density_model(stacked_batch)
            batch_np = outputs.detach().cpu().numpy()[:, 0, :, :]
            for i in range(batch_np.shape[0]):
                density_maps.append(batch_np[i].astype('float16'))

        self.frame_buffer = []
        return density_maps

    def check_for_anomaly(self, current_window_mse, adaptive_threshold):
        """
        Evaluates whether the current tracking deviation implies a genuine panic state.
        Filters out single-frame background noise anomalies using a sustained sequential tracking window.
        """
        # Step 1: Evaluate simple point breach
        if current_window_mse > adaptive_threshold:
            self.consecutive_breaches += 1
        else:
            # Drop window counter back down immediately to eliminate flickering false-alarms
            self.consecutive_breaches = max(0, self.consecutive_breaches - 1)

        # Step 2: Evaluate if breach is sustained over the target threshold frame block
        if self.consecutive_breaches >= self.sustained_window:
            return True
        return False

    def process_and_evaluate_stream(self, density_model, forecasting_model):
        """
        Processes pending frames, updates running scene baselines, and flags anomalous events.
        """
        new_grids = self.extract_density_grids(density_model)
        self.grid_history.extend(new_grids)

        if len(self.grid_history) < self.context_size:
            print(f"ℹ️ [{self.camera_id}] Building tracking context... ({len(self.grid_history)}/{self.context_size})")
            return {"camera_id": self.camera_id, "is_anomaly": False, "status": "GATHERING_CONTEXT"}

        video_grids_arr = np.array(self.grid_history)
        
        timesteps, future_steps, step = 10, 10, 3
        input_span = timesteps * step
        target_span = future_steps * step
        total_window = input_span + target_span

        full_block = video_grids_arr[-total_window:]
        sample_input = full_block[0 : input_span : step]           
        sample_target = full_block[input_span : total_window : step] 

        sample_input_5d = sample_input[np.newaxis, ..., np.newaxis]
        sample_target_5d = sample_target[np.newaxis, ..., np.newaxis]

        predicted_target = forecasting_model.predict(sample_input_5d, batch_size=1, verbose=0)
        current_window_mse = np.sum((predicted_target - sample_target_5d) ** 2) / sample_target_5d.size

        # Compute adaptive statistical upper bounds
        if not self.is_calibrated:
            self.calibration_scores.append(current_window_mse)
            
            # Keep compiling initial sequences until we have enough to check rolling trend volatility
            if len(self.calibration_scores) < self.calibration_limit:
                print(f"ℹ️ [{self.camera_id}] Calibrating rolling variance baselines... ({len(self.calibration_scores)}/{self.calibration_limit})")
                return {
                    "camera_id": self.camera_id, 
                    "is_anomaly": False, 
                    "status": "CALIBRATING"
                }
            
            # RESTORING THE HIGH-ACCURACY VOLATILITY INITIALIZATION MATH
            self.running_mean = np.mean(self.calibration_scores)
            self.running_std = np.std(self.calibration_scores) * 0.1  # Matches your exact high-accuracy setup
            
            self.is_calibrated = True
            is_anomaly = False
            adaptive_threshold = self.running_mean + (self.threshold_multiplier * self.running_std)
            
            # Memory cleanup
            del self.calibration_scores
        else:
            adaptive_threshold = self.running_mean + (self.threshold_multiplier * self.running_std)
            
            # Call the validation rule checker function
            is_anomaly = self.check_for_anomaly(current_window_mse, adaptive_threshold)

            # Update Running Metrics via Exponential Decay parameters safely
            self.running_mean = (self.alpha * current_window_mse) + ((1 - self.alpha) * self.running_mean)
            variance_step = self.alpha * ((current_window_mse - self.running_mean) ** 2) + ((1 - self.alpha) * (self.running_std ** 2))
            self.running_std = np.sqrt(variance_step)

        # Bound cache size to prevent memory leaks over permanent production runtimes
        max_retained_history = 120
        if len(self.grid_history) > max_retained_history:
            self.grid_history = self.grid_history[-self.context_size:]
        forecasted_sequence = predicted_target[0, :, :, :, 0].tolist()
        return {
            "camera_id": self.camera_id,
            "is_anomaly": is_anomaly,
            "consecutive_breaches": self.consecutive_breaches,
            "window_mse": float(current_window_mse),
            "adaptive_threshold": float(adaptive_threshold),
            "forecasted_grids": forecasted_sequence,
        }
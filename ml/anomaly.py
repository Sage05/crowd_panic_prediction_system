import numpy as np
from sklearn.ensemble import IsolationForest

def extract_frame_features(current_yolo_boxes, previous_yolo_boxes=None, fps=30):
    """
    Extracts high-level behavioral features from raw YOLO boxes for a single frame.
    
    yolo_boxes: List of coordinates [[x_min, y_min, x_max, y_max], ...]
    """
    # Feature 1: Total count of people in the current frame
    people_count = len(current_yolo_boxes)
    
    # Feature 2: Average velocity of the crowd
    mean_velocity = 0.0
    
    if previous_yolo_boxes is not None and len(previous_yolo_boxes) > 0 and people_count > 0:
        current_centers = []
        for box in current_yolo_boxes:
            # Calculate center points (x, y)
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            current_centers.append((cx, cy))
            
        prev_centers = []
        for box in previous_yolo_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            prev_centers.append((cx, cy))
            
        # Match displacements (simplification for baseline tracking)
        distances = []
        for curr in current_centers:
            # Find the closest person in the previous frame to calculate their individual movement
            min_dist = min([np.sqrt((curr[0] - p[0])**2 + (curr[1] - p[1])**2) for p in prev_centers])
            distances.append(min_dist)
            
        # Velocity = displacement * FPS (pixels per second)
        mean_velocity = np.mean(distances) * fps

    return [people_count, mean_velocity]


def run_engineered_baseline(video_features):
    """
    Trains and runs the Isolation Forest on structural crowd features.
    video_features: Array of shape (Total_Frames, 2) -> [people_count, mean_velocity]
    """
    # 5% contamination assumes 5% of our dataset might contain sudden panic spikes
    model = IsolationForest(contamination=0.05, random_state=42)
    
    # Train directly on the 2 macro features
    model.fit(video_features)
    predictions = model.predict(video_features)
    
    return predictions


if __name__ == "__main__":
    print("--- Testing Real Baseline Feature Extraction ---")
    
    # Frame 1: Normal dense crowd standing still
    frame1_boxes = [[100, 100, 150, 200], [200, 100, 250, 200], [300, 100, 350, 200]]
    
    # Frame 2: Same crowd suddenly moves rapidly (Panic state simulation)
    frame2_boxes = [[150, 100, 200, 200], [280, 100, 330, 200], [420, 100, 470, 200]]
    
    feat_normal = extract_frame_features(frame1_boxes, previous_yolo_boxes=None)
    feat_panic = extract_frame_features(frame2_boxes, previous_yolo_boxes=frame1_boxes)
    
    print(f"Normal Frame Metrics [Count, Avg Velocity]: {feat_normal}")
    print(f"Panic Frame Metrics  [Count, Avg Velocity]: {feat_panic}")
    
    # Mocking a video feature dataset matrix
    mock_video_features = np.array([
        [30, 5.0],  # Normal
        [32, 4.8],  # Normal
        [31, 62.1], # Anomaly (Sudden high velocity crowd burst)
        [29, 5.2],  # Normal
    ])
    
    decisions = run_engineered_baseline(mock_video_features)
    print(f"\nBaseline Isolation Forest Predictions: {decisions}")
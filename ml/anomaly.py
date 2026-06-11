import numpy as np
from sklearn.ensemble import IsolationForest

def calculate_grid_density(yolo_boxes, grid_shape=(10, 10), image_shape=(1080, 1920)):
    """
    Takes raw YOLO bounding boxes and distributes their weights across a grid 
    based on the percentage of area overlapping each cell.
    
    yolo_boxes: List of coordinates [[x_min, y_min, x_max, y_max], ...]
    """
    grid_h, grid_w = grid_shape
    img_h, img_w = image_shape
    
    density_matrix = np.zeros((grid_h, grid_w))
    
    cell_h = img_h / grid_h
    cell_w = img_w / grid_w
    
    for box in yolo_boxes:
        x_min, y_min, x_max, y_max = box
        box_area = (x_max - x_min) * (y_max - y_min)
        if box_area <= 0:
            continue
            
        start_row = int(y_min // cell_h)
        end_row = int(y_max // cell_h)
        start_col = int(x_min // cell_w)
        end_col = int(x_max // cell_w)
        
        start_row, end_row = max(0, start_row), min(grid_h - 1, end_row)
        start_col, end_col = max(0, start_col), min(grid_w - 1, end_col)
        
        for r in range(start_row, end_row + 1):
            for c in range(start_col, end_col + 1):
                int_x_min = max(x_min, c * cell_w)
                int_y_min = max(y_min, r * cell_h)
                int_x_max = min(x_max, (c + 1) * cell_w)
                int_y_max = min(y_max, (r + 1) * cell_h)
                
                int_area = max(0, int_x_max - int_x_min) * max(0, int_y_max - int_y_min)
                
                overlap_ratio = int_area / box_area
                density_matrix[r, c] += overlap_ratio
                
    return density_matrix
                
def detect_panic_baseline(flat_grid_data):
    """
    Baseline Isolation Forest model. 
    Trains on incoming grid features and flags abnormal frames.
    Returns: 1 for normal frame, -1 for anomaly/potential panic.
    """
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(flat_grid_data)
    return model.predict(flat_grid_data)

if __name__ == "__main__":
    print("--- Testing Step 1: Grid Density Matrix Generation ---")
    mock_yolo_boxes = [
        [900, 500, 1000, 600], 
        [950, 520, 1050, 620]
    ]
    
    density_map = calculate_grid_density(mock_yolo_boxes, grid_shape=(10, 10))
    print("Generated 10x10 Grid Layout (showing non-zero cells):")
    print(np.round(density_map[4:7, 4:7], 2))
    
    print("\n--- Testing Step 2: Isolation Forest Baseline ---")
    normal_frame = np.ones(100) * 0.1
    panic_frame = np.ones(100) * 4.5
    
    mock_dataset = np.array([normal_frame, normal_frame, panic_frame, normal_frame, normal_frame])
    
    predictions = detect_panic_baseline(mock_dataset)
    print("Frame analysis results (1 = Normal, -1 = Panic Anomaly):")
    print(predictions)
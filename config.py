"""
TrafficManagerCV Configuration File

Central configuration for all modules including:
- Model paths and parameters
- Detection thresholds
- MOG2 motion detection parameters
- SORT tracker settings
- Road analyzer configuration
- Speed estimation parameters
"""

import os
from pathlib import Path

# Project Paths

PROJECT_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = PROJECT_ROOT / "backend"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TESTS_DIR = PROJECT_ROOT / "tests"

# Create outputs directory if it doesn't exist
OUTPUTS_DIR.mkdir(exist_ok=True)

# Model Paths

# YOLO Models
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
YOLO_ALT_MODEL_PATH = MODELS_DIR / "yolo26n.pt"

# Trained SORT models (if any)
SORT_MODEL_PATH = None  # SORT is algorithm-based, not neural network

# Frame Processing

# Standard frame dimensions
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720

# Frame processing parameters
FRAME_SCALE = 1.0  # Scale factor for processing (1.0 = full resolution)
FRAME_SKIP = 0  # Process every Nth frame (0 = process all frames)
FPS_TARGET = 30  # Target FPS for output video
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections

# YOLO Detection Parameters

YOLO_CONFIG = {
    "model_path": YOLO_MODEL_PATH,
    "device": 0,  # 0 for GPU, "cpu" for CPU
    "conf": CONFIDENCE_THRESHOLD,  # Detection confidence threshold
    "iou": 0.45,  # IOU threshold for NMS
    "imgsz": 640,  # Input image size
    "max_det": 300,  # Maximum detections
    "classes": [2, 3, 5, 7],  # COCO classes: car, motorcycle, bus, truck
    "agnostic_nms": False,  # Class-agnostic NMS
    "verbose": False,  # Print progress
}

# MOG2 Motion Detection Parameters

MOG2_CONFIG = {
    "history": 500,  # Length of history in frames
    "var_threshold": 16,  # Variance threshold for detecting changes
    "detect_shadows": True,  # Detect shadows
    "learning_rate": -1,  # Learning rate (negative = auto)
    "min_area": 500,  # Minimum contour area in pixels
    "dilation_kernel_size": (7, 7),  # Kernel size for dilation
    "erosion_kernel_size": (3, 3),  # Kernel size for erosion
    "blur_kernel_size": (5, 5),  # Kernel size for Gaussian blur
}

# Hybrid Detection Parameters

HYBRID_CONFIG = {
    "validate_with_mog2": True,  # Use MOG2 to validate YOLO detections
    "motion_overlap_threshold": 0.3,  # Minimum overlap between YOLO and MOG2
    "mog2_confidence_weight": 0.7,  # Weight factor for MOG2 validation
    "min_detection_height": 20,  # Minimum detection bounding box height
    "max_detection_area": 0.9,  # Maximum detection area (% of frame)
}

# SORT Tracking Parameters

SORT_CONFIG = {
    "max_age": 30,  # Maximum frames to keep alive a track without detections
    "min_hits": 3,  # Minimum consecutive hits to start track
    "iou_threshold": 0.3,  # IOU threshold for assignment
    "use_kalman": True,  # Use Kalman filter for prediction
}

# Road Analyzer Configuration

ROAD_ANALYZER_CONFIG = {
    "frame_width": DEFAULT_FRAME_WIDTH,
    "frame_height": DEFAULT_FRAME_HEIGHT,
    "divider_pos": DEFAULT_FRAME_WIDTH // 2,  # Left/right split at center
    "buffer_zone": 30,  # Anti-flickering buffer zone in pixels
    "enable_lanes": True,  # Enable lane-wise analysis
    "num_lanes": 5,  # Number of lanes per road
    # Density thresholds
    "density_light_threshold": 5,  # Light: 0-5 vehicles
    "density_medium_threshold": 10,  # Medium: 6-10 vehicles
    # Heavy: >10 vehicles
}

# Speed Estimation Parameters

SPEED_CONFIG = {
    "calibration_distance_pixels": 300,  # Distance in pixels for calibration
    "calibration_distance_meters": 10,  # Real-world distance in meters
    "fps": FPS_TARGET,  # Frames per second
    "smoothing_window": 5,  # Number of frames for speed smoothing
    "min_confidence": 0.7,  # Minimum tracking confidence for speed calculation
}

# Visualization Parameters

VISUALIZATION_CONFIG = {
    # Colors (BGR format)
    "colors": {
        "divider": (0, 255, 255),  # Yellow for divider
        "road_a": (0, 255, 0),  # Green for Road A
        "road_b": (0, 0, 255),  # Red for Road B
        "light_traffic": (0, 255, 0),  # Green
        "medium_traffic": (0, 165, 255),  # Orange
        "heavy_traffic": (0, 0, 255),  # Red
        "text": (255, 255, 255),  # White
        "background": (0, 0, 0),  # Black
    },
    # Font settings
    "font": "cv2.FONT_HERSHEY_SIMPLEX",
    "font_scale": 0.7,
    "font_thickness": 2,
    # Bounding box thickness
    "bbox_thickness": 2,
    # Line thickness
    "line_thickness": 3,
}

# Evaluation Parameters

EVALUATION_CONFIG = {
    "metrics": ["precision", "recall", "f1_score", "fps"],
    "output_format": "json",
    "save_results": True,
    "results_dir": OUTPUTS_DIR / "evaluation",
}

# Logging & Debug

DEBUG_MODE = False  # Enable debug output
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
VERBOSE = False  # Print detailed information

# Video Input/Output

VIDEO_CONFIG = {
    "codec": "mp4v",  # Video codec (mp4v, XVID, MJPG, etc.)
    "fps": FPS_TARGET,
    "frame_size": (DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT),
}

# Helper Functions

def get_model_path(model_type="yolo"):
    """Get path to model file."""
    if model_type == "yolo":
        return YOLO_MODEL_PATH
    elif model_type == "yolo_alt":
        return YOLO_ALT_MODEL_PATH
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_output_path(filename):
    """Get path for output file."""
    return OUTPUTS_DIR / filename

def ensure_model_exists():
    """Check if required models exist."""
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {YOLO_MODEL_PATH}")
    return True

def print_config():
    """Print current configuration."""
    print("TrafficManagerCV Configuration")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Outputs Directory: {OUTPUTS_DIR}")
    print(f"\nYOLO Model: {YOLO_MODEL_PATH}")
    print(f"Frame Size: {DEFAULT_FRAME_WIDTH}x{DEFAULT_FRAME_HEIGHT}")
    print(f"Target FPS: {FPS_TARGET}")
    print(f"\nMOG2 History: {MOG2_CONFIG['history']}")
    print(f"SORT Max Age: {SORT_CONFIG['max_age']}")
    print(f"Road Analyzer Lanes: {ROAD_ANALYZER_CONFIG['num_lanes']}")
    print("-" * 40)

if __name__ == "__main__":
    # Print configuration when run directly
    print_config()

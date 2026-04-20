# TrafficManagerCV - Hybrid Vehicle Detection & Traffic Analysis System

Real-time vehicle detection, tracking, and traffic analysis using YOLO v8, MOG2 motion detection, and SORT tracking algorithm.

**Status:** Production-ready | **License:** Open Source | **Python:** 3.8+ | **GPU-Optional**

## Core Features

- **Hybrid Vehicle Detection**: Combines YOLO v8 object detection with MOG2 motion validation
- **Real-time Multi-Object Tracking**: SORT algorithm with persistent vehicle IDs
- **Dual-Road Traffic Analysis**: Per-road vehicle counting, density levels, and congestion estimation
- **Speed Estimation**: Pixel-to-real-world vehicle speed calculation (calibration-based)
- **Flexible Modes**: Run in hybrid mode (YOLO+MOG2) or MOG2-only for lighter systems
- **Performance Evaluation**: Detailed metrics comparison between detection methods
- **Real-time Visualization**: Live video output with vehicle trajectories and statistics

## Project Structure

```
TrafficManagerCV/
├── README.md                  # Main documentation (you are here)
├── README_ADVANCED.md         # Advanced topics & troubleshooting
├── config.py                  # Centralized configuration
├── requirements.txt           # Python dependencies
│
├── backend/                   # Main application code
│   ├── app.py                # Primary hybrid pipeline (YOLO+MOG2)
│   ├── mog2_app.py           # MOG2-only pipeline (lighter)
│   ├── compare_mog2_yolo.py  # Performance comparison tool
│   ├── detection/            # Detector modules
│   ├── tracking/             # SORT tracking module
│   ├── analysis/             # Road & lane analysis modules
│   ├── evaluation/           # Performance metrics
│   └── utils/                # Shared utilities
│
├── models/                    # Model weights (yolov8n.pt, etc.)
├── tests/                     # Test suite location
├── outputs/                   # Generated outputs & results
├── data/                      # Training/test data
└── runs/                      # Training runs & benchmarks
```

## Quick Start Guide

### 1. Setup Environment

```bash
# Clone and enter project
cd /path/to/TrafficManagerCV

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run on Sample Video

```bash
# Hybrid mode (YOLO + MOG2) - Recommended for accuracy
cd backend
python app.py

# Or on specific video
python app.py "/path/to/video.mp4"

# Or MOG2-only mode (lighter, faster)
python mog2_app.py "/path/to/video.mp4"
```

### 3. View Output

**Live Display:**
- Orange boxes = Road A vehicles (left half)
- Blue boxes = Road B vehicles (right half)
- Yellow line = Road divider
- Statistics shown in real-time

**Generated Files:**
- Video with bounding boxes → `outputs/video_output.mp4`
- Detection metrics → `outputs/metrics.json`
- Analysis results → `outputs/analysis.json`

### 2. Download Models

YOLOv8 models will auto-download on first use to `models/` directory.

### 3. Run Hybrid Pipeline (Default)

```bash
cd backend
python app.py "path/to/video.mp4"
```

**Features:**
- YOLO detections validated by MOG2 motion
- Real-time tracking with persistent IDs
- Per-road vehicle counting and speed estimation
- Displays live analysis overlays

### 4. Run MOG2-Only Pipeline

```bash
cd backend
python mog2_app.py "path/to/video.mp4" --show-mask
```

**Features:**
- Pure motion-based detection
- Optional YOLO calibration
- Motion mask visualization

### 5. Compare Detection Methods

```bash
cd backend
python compare_mog2_yolo.py "path/to/video.mp4"
```

**Output:**
- Side-by-side YOLO vs MOG2 comparison
- Performance metrics (precision, recall, F1 score)
- Detection accuracy analysis

## Configuration

Edit `config.py` to customize:

```python
# Detection
YOLO_CONFIG = {
    "conf": 0.5,        # Confidence threshold
    "iou": 0.45,        # NMS IOU threshold
    "classes": [2, 3, 5, 7],  # COCO vehicle classes
}

MOG2_CONFIG = {
    "history": 500,     # Motion history frames
    "var_threshold": 16,  # Motion variance threshold
}

# Tracking
SORT_CONFIG = {
    "max_age": 30,      # Max frames without detection
    "min_hits": 3,      # Minimum hits to start track
}

# Road Analysis
ROAD_ANALYZER_CONFIG = {
    "buffer_zone": 30,  # Anti-flickering buffer
    "enable_lanes": True,  # Lane tracking
    "num_lanes": 5,     # Lanes per road
}
```

## Core Components

### Detection
- **YOLODetector**: YOLOv8 nano model for vehicle detection
- **MOG2Detector**: Mixture of Gaussians for motion detection
- **HybridDetector**: YOLO + MOG2 validation (most accurate)

### Tracking
- **SortTracker**: SORT algorithm with persistent vehicle IDs
- Kalman filter prediction
- Hungarian algorithm assignment

### Analysis
- **RoadAnalyzer**: Divides frame into Road A/B, counts vehicles, estimates density
- **LaneAnalyzer**: Lane-wise vehicle distribution (optional)
- **SpeedEstimator**: Calculates vehicle speed from tracking

### Evaluation
- **MOG2Evaluator**: Performance metrics and comparison

## API Examples

### Hybrid Detection Pipeline

```python
from backend.detection import HybridDetector
from backend.tracking import SortTracker
from backend.analysis import RoadAnalyzer, SpeedEstimator
import cv2

# Initialize components
detector = HybridDetector()
tracker = SortTracker()
road_analyzer = RoadAnalyzer()
speed_estimator = SpeedEstimator()

# Process video
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect vehicles
    yolo_dets = detector.detect(frame)
    
    # Track vehicles
    tracks = tracker.update(yolo_dets)
    
    # Analyze traffic
    road_stats = road_analyzer.compute_road_stats(tracks, speed_estimator.speeds)
    
    print(f"Road A: {road_stats['Road A']['count']} vehicles")
    print(f"Road B: {road_stats['Road B']['count']} vehicles")

cap.release()
```

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

### Code Quality

```bash
# Format code
black backend/

# Check style
flake8 backend/
```

## Documentation

Detailed guides are archived in `archived_documentation/`:
- MOG2 detection guide
- Hybrid detection implementation
- Road analyzer documentation
- Performance evaluation details

## Performance

- **Detection**: ~20-30 FPS (GPU), ~5-10 FPS (CPU)
- **Tracking**: <1ms per frame
- **Analysis**: <2ms per frame
- **Overall**: Real-time performance on modern hardware

## Contributing

When modifying code:
1. Keep modules decoupled
2. Add tests to `tests/`
3. Update configuration in `config.py`
4. Follow PEP 8 style guide
5. Add docstrings to functions

## Known Limitations

- Lane detection limited to 5 lanes per road
- Speed estimation requires calibration
- MOG2 sensitive to lighting changes
- Performance varies by video resolution

## License

[Your License Here]

## Technologies Used

- **YOLOv8** - Real-time object detection
- **MOG2** - Motion detection algorithm
- **SORT** - Simple online and real-time tracking
- **OpenCV** - Computer vision library
- **SciPy** - Hungarian algorithm for assignment

## Support

For issues or questions, please refer to the detailed documentation in `archived_documentation/` or check the inline code comments.

---

**Last Updated:** April 12, 2026  
**Project Status:** Production Ready

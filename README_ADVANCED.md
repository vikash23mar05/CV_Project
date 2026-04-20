# TrafficManagerCV - Advanced Guide

Complete reference for configuration, customization, training, and troubleshooting.

---

## Configuration Guide

### Using config.py

All settings are centralized in [config.py](config.py). Modify parameters without changing code:

```python
from config import YOLO_CONFIG, MOG2_CONFIG, SORT_CONFIG, ROAD_ANALYZER_CONFIG

# Access configuration
print(YOLO_CONFIG["conf"])  # Detection confidence: 0.5
print(MOG2_CONFIG["history"])  # MOG2 history frames: 500
print(SORT_CONFIG["max_age"])  # Tracker max age: 30 frames
print(ROAD_ANALYZER_CONFIG["lanes"])  # Lanes per road: 5
```

### Key Configuration Parameters

#### YOLO Detection (YOLO_CONFIG)
```python
{
    "model_path": "models/yolov8n.pt",  # Nano model (fast)
    "conf": 0.5,                        # Confidence threshold (0-1)
    "iou": 0.45,                        # IoU threshold for NMS
    "classes": [2, 3, 5, 7],           # COCO classes: car, motorcycle, bus, truck
    "device": "cuda",                   # GPU if available, else CPU
}
```

#### MOG2 Detection (MOG2_CONFIG)
```python
{
    "history": 500,                    # Background model history frames
    "var_threshold": 16,               # Variance threshold for pixel detection
    "detect_shadows": True,            # Detect shadows as motion
    "kernel_size": 5,                  # Morphological operation kernel size
}
```

#### SORT Tracking (SORT_CONFIG)
```python
{
    "max_age": 30,                     # Frame threshold to remove dead track
    "min_hits": 3,                     # Hits before track is output
    "iou_threshold": 0.3,              # IoU threshold for Hungarian algorithm
}
```

#### Road Analysis (ROAD_ANALYZER_CONFIG)
```python
{
    "divider_pos": "center",           # Road divider position
    "buffer_zone": 30,                 # Pixels around divider
    "lanes": 5,                        # Lanes per road for analysis
    "density_thresholds": {
        "light": (0, 5),               # Light traffic: 0-5 vehicles
        "medium": (6, 10),             # Medium: 6-10 vehicles
        "heavy": (11, float('inf')),   # Heavy: 11+ vehicles
    }
}
```

### Tuning for Your Use Case

**For Accuracy (Traffic Law Enforcement)**
```python
YOLO_CONFIG["conf"] = 0.6              # Higher confidence
SORT_CONFIG["max_age"] = 50            # Keep tracks longer
MOG2_CONFIG["detect_shadows"] = False  # Reduce false positives
```

**For Speed (Real-time Monitoring)**
```python
YOLO_CONFIG["conf"] = 0.4              # Lower confidence
SORT_CONFIG["max_age"] = 15            # Shorter tracks
MOG2_CONFIG["history"] = 250           # Reduced history
```

**For Embedded Systems (CPU Only)**
```python
YOLO_CONFIG["device"] = "cpu"
YOLO_CONFIG["model_path"] = "models/yolov8n.pt"  # Smallest model
FRAME_PROCESSING["target_width"] = 640  # Smaller resolution
```

---

## API Reference

### Hybrid Pipeline

```python
from backend.detection import HybridDetector
from backend.tracking import SortTracker
from backend.analysis import RoadAnalyzer
import cv2

# Initialize
detector = HybridDetector()
tracker = SortTracker()
analyzer = RoadAnalyzer()

# Process video
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect vehicles
    detections = detector.detect(frame)
    
    # Track vehicles
    tracked_objects = tracker.update(detections)
    
    # Analyze traffic
    road_stats = analyzer.compute_road_stats(tracked_objects)
    
    # Use results
    print(f"Road A: {road_stats['road_a']['count']} vehicles")
    print(f"Road B: {road_stats['road_b']['count']} vehicles")

cap.release()
```

### MOG2-Only Pipeline

```python
from backend.detection import MOG2Detector
import cv2

detector = MOG2Detector()
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect motion
    mask = detector.detect(frame)
    contours = detector.get_contours(mask)
    
    # Use mask/contours
    cv2.imshow("Motion", mask)

cap.release()
```

### Speed Estimation

```python
from backend.analysis import SpeedEstimator

estimator = SpeedEstimator()

# Set calibration: 300 pixels = 10 meters in your video
estimator.set_calibration_points((150, 240), (450, 240), 10)

# Estimate speed from pixel-per-frame
pixels_per_frame = 5.0
speed_kmh = estimator.estimate_speed(pixels_per_frame)
print(f"Speed: {speed_kmh:.1f} km/h")
```

---

## Training Your Own Model

### 1. Prepare Dataset

```bash
# Format: COCO or YOLO format
# Expected structure:
# data/
#   ├── images/
#   │   ├── train/
#   │   └── val/
#   └── labels/
#       ├── train/
#       └── val/

# data.yaml should contain:
path: /path/to/data
train: images/train
val: images/val
nc: 1  # Number of classes
names: ['vehicle']  # Class names
```

### 2. Train Model

```bash
cd backend

# Train with ultralytics
python train_yolo.py \
    --model yolov8n.pt \
    --data data/data.yaml \
    --epochs 100 \
    --imgsz 640 \
    --device 0  # GPU device (0 for first GPU)
```

### 3. Evaluate and Deploy

```bash
# Evaluate model
python evaluate_mog2_performance.py \
    --model runs/detect/train/weights/best.pt \
    --video test_video.mp4

# Update config.py with new model path
# Then use: python app.py
```

---

## Performance Optimization

### Reduce Latency

```python
# In config.py
FRAME_PROCESSING = {
    "target_width": 640,      # Reduce from 1280
    "target_height": 360,     # Reduce from 720
    "scale_factor": 0.5,      # 50% downscaling
}

# Use smaller model
YOLO_CONFIG["model_path"] = "models/yolov8n.pt"  # Nano instead of small
```

### Increase Throughput

```python
# Process on GPU
YOLO_CONFIG["device"] = "cuda"

# Batch processing (if using offline)
detector = HybridDetector(batch_size=8)
results = detector.detect_batch([frame1, frame2, ...])
```

### Reduce Memory

```python
MOG2_CONFIG["history"] = 250  # Less history = less memory
SORT_CONFIG["max_age"] = 15   # Remove old tracks faster
```

---

## Troubleshooting

### Issue: YAML Not Found or Data Loading Fails

**Symptoms:** FileNotFoundError when running training

**Solutions:**
```bash
# Check data.yaml exists
ls data/data.yaml

# Verify COCO format structure
find data/images -type f | head  # Should show images
find data/labels -type f | head  # Should show .txt files

# Check file permissions
chmod +r data/data.yaml
chmod +r data/images/*.jpg
```

### Issue: Out of Memory (OOM)

**Symptoms:** CUDA/CPU runs out of memory

**Solutions:**
```python
# Reduce resolution
FRAME_PROCESSING["target_width"] = 480

# Use smaller model
YOLO_CONFIG["model_path"] = "models/yolov8n.pt"  # Nano

# Reduce batch size
SORT_CONFIG["max_age"] = 10  # Keep fewer tracks
```

### Issue: Low Detection Accuracy

**Symptoms:** Missing vehicles or too many false positives

**Solutions:**
```python
# Increase YOLO confidence
YOLO_CONFIG["conf"] = 0.6  # Skip uncertain detections

# Use hybrid mode (add MOG2 validation)
# Already enabled - YOLO detections validated by MOG2

# Train custom model on your data
# See "Training Your Own Model" section above
```

### Issue: Tracking IDs Keep Changing

**Symptoms:** Vehicle IDs change between frames

**Solutions:**
```python
# Increase SORT patience
SORT_CONFIG["max_age"] = 50  # Allow gaps between detections

# Increase minimum hits requirement
SORT_CONFIG["min_hits"] = 2  # Start tracking faster

# Adjust IoU threshold
SORT_CONFIG["iou_threshold"] = 0.4  # More lenient matching
```

### Issue: Speed Estimation is Wrong

**Symptoms:** Calculated speeds don't match reality

**Solutions:**
```bash
# Re-calibrate the system
# Method: Mark two points in the video that are known distance apart
# Then update SpeedEstimator calibration

python -c "
from backend.analysis import SpeedEstimator
estimator = SpeedEstimator()
# Set calibration: if 2 points are 10 meters apart and 300 pixels apart
estimator.set_calibration_points((100, 250), (400, 250), 10)
"
```

---

## Evaluation & Metrics

### Compare Detection Methods

```bash
cd backend

# Run full evaluation
python compare_mog2_yolo.py video.mp4 \
    --metrics precision,recall,f1 \
    --output eval_results.json
```

### View Detailed Metrics

```bash
# View metrics from last run
cat outputs/metrics.json | python -m json.tool

# Expected format:
{
    "frame_count": 1200,
    "total_vehicles": 345,
    "avg_fps": 25.3,
    "yolo": {
        "detections": 1456,
        "confidence_mean": 0.72
    },
    "mog2": {
        "motion_objects": 1289,
        "shadow_ratio": 0.12
    },
    "tracking": {
        "active_tracks": 12,
        "completed_tracks": 34
    }
}
```

---

## Development & Debugging

### Enable Verbose Logging

```bash
# Set debug level
export DEBUG=1
python backend/app.py video.mp4

# Or in code:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Debug Detections

```python
# Visualize detection types
from backend.detection import HybridDetector

detector = HybridDetector(debug=True)
yolo_detections = detector.yolo_detector.detect(frame)
mog2_mask = detector.mog2_detector.detect(frame)
final_detections = detector.detect(frame)

# Compare:
print(f"YOLO: {len(yolo_detections)} detections")
print(f"MOG2: {np.count_nonzero(mog2_mask)} pixels")
print(f"Hybrid: {len(final_detections)} detections")
```

### Profile Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run processing
detector = HybridDetector()
detections = detector.detect(frame)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

---

## Additional Resources

- **ultralytics/yolov8** - Official YOLOv8 documentation
- **scipy** - Documentation for SORT algorithm implementation
- **OpenCV** - MOG2 background subtraction reference
- **Data Format** - See `data.yaml` example in data/ directory

---

## Common Use Cases

### Road Congestion Monitoring
```python
# Use config: ROAD_ANALYZER_CONFIG with density thresholds
# Output: alerts when density exceeds threshold
if road_stats['road_a']['density'] == 'heavy':
    send_alert("Road A congested")
```

### License Plate Recognition
```python
# Extend with: pip install easyocr
# Crop region around detection, run OCR
# Store with vehicle track ID for enforcement
```

### Vehicle Classification
```python
# Use: YOLO_CONFIG["classes"] = [2, 3, 5, 7, 8, 9]
# Classify: car, motorcycle, bus, truck, van, taxi
# Separate lanes by vehicle type
```

### Highway Incident Detection
```python
# Monitor sudden drops in vehicle count
# Detect unusual speed patterns
# Alert on vehicle stopping in traffic lanes
```

---

## Getting Help

**Check these first:**
1. This file (README_ADVANCED.md)
2. Code comments in relevant module
3. Example scripts in backend/
4. GitHub issues for known problems

**Common errors:**
- Model file missing → Check models/ directory
- CUDA errors → Verify GPU and nvidia-smi
- Memory errors → See "Performance Optimization"
- Accuracy issues → See troubleshooting section

---

**Last Updated:** April 12, 2026  
**Status:** Production Ready


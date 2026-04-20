
import cv2
import glob
import os
import sys

# Fix Wayland display issues on Linux
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_FONT_DPI", "96")

# Allow imports from backend folder
sys.path.insert(0, os.path.dirname(__file__))

from detection.yolo_detector    import YOLODetector
from detection.hybrid_detector  import HybridDetector, HybridAnalyzer
from evaluation.mog2_evaluator  import MOG2BackgroundSubtractor
from tracking.sort_tracker      import SortTracker
from analysis.speed_estimator   import SpeedEstimator
from analysis.road_analyzer     import RoadAnalyzer


# Settings
VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "videos")
BOX_COLOR = (0, 255, 0)  # green boxes
BOX_THICKNESS = 2
MAX_AGE = 5  # tracker keeps object for 5 frames
MIN_HITS = 1  # show ID after 1 detection
IOU_THRESHOLD = 0.3  # matching threshold
FRAME_W = 1280
ROI_X_MIN = 552  # tuned to right-hand carriageway left edge
ROI_X_MAX = 1160  # tuned to right-hand carriageway right edge
ROI_Y_TOP = 220   # lane perspective starts higher for this video
ROI_Y_BOTTOM = 720

# Hybrid detection settings
USE_HYBRID_DETECTION = True  # Enable YOLO + MOG2 hybrid mode
USE_MOG2_DETECTION = True    # Enable separate MOG2 motion detection
USE_MOTION_OVERLAY = False   # Show MOG2 motion mask overlay

DATASET_SAMPLE_VIDEO = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "archive(2)",
    "Vehicle_Detection_Image_Dataset",
    "sample_video.mp4",
)


# ── Per-video processing ──────────────────────────────────────────────────────

def _compute_iou(box1, box2):
    """Compute Intersection over Union between two boxes [x1,y1,x2,y2]."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def compute_traffic_state(total, avg_speed):
    """Simple temporary traffic state without lane analysis."""
    if total < 6 and avg_speed > 10:
        return "LIGHT"
    elif total > 12 or avg_speed < 4:
        return "HEAVY"
    return "MEDIUM"

def process_video(video_path, loop=True):
    """Run full pipeline on a video with two-road analysis. Loop until Ctrl+C if loop=True."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {video_path} — skipping.")
        return True

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = max(int(1000 / fps), 1) if fps and fps > 1 else 33

    print(f"\n[INFO] Processing: {os.path.basename(video_path)}")
    print(f"[INFO] Hybrid Detection: {'ENABLED' if USE_HYBRID_DETECTION else 'DISABLED'}")
    print("[INFO] Press Ctrl+C in the terminal to stop.")
    if USE_MOTION_OVERLAY:
        print("[INFO] Motion overlay: ENABLED (red = motion detected)")

    # Create display windows
    cv2.namedWindow("Traffic Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Detection", 1280, 720)

    FRAME_HEIGHT = 720
    FRAME_WIDTH = 1280

    def reset_pipeline():
        components = (
            YOLODetector(),
            SortTracker(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD),
            SpeedEstimator(),
            RoadAnalyzer(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT),
        )
        
        # Initialize hybrid detector if enabled
        if USE_HYBRID_DETECTION:
            mog2 = MOG2BackgroundSubtractor(history=500, var_threshold=16)
            hybrid = HybridDetector(components[0], mog2)
            return components + (hybrid, mog2)
        
        return components

    pipeline = reset_pipeline()
    detector = pipeline[0]
    tracker = pipeline[1]
    estimator = pipeline[2]
    road_analyzer = pipeline[3]
    hybrid_detector = pipeline[4] if len(pipeline) > 4 else None
    mog2 = pipeline[5] if len(pipeline) > 5 else None

    while True:
        ret, frame = cap.read()
        if not ret:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                pipeline = reset_pipeline()
                detector = pipeline[0]
                tracker = pipeline[1]
                estimator = pipeline[2]
                road_analyzer = pipeline[3]
                hybrid_detector = pipeline[4] if len(pipeline) > 4 else None
                mog2 = pipeline[5] if len(pipeline) > 5 else None
                continue
            break  # End of video

        # Resize to 1280x720 for processing
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Detect vehicles with YOLOv8 or Hybrid (YOLO + MOG2)
        hybrid_detections = []  # Store hybrid detections for drawing later
        if USE_HYBRID_DETECTION and hybrid_detector:
            # Use hybrid detector that validates with MOG2
            hybrid_detections = hybrid_detector.detect(frame, use_motion_validation=True)
            # Convert to original format: (x1, y1, x2, y2, confidence)
            detections = [(x1, y1, x2, y2, conf) for x1, y1, x2, y2, conf, _ in hybrid_detections]
            motion_validated_flags = {(int(x1), int(y1), int(x2), int(y2)): validated 
                                     for x1, y1, x2, y2, _, validated in hybrid_detections}
        else:
            # Use original YOLO detector only
            detections = detector.detect(frame)
            motion_validated_flags = {}

        # Build confidence map: vehicle_id -> confidence score
        # Map detections to a dict of box -> confidence for later lookup
        detection_confidences = {(int(d[0]), int(d[1]), int(d[2]), int(d[3])): d[4] for d in detections}

        # Track vehicles across frames
        tracked_vehicles = tracker.update(detections)

        # Map tracked vehicle IDs to confidence scores
        # Match tracked boxes to original detections to get confidence
        confidence_map = {}
        for tracked in tracked_vehicles:
            x1, y1, x2, y2, vid = tracked
            # Find closest matching detection
            best_conf = 0.5  # default confidence
            for det in detections:
                dx1, dy1, dx2, dy2, conf = det
                # Check if boxes overlap significantly
                iou = _compute_iou([x1, y1, x2, y2], [dx1, dy1, dx2, dy2])
                if iou > 0.3:  # Good match
                    best_conf = conf
                    break
            confidence_map[vid] = best_conf

        # Speed estimation for all tracked vehicles
        speeds = estimator.estimate_speed(tracked_vehicles)

        # Assign vehicles to roads and compute statistics
        road_stats = road_analyzer.compute_road_stats(tracked_vehicles, speeds)

        # Draw road divider line and labels
        frame = road_analyzer.draw_road_divider(frame, color=(0, 255, 255), thickness=2)
        frame = road_analyzer.draw_road_labels(frame, font_scale=0.75, thickness=2)

        # Draw road statistics (vehicle count, density, avg speed)
        frame = road_analyzer.draw_road_statistics(
            frame,
            road_stats,
            font_scale=0.65,
            thickness=2,
        )

        # Draw detected vehicles with IDs, speeds, and confidence
        if USE_HYBRID_DETECTION and hybrid_detections:
            # Use hybrid drawing with GREEN for motion-validated, ORANGE for static
            frame = HybridAnalyzer.draw_detections(frame, hybrid_detections, draw_motion_status=True)
        else:
            # Draw with road-based colors (original logic)
            frame = road_analyzer.draw_detected_vehicles(
                frame,
                road_stats,
                speeds,
                confidence_map,  # Pass confidence scores
                box_thickness=2,
                font_scale=0.45,
            )

        # Draw MOG2 motion detection boxes (RED) as separate layer
        if USE_MOG2_DETECTION and mog2:
            mog2_boxes = mog2.get_bounding_boxes(frame, min_area=400, max_area=None)
            frame = HybridAnalyzer.draw_mog2_detections(frame, mog2_boxes)

        # Show current filename at bottom
        cv2.putText(
            frame,
            os.path.basename(video_path),
            (20, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Optional motion overlay
        if USE_MOTION_OVERLAY and mog2:
            motion_mask = mog2.get_foreground_mask(frame)
            frame = HybridAnalyzer.draw_motion_overlay(frame, motion_mask, alpha=0.2)

        # Show windows
        cv2.imshow("Traffic Detection", frame)

        # ESC skips to next video
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == 27:
            print("[INFO] ESC pressed - skipping to next video.")
            cap.release()
            return False

    cap.release()
    return True


# Main entry point

def main():
    if len(sys.argv) > 1:
        source = sys.argv[1]
        if os.path.isdir(source):
            videos_dir = os.path.normpath(source)
            video_files = (
                glob.glob(os.path.join(videos_dir, "*.avi")) +
                glob.glob(os.path.join(videos_dir, "*.mp4")) +
                glob.glob(os.path.join(videos_dir, "*.mov")) +
                glob.glob(os.path.join(videos_dir, "*.MOV"))
            )
        else:
            video_files = [source]
        video_files.sort()
    else:
        # Collect all supported video files in the videos directory.
        videos_dir = os.path.normpath(VIDEOS_DIR)
        video_files = (
            glob.glob(os.path.join(videos_dir, "*.avi")) +
            glob.glob(os.path.join(videos_dir, "*.mp4")) +
            glob.glob(os.path.join(videos_dir, "*.mov")) +
            glob.glob(os.path.join(videos_dir, "*.MOV"))
        )
        if not video_files and os.path.exists(DATASET_SAMPLE_VIDEO):
            video_files = [DATASET_SAMPLE_VIDEO]
        video_files.sort()   # process in alphabetical order

    if not video_files:
        print(f"[ERROR] No video files found in: {VIDEOS_DIR}")
        print("Pass a video path as an argument, or place videos in data/videos.")
        sys.exit(1)

    print(f"[INFO] Found {len(video_files)} video(s)")

    # The lane analyser is built once and reused across all videos because
    # the lane geometry stays the same for a fixed camera.
    for idx, video_path in enumerate(video_files, start=1):
        print(f"[INFO] Video {idx}/{len(video_files)}: {os.path.basename(video_path)}")
        process_video(video_path, loop=True)

    cv2.destroyAllWindows()
    print("[INFO] All videos processed. Exiting.")


if __name__ == "__main__":
    main()


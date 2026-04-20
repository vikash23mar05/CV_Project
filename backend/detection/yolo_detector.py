import os

from ultralytics import YOLO


class YOLODetector:
    """YOLOv8 detector for vehicle classes used by the tracking pipeline."""

    VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "vehicle"}
    
    # Filtering thresholds to reduce false positives
    CONFIDENCE_THRESHOLD = 0.65     # Only keep detections with confidence >= 65%
    MIN_ASPECT_RATIO = 0.4          # Vehicle width/height >= 0.4 (exclude tall narrow signs)
    MAX_ASPECT_RATIO = 3.0          # Vehicle width/height <= 3.0 (exclude very wide objects)
    MIN_SIZE = 20                   # Minimum bounding box dimension (pixels)

    def __init__(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        trained_weights = os.path.join(
            project_root,
            "runs",
            "detect",
            "train6",
            "weights",
            "best.pt",
        )

        self.device = "cuda" if self._cuda_available() else "cpu"
        self.weights_path = trained_weights if os.path.exists(trained_weights) else "yolov8n.pt"
        self.model = YOLO(self.weights_path)
        self.model.to(self.device)

    @staticmethod
    def _cuda_available():
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def detect(self, frame):
        """Run inference and return detections as [[x1,y1,x2,y2,conf], ...].
        
        Applies filtering to reduce false positives:
        - Confidence threshold (50%)
        - Aspect ratio (0.4 - 3.0) to exclude signs/boards
        - Minimum size (20px) to exclude noise
        """
        results = self.model(frame, verbose=False, device=self.device)
        if not results:
            return []

        result = results[0]
        names = result.names
        detections = []

        for box in result.boxes:
            cls_idx = int(box.cls[0].item())
            cls_name = str(names.get(cls_idx, "")).lower()
            if cls_name not in self.VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            
            # Confidence threshold
            if conf < self.CONFIDENCE_THRESHOLD:
                continue
            
            # Size constraints
            width = x2 - x1
            height = y2 - y1
            if width < self.MIN_SIZE or height < self.MIN_SIZE:
                continue
            
            # Aspect ratio (exclude tall narrow signs like billboards)
            aspect_ratio = width / height if height > 0 else 1.0
            if aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO:
                continue
            
            detections.append([x1, y1, x2, y2, conf])

        return detections
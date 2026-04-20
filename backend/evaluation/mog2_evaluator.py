"""MOG2 evaluation against YOLO detections."""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from detection.yolo_detector import YOLODetector


# Utility Functions

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Compute union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_detections(mog2_boxes: List[List[float]], 
                     yolo_boxes: List[List[float]], 
                     iou_threshold: float = 0.5) -> Tuple[int, int, int]:
    """
    Match MOG2 detections with YOLO detections using IoU.
    
    Args:
        mog2_boxes: List of [x1, y1, x2, y2] from MOG2
        yolo_boxes: List of [x1, y1, x2, y2] from YOLO
        iou_threshold: IoU threshold for a match (default 0.5)
    
    Returns:
        (TP, FP, FN) counts for this frame
    """
    tp = 0
    fp = 0
    fn = 0
    
    matched_yolo_indices = set()
    
    # Try to match each MOG2 box to a YOLO box
    for mog2_box in mog2_boxes:
        best_iou = 0
        best_yolo_idx = -1
        
        for yolo_idx, yolo_box in enumerate(yolo_boxes):
            if yolo_idx in matched_yolo_indices:
                continue  # Already matched
            
            iou = compute_iou(mog2_box, yolo_box)
            if iou > best_iou:
                best_iou = iou
                best_yolo_idx = yolo_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_yolo_indices.add(best_yolo_idx)
        else:
            fp += 1
    
    # Unmatched YOLO boxes are False Negatives (MOG2 missed them)
    fn = len(yolo_boxes) - len(matched_yolo_indices)
    
    return tp, fp, fn


# MOG2 Background Subtractor

class MOG2BackgroundSubtractor:
    """MOG2-based foreground detection with noise filtering."""
    
    def __init__(self, history=500, var_threshold=16, detect_shadows=True):
        """
        Initialize MOG2 background subtractor.
        
        Args:
            history: Number of frames to remember (default 500)
            var_threshold: Variance threshold (default 16)
            detect_shadows: Whether to detect shadows (default True)
        """
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        
        # Morphological kernel for noise removal
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    
    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Get foreground mask from MOG2.
        
        Args:
            frame: Input frame
        
        Returns:
            Binary foreground mask
        """
        fg_mask = self.mog2.apply(frame)
        
        # Remove shadows (value 128 in MOG2 output)
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to reduce noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        
        return fg_mask
    
    def get_bounding_boxes(self, frame: np.ndarray, 
                          min_area: float = 400,
                          max_area: float = None) -> List[List[float]]:
        """
        Extract bounding boxes from foreground mask.
        
        Args:
            frame: Input frame (used for height/width)
            min_area: Minimum contour area to consider (default 400 pixels²)
            max_area: Maximum contour area (default frame area)
        
        Returns:
            List of [x1, y1, x2, y2] bounding boxes
        """
        if max_area is None:
            max_area = frame.shape[0] * frame.shape[1]
        
        fg_mask = self.get_foreground_mask(frame)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])
        
        return boxes


# Metrics Accumulator

class MetricsAccumulator:
    """Accumulate and compute evaluation metrics across frames."""
    
    def __init__(self):
        """Initialize metrics."""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.frame_count = 0
        self.per_frame_metrics = []
    
    def update(self, tp: int, fp: int, fn: int):
        """
        Update metrics with frame-level counts.
        
        Args:
            tp: True Positives in this frame
            fp: False Positives in this frame
            fn: False Negatives in this frame
        """
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.frame_count += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.per_frame_metrics.append({
            'frame': self.frame_count,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    def get_aggregated_metrics(self) -> Dict:
        """
        Get aggregated metrics across all frames.
        
        Returns:
            Dict with Precision, Recall, F1 Score
        """
        precision = (self.total_tp / (self.total_tp + self.total_fp) 
                    if (self.total_tp + self.total_fp) > 0 else 0)
        recall = (self.total_tp / (self.total_tp + self.total_fn) 
                 if (self.total_tp + self.total_fn) > 0 else 0)
        f1 = (2 * (precision * recall) / (precision + recall) 
             if (precision + recall) > 0 else 0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_tp': self.total_tp,
            'total_fp': self.total_fp,
            'total_fn': self.total_fn,
            'total_frames': self.frame_count
        }
    
    def print_summary(self):
        """Print final metrics summary."""
        metrics = self.get_aggregated_metrics()
        
        print("MOG2 EVALUATION SUMMARY")
        print(f"Total Frames Processed: {metrics['total_frames']}")
        print(f"Total True Positives:    {metrics['total_tp']}")
        print(f"Total False Positives:   {metrics['total_fp']}")
        print(f"Total False Negatives:   {metrics['total_fn']}")
        print(f"\nPrecision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1 Score:  {metrics['f1']:.4f}")


# Visualization

def draw_detections(frame: np.ndarray,
                   mog2_boxes: List[List[float]],
                   yolo_boxes: List[List[float]],
                   metrics: Dict,
                   draw_matched: bool = True) -> np.ndarray:
    """
    Draw MOG2 and YOLO detections on frame with different colors.
    
    Args:
        frame: Input frame
        mog2_boxes: MOG2 bounding boxes [x1, y1, x2, y2]
        yolo_boxes: YOLO bounding boxes [x1, y1, x2, y2]
        metrics: Metrics dict with TP, FP, FN
        draw_matched: Whether to highlight matched detections in green
    
    Returns:
        Frame with drawn detections
    """
    frame = frame.copy()
    
    # Find matched boxes for highlighting
    matched_mog2_indices = set()
    matched_yolo_indices = set()
    
    if draw_matched:
        for mog2_idx, mog2_box in enumerate(mog2_boxes):
            best_iou = 0
            best_yolo_idx = -1
            
            for yolo_idx, yolo_box in enumerate(yolo_boxes):
                iou = compute_iou(mog2_box, yolo_box)
                if iou > best_iou:
                    best_iou = iou
                    best_yolo_idx = yolo_idx
            
            if best_iou >= 0.5:
                matched_mog2_indices.add(mog2_idx)
                matched_yolo_indices.add(best_yolo_idx)
    
    # Draw MOG2 boxes (red for unmatched, green for matched)
    for idx, box in enumerate(mog2_boxes):
        x1, y1, x2, y2 = map(int, box)
        
        if draw_matched and idx in matched_mog2_indices:
            color = (0, 255, 0)  # Green for matched
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "TP", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
        else:
            color = (0, 0, 255)  # Red for unmatched (FP)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "FP", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
    
    # Draw YOLO boxes (blue for unmatched, green for matched)
    for idx, box in enumerate(yolo_boxes):
        x1, y1, x2, y2 = map(int, box)
        
        if draw_matched and idx in matched_yolo_indices:
            color = (0, 255, 0)  # Green for matched
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        else:
            color = (255, 0, 0)  # Blue for unmatched (FN)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "FN", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
    
    # Draw metrics on frame
    metrics_text = [
        f"TP: {metrics.get('tp', 0)}  FP: {metrics.get('fp', 0)}  FN: {metrics.get('fn', 0)}",
        f"Precision: {metrics.get('precision', 0):.2%}  Recall: {metrics.get('recall', 0):.2%}",
        f"F1: {metrics.get('f1', 0):.3f}"
    ]
    
    y_offset = 30
    for text in metrics_text:
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 25
    
    # Legend
    cv2.rectangle(frame, (10, frame.shape[0] - 70), (300, frame.shape[0] - 10), 
                  (0, 0, 0), -1)
    cv2.putText(frame, "Red=FP (MOG2 only)", (20, frame.shape[0] - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, "Blue=FN (YOLO only)", (20, frame.shape[0] - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(frame, "Green=TP (Both matched)", (20, frame.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame


def draw_foreground_mask(frame: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Draw foreground mask overlay on frame.
    
    Args:
        frame: Original frame
        fg_mask: Foreground mask from MOG2
    
    Returns:
        Frame with mask overlay
    """
    frame = frame.copy()
    
    # Convert mask to 3-channel for overlay
    mask_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    mask_color[fg_mask > 0] = [0, 255, 0]  # Green for foreground
    
    # Blend with original frame
    alpha = 0.3
    frame = cv2.addWeighted(frame, 1 - alpha, mask_color, alpha, 0)
    
    return frame

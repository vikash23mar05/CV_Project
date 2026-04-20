#!/usr/bin/env python3
"""
Hybrid Detector: YOLO + MOG2 Integration

Combines YOLO (high accuracy detection) with MOG2 (motion validation)
for improved traffic detection in busy scenes.

Strategies for integrating YOLO and MOG2.

Expected improvements:
- Reduce false positives (static objects)
- Increase detection confidence for moving vehicles
- Detect anomalies (vehicles suddenly stopped)
"""

import cv2
import numpy as np
from typing import List, Tuple

class HybridDetector:
    """
    Combines YOLO and MOG2 for improved detection accuracy.
    
    YOLO provides: High accuracy object detection
    MOG2 provides: Motion validation and foreground masking
    """
    
    def __init__(self, yolo_detector, mog2_subtractor):
        """
        Initialize hybrid detector.
        
        Args:
            yolo_detector: YOLODetector instance
            mog2_subtractor: MOG2BackgroundSubtractor instance
        """
        self.yolo = yolo_detector
        self.mog2 = mog2_subtractor
        self.motion_threshold = 0.3  # Min overlap with motion for validation
        
    def detect(self, frame: np.ndarray, use_motion_validation: bool = True):
        """
        Detect vehicles using hybrid approach.
        
        Args:
            frame: Input frame
            use_motion_validation: If True, validate with MOG2; if False, YOLO only
        
        Returns:
            List of (x1, y1, x2, y2, confidence, motion_validated)
            - confidence: Boosted if motion-validated
            - motion_validated: Boolean flag
        """
        # YOLO detections
        yolo_detections = self.yolo.detect(frame)
        if not yolo_detections:
            return []
        
        if not use_motion_validation:
            # Mode 1: YOLO only (original behavior)
            return [(x1, y1, x2, y2, conf, False) for x1, y1, x2, y2, conf in yolo_detections]
        
        # MOG2 motion mask and detections
        motion_mask = self.mog2.get_foreground_mask(frame)
        mog2_boxes = self.mog2.get_bounding_boxes(frame, min_area=400)
        
        # Validate each YOLO detection against motion
        validated_detections = []
        
        for x1, y1, x2, y2, conf in yolo_detections:
            # Check motion overlap
            motion_overlap = self._compute_motion_overlap(
                x1, y1, x2, y2, 
                motion_mask
            )
            
            # Check if this vehicle has corresponding MOG2 detection
            mog2_match = self._find_matching_mog2_box(
                x1, y1, x2, y2,
                mog2_boxes
            )
            
            # Decision logic
            motion_validated = motion_overlap > self.motion_threshold or mog2_match
            
            # Boost confidence if motion-validated
            if motion_validated:
                boosted_conf = min(1.0, conf + 0.15)  # Boost by 15%
            else:
                boosted_conf = conf * 0.85  # Reduce by 15% (stationary)
            
            validated_detections.append(
                (x1, y1, x2, y2, boosted_conf, motion_validated)
            )
        
        return validated_detections
    
    def _compute_motion_overlap(self, x1: int, y1: int, x2: int, y2: int, 
                                motion_mask: np.ndarray) -> float:
        """
        Compute what fraction of YOLO box contains motion.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            motion_mask: MOG2 foreground mask (binary)
        
        Returns:
            Overlap ratio (0.0 - 1.0)
        """
        # Extract box region from mask (convert to int for slicing)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box_region = motion_mask[y1:y2, x1:x2]
        
        if box_region.size == 0:
            return 0.0
        
        # Compute fraction of motion pixels in box
        motion_pixels = np.count_nonzero(box_region)
        total_pixels = box_region.size
        
        overlap = motion_pixels / total_pixels
        return overlap
    
    def _find_matching_mog2_box(self, x1: int, y1: int, x2: int, y2: int,
                                mog2_boxes: List, iou_threshold: float = 0.3) -> bool:
        """
        Check if YOLO box matches any MOG2 box.
        
        Args:
            x1, y1, x2, y2: YOLO bounding box
            mog2_boxes: List of MOG2 boxes
            iou_threshold: Min IoU to consider match
        
        Returns:
            True if matching MOG2 box found
        """
        for mx1, my1, mx2, my2 in mog2_boxes:
            iou = self._compute_iou(x1, y1, x2, y2, mx1, my1, mx2, my2)
            if iou > iou_threshold:
                return True
        
        return False
    
    @staticmethod
    def _compute_iou(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b) -> float:
        """Compute Intersection over Union."""
        # Convert to int in case of float coordinates
        x1a, y1a, x2a, y2a = int(x1a), int(y1a), int(x2a), int(y2a)
        x1b, y1b, x2b, y2b = int(x1b), int(y1b), int(x2b), int(y2b)
        
        xi1 = max(x1a, x1b)
        yi1 = max(y1a, y1b)
        xi2 = min(x2a, x2b)
        yi2 = min(y2a, y2b)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area_a = (x2a - x1a) * (y2a - y1a)
        area_b = (x2b - x1b) * (y2b - y1b)
        union = area_a + area_b - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_stats(self) -> dict:
        """Get statistics about detection quality."""
        return {
            "motion_threshold": self.motion_threshold,
            "description": "Hybrid YOLO+MOG2 detector"
        }


class HybridAnalyzer:
    """
    Analyze and visualize hybrid detection results.
    """
    
    @staticmethod
    def draw_detections(frame: np.ndarray, 
                       detections: List,
                       draw_motion_status: bool = True) -> np.ndarray:
        """
        Draw detections with motion validation status.
        
        Args:
            frame: Input frame
            detections: List of (x1, y1, x2, y2, conf, motion_validated)
            draw_motion_status: If True, show motion status
        
        Returns:
            Frame with drawn detections
        """
        result = frame.copy()
        
        motion_validated_count = 0
        total_count = len(detections)
        
        for x1, y1, x2, y2, conf, motion_validated in detections:
            # Convert to integers for OpenCV
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Choose color based on motion validation
            if motion_validated:
                color = (0, 255, 0)  # Green = motion-validated ✓
                motion_validated_count += 1
            else:
                color = (0, 165, 255)  # Orange = stationary (watch out!)
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence and status
            status = "OK" if motion_validated else "STATIC"
            label = f"{conf*100:.0f}% {status}"
            
            cv2.putText(
                result, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )
        
        # Draw summary
        summary = f"Vehicles: {total_count} | Motion-Validated: {motion_validated_count} | Static: {total_count - motion_validated_count}"
        cv2.putText(
            result, summary,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        
        return result
    
    @staticmethod
    def draw_motion_overlay(frame: np.ndarray, motion_mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Overlay motion mask on frame.
        
        Args:
            frame: Input frame
            motion_mask: Binary motion mask from MOG2
            alpha: Transparency (0.0-1.0)
        
        Returns:
            Frame with motion overlay
        """
        result = frame.copy()
        
        # Create colored overlay (red for motion)
        motion_overlay = np.zeros_like(frame)
        motion_overlay[motion_mask > 0] = (0, 0, 255)  # Red
        
        # Blend
        result = cv2.addWeighted(result, 1.0, motion_overlay, alpha, 0)
        
        return result
    
    @staticmethod
    def draw_mog2_detections(frame: np.ndarray, mog2_boxes: List) -> np.ndarray:
        """
        Draw MOG2 motion detection boxes in RED.
        
        Args:
            frame: Input frame
            mog2_boxes: List of [x1, y1, x2, y2] bounding boxes from MOG2
        
        Returns:
            Frame with MOG2 boxes drawn
        """
        result = frame.copy()
        color = (0, 0, 255)  # RED for MOG2 motion detection
        
        for box in mog2_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                result, "MOG2",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1
            )
        
        # Draw summary
        summary = f"MOG2 Motion Detections: {len(mog2_boxes)}"
        cv2.putText(
            result, summary,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )
        
        return result


class HybridMetrics:
    """Track metrics for hybrid detection."""
    
    def __init__(self):
        self.total_detections = 0
        self.motion_validated = 0
        self.static_detections = 0
        self.confidence_scores = []
        
    def update(self, detections: List):
        """Update metrics from detections."""
        for x1, y1, x2, y2, conf, motion_validated in detections:
            self.total_detections += 1
            self.confidence_scores.append(conf)
            
            if motion_validated:
                self.motion_validated += 1
            else:
                self.static_detections += 1
    
    def print_summary(self):
        """Print metrics summary."""
        print("HYBRID DETECTION SUMMARY")
        print(f"Total Detections: {self.total_detections}")
        print(f"Motion-Validated: {self.motion_validated} ({100*self.motion_validated/max(1, self.total_detections):.1f}%)")
        print(f"Static Objects: {self.static_detections} ({100*self.static_detections/max(1, self.total_detections):.1f}%)")
        
        if self.confidence_scores:
            avg_conf = np.mean(self.confidence_scores)
            print(f"Average Confidence: {avg_conf:.2%}")

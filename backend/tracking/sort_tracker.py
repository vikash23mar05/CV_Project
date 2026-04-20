"""SORT multi-object tracking: Kalman filter + Hungarian matching for persistent IDs."""

import numpy as np
from scipy.optimize import linear_sum_assignment


# Coordinate conversion helpers

def _bbox_to_z(bbox):
    """Convert box [x1,y1,x2,y2] to measurement [cx,cy,scale,aspect_ratio]."""
    w  = bbox[2] - bbox[0]
    h  = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s  = w * h
    r  = w / float(h) if h > 0 else 1.0
    return np.array([cx, cy, s, r], dtype=float)


def _z_to_bbox(z):
    """Convert measurement [cx,cy,scale,ratio] back to box [x1,y1,x2,y2]."""
    # Recover w and h from area s and ratio r:  s = w*h,  r = w/h
    w  = np.sqrt(max(z[2] * z[3], 0.0))
    h  = z[2] / w if w > 0 else 0.0
    x1 = z[0] - w / 2.0
    y1 = z[1] - h / 2.0
    x2 = z[0] + w / 2.0
    y2 = z[1] + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=float)


def _iou(bb_a, bb_b):
    """Compute Intersection-over-Union between two boxes [x1,y1,x2,y2]."""
    ix1 = max(bb_a[0], bb_b[0])
    iy1 = max(bb_a[1], bb_b[1])
    ix2 = min(bb_a[2], bb_b[2])
    iy2 = min(bb_a[3], bb_b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h

    area_a = (bb_a[2] - bb_a[0]) * (bb_a[3] - bb_a[1])
    area_b = (bb_b[2] - bb_b[0]) * (bb_b[3] - bb_b[1])
    union   = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# Kalman filter tracker for single vehicle

class KalmanBoxTracker:
    """Single-vehicle Kalman tracker. State: [cx,cy,scale,ratio,vx,vy,vs]."""

    # Class-level counter so every new tracker gets a unique ID.
    _count = 0

    def __init__(self, bbox):
        """Initialize Kalman tracker from initial bbox [x1,y1,x2,y2]."""
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count

        # Kalman filter matrices
        n, m = 7, 4  # state dim=7, measurement dim=4

        # State transition F (constant-velocity model, dt=1 frame)
        #   cx' = cx + vx,  cy' = cy + vy,  s' = s + vs,  r' = r
        self.F = np.eye(n)
        self.F[0, 4] = 1   # cx += vx
        self.F[1, 5] = 1   # cy += vy
        self.F[2, 6] = 1   # s  += vs

        # Measurement matrix H (we observe cx, cy, s, r directly)
        self.H = np.zeros((m, n))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        # Process noise Q (motion model trust level)
        self.Q = np.diag([1., 1., 10., 1., 0.01, 0.01, 0.0001])

        # Measurement noise R (detector/sensor noise)
        self.R = np.diag([1., 1., 10., 1.])

        # Initial covariance P (high uncertainty in velocity at birth)
        self.P = np.diag([10., 10., 10., 10., 1e4, 1e4, 1e4])

        # Initial state: positions from bbox, velocities = 0
        z0 = _bbox_to_z(bbox)
        self.x = np.zeros((n, 1))
        self.x[:4, 0] = z0

        # Bookkeeping counters
        self.time_since_update = 0   # frames since last match
        self.hit_streak         = 0   # consecutive frames with match
        self.age                = 0   # total frames alive

    def predict(self):
        """Step forward 1 frame using motion model. Return predicted box."""
        # Clamp area so it never goes negative
        if self.x[2] < 0:
            self.x[2] = 0.0

        # x = F @ x,  P = F @ P @ F^T + Q
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age              += 1
        self.time_since_update += 1

        return _z_to_bbox(self.x[:4, 0])

    def update(self, bbox):
        """Correct state using matched detection [x1,y1,x2,y2]."""
        z = _bbox_to_z(bbox).reshape(4, 1)

        # Innovation y = z - H @ x
        y = z - self.H @ self.x

        # Innovation covariance S = H @ P @ H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain K = P @ H^T @ S^{-1}
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated state x = x + K @ y
        self.x = self.x + K @ y

        # Updated covariance P = (I - K @ H) @ P
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ self.H) @ self.P

        self.time_since_update  = 0
        self.hit_streak        += 1

    def get_bbox(self):
        """Return current estimated box [x1,y1,x2,y2]."""
        return _z_to_bbox(self.x[:4, 0])


# Association helpers (Hungarian matching)

def _hungarian_match(detections, predictions, iou_threshold=0.3):
    """Match detections ↔ predictions using IoU + Hungarian algorithm."""
    n_det  = len(detections)
    n_pred = len(predictions)

    if n_pred == 0:
        return [], list(range(n_det)), []

    if n_det == 0:
        return [], [], list(range(n_pred))

    # Build IoU matrix  (n_det × n_pred)
    iou_matrix = np.zeros((n_det, n_pred), dtype=float)
    for d, det in enumerate(detections):
        for p, pred in enumerate(predictions):
            iou_matrix[d, p] = _iou(det, pred)

    # Hungarian algorithm minimises cost → use 1 - IoU as cost
    row_idx, col_idx = linear_sum_assignment(1 - iou_matrix)

    matches, unmatched_dets, unmatched_preds = [], [], []

    matched_det_set  = set()
    matched_pred_set = set()

    for d, p in zip(row_idx, col_idx):
        if iou_matrix[d, p] >= iou_threshold:
            matches.append((d, p))
            matched_det_set.add(d)
            matched_pred_set.add(p)

    for d in range(n_det):
        if d not in matched_det_set:
            unmatched_dets.append(d)

    for p in range(n_pred):
        if p not in matched_pred_set:
            unmatched_preds.append(p)

    return matches, unmatched_dets, unmatched_preds


# Main SORT tracker (multi-object tracking)

class SortTracker:
    """Multi-object SORT: Kalman + Hungarian. Call update() each frame."""

    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections):
        """Run one SORT step. Input detections as [[x1,y1,x2,y2,score], ...].
        
        Returns: [(x1,y1,x2,y2,track_id), ...] for active confirmed tracks."""
        self.frame_count += 1

        # Extract [x1, y1, x2, y2], ignore score column if present
        det_boxes = np.array(
            [[d[0], d[1], d[2], d[3]] for d in detections],
            dtype=float
        ) if detections else np.empty((0, 4))

        # Predict new locations for all existing tracks
        pred_boxes = np.array(
            [t.predict() for t in self.trackers],
            dtype=float
        ) if self.trackers else np.empty((0, 4))

        # Match detections ↔ predictions using Hungarian algorithm
        matches, unmatched_dets, unmatched_preds = _hungarian_match(
            det_boxes, pred_boxes, self.iou_threshold
        )

        # Update matched tracks with their detection
        for d_idx, p_idx in matches:
            self.trackers[p_idx].update(det_boxes[d_idx])

        # Spawn new tracks for unmatched detections
        for d_idx in unmatched_dets:
            new_tracker = KalmanBoxTracker(det_boxes[d_idx])
            self.trackers.append(new_tracker)

        # Remove stale tracks & collect results
        active_tracks = []
        results       = []

        for tracker in self.trackers:
            if tracker.time_since_update > self.max_age:
                # Track went missing too long — discard it
                continue

            active_tracks.append(tracker)

            # Only report confirmed tracks (min_hits) or fresh ones
            if tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                x1, y1, x2, y2 = tracker.get_bbox()
                results.append((
                    int(x1), int(y1), int(x2), int(y2),
                    tracker.id
                ))

        self.trackers = active_tracks
        return results

"""Road analysis logic for traffic management."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class RoadAnalyzer:
    """
    Advanced road and traffic analysis system.
    
    Supports:
    - Configurable road division (left/right split)
    - Vehicle assignment based on centroid (no duplicate counting)
    - Per-road statistics (count, density, speed)
    - Lane analysis (5 lanes per road)
    - Anti-flickering buffer zone for smooth transitions
    - Production-ready visualization
    
    Data Structure:
    {
        "Road A": {
            "count": int,
            "density": "LIGHT|MEDIUM|HEAVY",
            "avg_speed": float,
            "vehicles": [(x1,y1,x2,y2,id), ...],
            "lanes": [{count, vehicles}, ...] (optional)
        },
        "Road B": {...},
        "total": int,
        "assignments": {vehicle_id: "Road A"|"Road B"}
    }
    """
    
    # Density levels
    DENSITY_LIGHT = "LIGHT"
    DENSITY_MEDIUM = "MEDIUM"
    DENSITY_HEAVY = "HEAVY"
    
    DENSITY_LIGHT_THRESHOLD = 5       # 0-5 vehicles
    DENSITY_MEDIUM_THRESHOLD = 10     # 6-10 vehicles
    # >10 is HEAVY
    
    # Color schemes
    COLOR_LIGHT = (0, 255, 0)      # GREEN
    COLOR_MEDIUM = (0, 165, 255)   # ORANGE
    COLOR_HEAVY = (0, 0, 255)      # RED
    
    def __init__(
        self, 
        frame_width: int = 1280, 
        frame_height: int = 720, 
        divider_pos: Optional[int] = None,
        buffer_zone: int = 0,
        enable_lanes: bool = True,
        num_lanes: int = 5
    ):
        """
        Initialize Road Analyzer.
        
        Args:
            frame_width: Video frame width (pixels)
            frame_height: Video frame height (pixels)
            divider_pos: X-coordinate of vertical divider (default: center)
            buffer_zone: Zone width (pixels) near divider to prevent flickering (default: 0)
            enable_lanes: Whether to track lanes per road
            num_lanes: Number of lanes per road
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.divider_pos = divider_pos if divider_pos is not None else frame_width // 2
        self.buffer_zone = buffer_zone
        self.enable_lanes = enable_lanes
        self.num_lanes = num_lanes
        
        # Vehicle history for anti-flickering (stores previous road assignment)
        self.vehicle_road_history: Dict[int, str] = {}
        
        # Initialize lane boundaries if lanes enabled
        if self.enable_lanes:
            self._init_lanes()
    
    # Lane initialization
    def _init_lanes(self):
        """Initialize 5-lane divisions for each road."""
        # Road A lanes (left side)
        left_width = self.divider_pos
        lane_width_a = left_width / self.num_lanes
        
        self.road_a_lanes = [
            (int(i * lane_width_a), int((i + 1) * lane_width_a))
            for i in range(self.num_lanes)
        ]
        
        # Road B lanes (right side)
        right_width = self.frame_width - self.divider_pos
        lane_width_b = right_width / self.num_lanes
        
        self.road_b_lanes = [
            (int(self.divider_pos + i * lane_width_b), 
             int(self.divider_pos + (i + 1) * lane_width_b))
            for i in range(self.num_lanes)
        ]
    
    # Core assignment logic
    def get_centroid(self, bbox: List) -> Tuple[float, float]:
        """
        Calculate bounding box centroid.
        
        Args:
            bbox: [x1, y1, x2, y2, ...] bounding box
        
        Returns:
            (center_x, center_y)
        """
        x1, y1, x2, y2 = bbox[:4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return cx, cy
    
    def get_road_for_centroid(self, cx: float, use_history: bool = False, 
                             vehicle_id: Optional[int] = None) -> str:
        """
        Determine road for a vehicle centroid with anti-flickering.
        
        Vehicles near divider use their previous assignment to prevent
        flickering. Once they move far enough from divider, they switch.
        
        Args:
            cx: Centroid X-coordinate
            use_history: Use previous assignment to prevent flickering near divider
            vehicle_id: Vehicle ID for history lookup
        
        Returns:
            "Road A" or "Road B"
        """
        # Check history first if enabled
        if use_history and vehicle_id is not None:
            if vehicle_id in self.vehicle_road_history:
                prev_road = self.vehicle_road_history[vehicle_id]
                # If in buffer zone, stick with previous assignment
                if abs(cx - self.divider_pos) <= self.buffer_zone:
                    return prev_road
        
        # Assign based on divider position
        road = "Road A" if cx < self.divider_pos else "Road B"
        
        # Update history
        if vehicle_id is not None:
            self.vehicle_road_history[vehicle_id] = road
        
        return road
    
    def get_lane_index(self, cx: float, road: str) -> int:
        """
        Get lane index (0-4) for a vehicle.
        
        Args:
            cx: Centroid X-coordinate
            road: "Road A" or "Road B"
        
        Returns:
            Lane index (0-4), or -1 if out of range
        """
        if not self.enable_lanes:
            return -1
        
        lanes = self.road_a_lanes if road == "Road A" else self.road_b_lanes
        
        for lane_idx, (lane_start, lane_end) in enumerate(lanes):
            if lane_start <= cx < lane_end:
                return lane_idx
        
        return -1  # Out of range
    
    # Statistics computation
    def calculate_density_category(self, vehicle_count: int) -> str:
        """
        Classify traffic density.
        
        Args:
            vehicle_count: Number of vehicles
        
        Returns:
            "LIGHT" (0-5), "MEDIUM" (6-10), or "HEAVY" (>10)
        """
        if vehicle_count <= self.DENSITY_LIGHT_THRESHOLD:
            return self.DENSITY_LIGHT
        elif vehicle_count <= self.DENSITY_MEDIUM_THRESHOLD:
            return self.DENSITY_MEDIUM
        else:
            return self.DENSITY_HEAVY
    
    def get_density_color(self, density: str) -> Tuple[int, int, int]:
        """Get BGR color for density level."""
        if density == self.DENSITY_LIGHT:
            return self.COLOR_LIGHT     # GREEN
        elif density == self.DENSITY_MEDIUM:
            return self.COLOR_MEDIUM    # ORANGE
        else:
            return self.COLOR_HEAVY     # RED
    
    def assign_vehicle_to_road(self, track: List, vehicle_id: Optional[int] = None) -> str:
        """
        Assign single vehicle to a road.
        
        Args:
            track: [x1, y1, x2, y2, vid] track data
            vehicle_id: Vehicle ID (from track[4] if not provided)
        
        Returns:
            "Road A" or "Road B"
        """
        if vehicle_id is None:
            vehicle_id = int(track[4]) if len(track) > 4 else None
        
        cx, _ = self.get_centroid(track)
        road = self.get_road_for_centroid(cx, use_history=True, vehicle_id=vehicle_id)
        
        return road
    
    def assign_vehicles_to_roads(self, tracks: List) -> Dict:
        """
        Assign all vehicles to roads with anti-flickering.
        
        Each vehicle is assigned to exactly ONE road based on its
        centroid X-coordinate. Vehicles near divider stick to their
        previous assignment to prevent flickering.
        
        Args:
            tracks: List of [x1, y1, x2, y2, vehicle_id] tracks
        
        Returns:
            {
                "Road A": [...vehicles],
                "Road B": [...vehicles],
                "assignments": {vehicle_id: road}
            }
        """
        road_a_vehicles = []
        road_b_vehicles = []
        assignments = {}
        
        for track in tracks:
            x1, y1, x2, y2 = track[:4]
            vehicle_id = int(track[4]) if len(track) > 4 else None
            
            cx, cy = self.get_centroid(track)
            road = self.get_road_for_centroid(cx, use_history=True, vehicle_id=vehicle_id)
            
            if vehicle_id is not None:
                assignments[vehicle_id] = road
            
            vehicle_data = (x1, y1, x2, y2, vehicle_id)
            
            if road == "Road A":
                road_a_vehicles.append(vehicle_data)
            else:
                road_b_vehicles.append(vehicle_data)
        
        return {
            "Road A": road_a_vehicles,
            "Road B": road_b_vehicles,
            "assignments": assignments,
        }

    
    def compute_road_stats(self, tracks: List, speeds_dict: Dict) -> Dict:
        """
        Compute comprehensive statistics for both roads.
        
        Computes per-road:
        - Vehicle count
        - Traffic density (LIGHT/MEDIUM/HEAVY)
        - Average speed
        - Lane-wise distribution (optional)
        
        Args:
            tracks: List of tracked vehicles
            speeds_dict: {vehicle_id: speed} mapping
        
        Returns:
            {
                "Road A": {
                    "count": int,
                    "density": str,
                    "avg_speed": float,
                    "vehicles": [...],
                    "lanes": [{...}, ...] if lanes enabled
                },
                "Road B": {...},
                "total": int,
                "assignments": {}
            }
        """
        assignments_data = self.assign_vehicles_to_roads(tracks)
        road_a_vehicles = assignments_data["Road A"]
        road_b_vehicles = assignments_data["Road B"]
        assignments = assignments_data["assignments"]
        
        # Compute Road A statistics
        road_a_count = len(road_a_vehicles)
        road_a_speeds = [
            speeds_dict.get(int(vid), 0) for _, _, _, _, vid in road_a_vehicles
            if vid is not None
        ]
        road_a_avg_speed = (
            sum(road_a_speeds) / len(road_a_speeds) 
            if road_a_speeds else 0
        )
        road_a_density = self.calculate_density_category(road_a_count)
        
        # Compute Road B statistics
        road_b_count = len(road_b_vehicles)
        road_b_speeds = [
            speeds_dict.get(int(vid), 0) for _, _, _, _, vid in road_b_vehicles
            if vid is not None
        ]
        road_b_avg_speed = (
            sum(road_b_speeds) / len(road_b_speeds)
            if road_b_speeds else 0
        )
        road_b_density = self.calculate_density_category(road_b_count)
        
        result = {
            "Road A": {
                "count": road_a_count,
                "density": road_a_density,
                "avg_speed": road_a_avg_speed,
                "vehicles": road_a_vehicles,
            },
            "Road B": {
                "count": road_b_count,
                "density": road_b_density,
                "avg_speed": road_b_avg_speed,
                "vehicles": road_b_vehicles,
            },
            "total": len(tracks),
            "assignments": assignments,
        }
        
        # Optionally compute lane-wise distribution
        if self.enable_lanes:
            result["Road A"]["lanes"] = self._compute_lane_distribution(
                road_a_vehicles, "Road A"
            )
            result["Road B"]["lanes"] = self._compute_lane_distribution(
                road_b_vehicles, "Road B"
            )
        
        return result
    
    def _compute_lane_distribution(self, vehicles: List, road: str) -> List[Dict]:
        """Compute per-lane vehicle distribution."""
        lanes = [{"index": i, "count": 0, "vehicles": []} for i in range(self.num_lanes)]
        
        for vehicle in vehicles:
            x1, y1, x2, y2, vid = vehicle
            cx, _ = self.get_centroid((x1, y1, x2, y2))
            lane_idx = self.get_lane_index(cx, road)
            
            if 0 <= lane_idx < self.num_lanes:
                lanes[lane_idx]["count"] += 1
                lanes[lane_idx]["vehicles"].append(vid)
        
        return lanes
    
    # Visualization
    def draw_road_divider(
        self, 
        frame: np.ndarray, 
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw vertical divider line.
        
        Args:
            frame: Input frame
            color: BGR color tuple
            thickness: Line thickness
        
        Returns:
            Modified frame
        """
        cv2.line(
            frame,
            (self.divider_pos, 0),
            (self.divider_pos, self.frame_height),
            color,
            thickness,
        )
        return frame
    
    def draw_road_labels(
        self,
        frame: np.ndarray,
        font_scale: float = 0.8,
        thickness: int = 2,
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        Draw "ROAD A" and "ROAD B" labels.
        
        Args:
            frame: Input frame
            font_scale: Font size
            thickness: Text thickness
            color: BGR color
        
        Returns:
            Modified frame
        """
        # Road A label (left side, centered horizontally)
        road_a_text = "ROAD A"
        text_size_a = cv2.getTextSize(
            road_a_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0]
        road_a_x = (self.divider_pos // 2) - (text_size_a[0] // 2)
        cv2.putText(
            frame,
            road_a_text,
            (max(road_a_x, 10), 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        
        # Road B label (right side, centered horizontally)
        road_b_text = "ROAD B"
        text_size_b = cv2.getTextSize(
            road_b_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0]
        road_b_x = (
            self.divider_pos + 
            (self.frame_width - self.divider_pos) // 2 - 
            (text_size_b[0] // 2)
        )
        cv2.putText(
            frame,
            road_b_text,
            (road_b_x, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        
        return frame
    
    def draw_road_statistics(
        self,
        frame: np.ndarray,
        road_stats: Dict,
        font_scale: float = 0.65,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw road statistics on frame.
        
        Colors are density-aware:
        - GREEN for LIGHT traffic
        - ORANGE for MEDIUM traffic
        - RED for HEAVY traffic
        
        Args:
            frame: Input frame
            road_stats: Result from compute_road_stats()
            font_scale: Font size
            thickness: Text thickness
        
        Returns:
            Modified frame with drawn statistics
        """
        # Left panel - Road A stats
        road_a_data = road_stats["Road A"]
        road_a_density_color = self.get_density_color(road_a_data["density"])
        
        y_offset = 80
        line_height = 28
        
        cv2.putText(
            frame, f"Vehicles: {road_a_data['count']}",
            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, road_a_density_color, thickness, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Density: {road_a_data['density']}",
            (20, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, road_a_density_color, thickness, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Avg Speed: {road_a_data['avg_speed']:.1f} px/fr",
            (20, y_offset + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, road_a_density_color, thickness, cv2.LINE_AA
        )
        
        # Right panel - Road B stats
        road_b_data = road_stats["Road B"]
        road_b_density_color = self.get_density_color(road_b_data["density"])
        
        road_b_x = self.frame_width - 280
        cv2.putText(
            frame, f"Vehicles: {road_b_data['count']}",
            (road_b_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, road_b_density_color, thickness, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Density: {road_b_data['density']}",
            (road_b_x, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, road_b_density_color, thickness, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Avg Speed: {road_b_data['avg_speed']:.1f} px/fr",
            (road_b_x, y_offset + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, road_b_density_color, thickness, cv2.LINE_AA
        )
        
        # Center top - Total vehicles
        total_text = f"TOTAL VEHICLES: {road_stats['total']}"
        text_size = cv2.getTextSize(
            total_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale + 0.1, thickness
        )[0]
        total_x = (self.frame_width - text_size[0]) // 2
        cv2.putText(
            frame, total_text,
            (total_x, 40), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale + 0.1, (200, 200, 200), thickness + 1, cv2.LINE_AA
        )
        
        return frame
    
    def draw_detected_vehicles(
        self,
        frame: np.ndarray,
        road_stats: Dict,
        speeds_dict: Dict,
        confidence_dict: Optional[Dict] = None,
        box_thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Draw vehicle boxes with IDs, speeds, and confidence.
        
        Boxes are colored by road density:
        - GREEN for LIGHT traffic roads
        - ORANGE for MEDIUM traffic roads
        - RED for HEAVY traffic roads
        
        Args:
            frame: Input frame
            road_stats: Result from compute_road_stats()
            speeds_dict: {vehicle_id: speed}
            confidence_dict: {box_key: confidence} optional
            box_thickness: Bounding box thickness
            font_scale: Text font size
        
        Returns:
            Modified frame with drawn vehicles
        """
        # Draw Road A vehicles
        road_a_color = self.get_density_color(road_stats["Road A"]["density"])
        for x1, y1, x2, y2, vid in road_stats["Road A"]["vehicles"]:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), road_a_color, box_thickness)
            
            # Draw info
            speed = speeds_dict.get(int(vid), 0) if vid else 0
            text = f"ID:{int(vid)} {speed:.1f}px/fr" if vid else "Unknown"
            cv2.putText(
                frame, text,
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, road_a_color, 1, cv2.LINE_AA
            )
        
        # Draw Road B vehicles
        road_b_color = self.get_density_color(road_stats["Road B"]["density"])
        for x1, y1, x2, y2, vid in road_stats["Road B"]["vehicles"]:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), road_b_color, box_thickness)
            
            # Draw info
            speed = speeds_dict.get(int(vid), 0) if vid else 0
            text = f"ID:{int(vid)} {speed:.1f}px/fr" if vid else "Unknown"
            cv2.putText(
                frame, text,
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, road_b_color, 1, cv2.LINE_AA
            )
        
        return frame
    
    # Backwards compatibility
    # These methods maintain compatibility with existing code
    
    def compute_road_statistics(self, tracks: List, speeds_dict: Dict) -> Dict:
        """Backwards compatibility wrapper for compute_road_stats()."""
        stats = self.compute_road_stats(tracks, speeds_dict)
        
        # Convert new field names to old ones for compatibility
        return {
            "Road A": {
                "vehicle_count": stats["Road A"]["count"],
                "density": stats["Road A"]["density"],
                "avg_speed": stats["Road A"]["avg_speed"],
                "vehicles": stats["Road A"]["vehicles"],
            },
            "Road B": {
                "vehicle_count": stats["Road B"]["count"],
                "density": stats["Road B"]["density"],
                "avg_speed": stats["Road B"]["avg_speed"],
                "vehicles": stats["Road B"]["vehicles"],
            },
            "total_vehicles": stats["total"],
            "assignments": stats["assignments"],
        }
    
    # Utilities
    def get_summary_stats(self, road_stats: Dict) -> str:
        """
        Get human-readable summary of road statistics.
        
        Args:
            road_stats: Result from compute_road_stats()
        
        Returns:
            Formatted string summary
        """
        a = road_stats["Road A"]
        b = road_stats["Road B"]
        total = road_stats.get("total", road_stats.get("total_vehicles", 0))
        
        summary = (
            f"ROAD ANALYSIS SUMMARY\n"
            f"Total Vehicles: {total}\n"
            f"\nROAD A (LEFT):\n"
            f"  Vehicles: {a['count'] if 'count' in a else a.get('vehicle_count', 0)}\n"
            f"  Density: {a['density']}\n"
            f"  Avg Speed: {a['avg_speed']:.2f} px/fr\n"
            f"\nROAD B (RIGHT):\n"
            f"  Vehicles: {b['count'] if 'count' in b else b.get('vehicle_count', 0)}\n"
            f"  Density: {b['density']}\n"
            f"  Avg Speed: {b['avg_speed']:.2f} px/fr\n"
        )
        
        return summary



#!/usr/bin/env python3
"""
MOG2-Only Motion Detection Application
Runs pure motion detection trained/calibrated by YOLO accuracy
"""

import cv2
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detection.yolo_detector import YOLODetector
from detection.mog2_detector import MOG2OnlyDetector, MOG2Analyzer

# Settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MOG2_MIN_AREA = 400


def main():
    parser = argparse.ArgumentParser(description="MOG2-Only Motion Detection")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate MOG2 using YOLO")
    parser.add_argument("--show-mask", action="store_true", help="Show motion mask overlay")
    parser.add_argument("--show-yolo", action="store_true", help="Show YOLO boxes for comparison")
    parser.add_argument("--loop", action="store_true", help="Loop video")
    
    args = parser.parse_args()
    
    video_path = args.video_path
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    print(f"Processing Video: {os.path.basename(video_path)}")
    print(f"Calibration: {args.calibrate} | Mask: {args.show_mask} | YOLO: {args.show_yolo}")
    print("Press 'q' to quit, 'p' to pause\n")
    
    # Initialize detectors
    mog2_detector = MOG2OnlyDetector(history=500, var_threshold=16)
    
    yolo_detector = None
    if args.calibrate or args.show_yolo:
        try:
            yolo_detector = YOLODetector(model_name="yolov8n.pt")
            print("[INFO] YOLO detector loaded for calibration/comparison")
        except Exception as e:
            print(f"[WARNING] Could not load YOLO: {e}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video")
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frame_count += 1
                
                # MOG2 detection
                mog2_boxes = mog2_detector.get_motion_boxes(frame)
                motion_mask = mog2_detector.get_foreground_mask(frame)
                
                # Compute congestion statistics for Road A and Road B
                road_stats = mog2_detector.compute_road_statistics(mog2_boxes)
                
                # YOLO detection (optional calibration)
                yolo_boxes = None
                if yolo_detector:
                    yolo_boxes = yolo_detector.detect(frame)
                    
                    if args.calibrate and frame_count % 10 == 0:
                        # Calibrate every 10 frames
                        mog2_detector.calibrate_from_yolo(frame, yolo_boxes, iou_threshold=0.2)
                
                # Draw frame with road statistics
                display_frame = MOG2Analyzer.draw_mog2_frame(
                    frame,
                    mog2_boxes,
                    road_stats=road_stats,
                    yolo_boxes=yolo_boxes if args.show_yolo else None,
                    show_mask=args.show_mask,
                    motion_mask=motion_mask if args.show_mask else None
                )
                
                # Add frame counter and calibration status
                status = "CALIBRATING" if (args.calibrate and not mog2_detector.calibrated) else "READY"
                cv2.putText(
                    display_frame,
                    f"Frame: {frame_count} | Status: {status}",
                    (10, FRAME_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1
                )
                
                # Display
                cv2.imshow("MOG2-Only Detection", display_frame)
            
            # Keyboard control
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")
            elif key == ord('s') and not paused:
                # Save frame
                filename = f"mog2_frame_{frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"[INFO] Saved {filename}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Frames processed: {frame_count}")
        print(f"MOG2 status: Calibrated={mog2_detector.calibrated}, Min Area={mog2_detector.min_area}")


if __name__ == "__main__":
    main()

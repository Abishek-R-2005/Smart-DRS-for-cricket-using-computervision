import cv2
import numpy as np
from ultralytics import YOLO
from utils import (
    detect_white_lines,
    filter_horizontal_lines,
    get_crease_line_y,
    get_bowler_feet,
    check_no_ball,
    draw_crease_line,
    draw_foot_markers,
    draw_no_ball_overlay,
    DecisionSmoother,
    CreaseTracker,
)

POSE_MODEL_PATH = "yolov8n-pose.pt"
POSE_CONFIDENCE = 0.25          # min confidence for pose detection
KEYPOINT_CONFIDENCE = 0.3       # min confidence for ankle keypoints
NO_BALL_TOLERANCE = 5           # pixels tolerance before calling no-ball
CREASE_MIN_LINE_LENGTH = 40     # min pixel length for a crease line
CREASE_ANGLE_THRESHOLD = 30     # max degrees from horizontal
DECISION_WINDOW = 9             # frames to average decision over
DECISION_THRESHOLD = 0.4        # % of window needed to confirm no-ball
CREASE_ROI = None  # Will auto-detect; set manually for better accuracy

# Load model globally to avoid reloading multiple times
print("[INFO] Loading YOLOv8 Pose model...")
pose_model = YOLO(POSE_MODEL_PATH)
print("[INFO] Model loaded successfully!")

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Use avc1 codec for better browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    crease_tracker = CreaseTracker(alpha=0.3)
    decision_smoother = DecisionSmoother(window_size=DECISION_WINDOW,
                                          threshold=DECISION_THRESHOLD)
    frame_count = 0
    no_ball_frames = 0
    legal_frames = 0
    detection_frames = 0
    bowler_region = (0, height // 3, width, height)  # x1, y1, x2, y2

    print("[INFO] Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()
        white_lines = detect_white_lines(frame, roi=CREASE_ROI)
        horizontal_lines = filter_horizontal_lines(
            white_lines,
            angle_thresh=CREASE_ANGLE_THRESHOLD,
            min_length=CREASE_MIN_LINE_LENGTH
        )

        crease_info = get_crease_line_y(horizontal_lines)
        crease_tracker.update(crease_info)
        stable_crease = crease_tracker.get_position()

        pose_results = pose_model(frame, conf=POSE_CONFIDENCE, verbose=False)
        feet_info = get_bowler_feet(
            pose_results, width, height,
            confidence_thresh=KEYPOINT_CONFIDENCE,
            bowler_region=bowler_region
        )

        crease_y = stable_crease[0] if stable_crease else None
        front_foot = feet_info.get('front_foot') if feet_info else None
        is_no_ball = False
        margin = 0

        if front_foot and crease_y is not None:
            is_no_ball, margin = check_no_ball(front_foot, crease_y, 
                                                tolerance=NO_BALL_TOLERANCE)
            detection_frames += 1
            conf = min(1.0, crease_tracker.confidence)
            decision_smoother.update(is_no_ball, confidence=conf)
        smoothed_decision, decision_confidence = decision_smoother.get_decision()

        if detection_frames > 0:
            if smoothed_decision:
                no_ball_frames += 1
            else:
                legal_frames += 1

        # Only show NO BALL or LEGAL DELIVERY text
        if smoothed_decision:
            cv2.putText(display, "NO BALL", (width // 2 - 120, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            cv2.putText(display, "LEGAL DELIVERY", (width // 2 - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        out.write(display)

        # Progress
        if frame_count % 30 == 0:
            pct = frame_count / max(total_frames, 1) * 100
            print(f"[PROGRESS] {pct:.1f}% ({frame_count}/{total_frames})")

    cap.release()
    out.release()

    print("\n" + "=" * 50)
    print("           NO BALL DETECTION SUMMARY")
    print("=" * 50)
    print(f"  Total frames processed : {frame_count}")
    print(f"  Frames with detection  : {detection_frames}")
    # ... more summary prints could go here if needed ...
    print(f"  Output saved to: {output_path}")
    print("=" * 50)

if __name__ == "__main__":
    # Test call
    process_video("no2.mp4", "output.mp4")
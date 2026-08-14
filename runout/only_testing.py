from ultralytics import YOLO
import cv2
import numpy as np

# ── Model ────────────────────────────────────────────────────────────────────
model = YOLO("best.pt")

# ── Video ────────────────────────────────────────────────────────────────────
video_path = "temp_video.mp4"
cap = cv2.VideoCapture(video_path)

# ── Decision state ───────────────────────────────────────────────────────────
decision        = None          # "OUT" | "NOT OUT" | "INCONCLUSIVE"
decision_color  = (0, 0, 255)   # BGR
decision_frame  = -1
banner_frames   = 0             # how many frames to keep banner visible


def draw_banner(frame, text, color):
    """Overlay a semi-transparent verdict banner at the bottom of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    banner_h = 120
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2.5
    thickness  = 4
    text_size  = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x     = (w - text_size[0]) // 2
    text_y     = h - banner_h + (banner_h + text_size[1]) // 2

    # Shadow
    cv2.putText(frame, text, (text_x + 3, text_y + 3),
                font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Main text
    cv2.putText(frame, text, (text_x, text_y),
                font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def determine_decision(detections):
    """
    Use YOLO detections at the bail-dislodge frame to decide OUT / NOT OUT.

    Logic (side-view camera assumption):
      - Check if the bat bounding box and pitch line bounding box overlap or cross over.
    """
    bat        = next((d for d in detections if d["class"] == "Bat"),        None)
    pitch_line = next((d for d in detections if d["class"] == "Pitch_Line"), None)

    if bat is None:
        return "OUT", (0, 0, 255)                   # bat not visible → OUT

    if pitch_line is None:
        return "INCONCLUSIVE", (0, 200, 255)        # can't find crease

    bat_x1, bat_y1, bat_x2, bat_y2 = bat["coords"]
    pitch_x1, pitch_y1, pitch_x2, pitch_y2 = pitch_line["coords"]

    # Check for horizontal intersection or if the bat is well within the crease
    if (bat_x2 > pitch_x1 and bat_x1 < pitch_x2) or (bat_y2 > pitch_y1 and bat_y1 < pitch_y2):
        return "NOT OUT", (0, 220, 0)
    else:
        return "OUT", (0, 0, 255)


# ── Main loop ────────────────────────────────────────────────────────────────
BANNER_HOLD = 120   # keep verdict on screen for ~4 s at 30 fps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    # ── Parse detections ──────────────────────────────────────────────────
    detections = []
    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            name = model.names[cls]
            coords = box.xyxy[0].tolist()
            conf   = float(box.conf[0])
            detections.append({"class": name, "coords": coords, "conf": conf})

    # ── Annotate frame ────────────────────────────────────────────────────
    annotated = results[0].plot()

    # ── Detect bail dislodgement & latch decision ─────────────────────────
    if decision is None:
        for d in detections:
            if d["class"] == "Dis_Wicket":
                decision, decision_color = determine_decision(detections)
                banner_frames = BANNER_HOLD
                print(f"[DECISION] {decision} (frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))})")
                break

    # ── Draw verdict banner if active ─────────────────────────────────────
    if decision is not None and banner_frames > 0:
        annotated = draw_banner(annotated, decision, decision_color)
        banner_frames -= 1
    elif decision is not None:
        # Keep a small persistent label in the corner after the banner fades
        cv2.putText(annotated, f"VERDICT: {decision}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    decision_color, 2, cv2.LINE_AA)

    cv2.imshow("YOLOv8 Run-Out Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if decision:
    print(f"\n{'='*40}\nFINAL DECISION: {decision}\n{'='*40}")
else:
    print("\n bail dislodgement detected in the video.")
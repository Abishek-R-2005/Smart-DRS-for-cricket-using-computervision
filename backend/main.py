"""
FastAPI Backend for LBW Cricket Analysis System
"""

import os
import cv2
import uuid
import shutil
import tempfile
import numpy as np
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from roboflow import Roboflow
import supervision as sv

app = FastAPI(title="LBW Analysis System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- Storage ---------------
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
FRAMES_DIR = Path("frames")

for d in [UPLOAD_DIR, PROCESSED_DIR, FRAMES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Mount static dirs
app.mount("/processed", StaticFiles(directory=str(PROCESSED_DIR)), name="processed")
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")

# Serve vid folder (check both root/vid and frontend/vid just in case)
vid_dir = Path("vid") if Path("vid").exists() else (Path(__file__).parent.parent / "vid")
app.mount("/vid", StaticFiles(directory=str(vid_dir)), name="vid")

# --------------- Roboflow Model ---------------
rf = Roboflow(api_key="YOUR_API_KEY")
stump_model = rf.workspace().project("stump-line-detector").version(1).model

# --------------- Session Store ---------------
sessions = {}


# --------------- Helpers ---------------
def roboflow_to_sv(result):
    boxes, confidences, class_ids = [], [], []
    for item in result["predictions"]:
        x, y, w, h = item["x"], item["y"], item["width"], item["height"]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes.append([x1, y1, x2, y2])
        confidences.append(item["confidence"])
        class_ids.append(0)
    if len(boxes) == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes),
        confidence=np.array(confidences),
        class_id=np.array(class_ids),
    )


def detect_stump(image_path):
    image = cv2.imread(image_path)
    result = stump_model.predict(image, confidence=50, overlap=50).json()
    detections = roboflow_to_sv(result)
    box_annotator = sv.BoxAnnotator()
    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    return annotated, detections


def lbw_decision(pitch, impact, pred, stump_x1, stump_x2):
    pitch_in_line = stump_x1 <= pitch[0] <= stump_x2
    impact_in_line = stump_x1 <= impact[0] <= stump_x2
    hitting = stump_x1 <= pred[0] <= stump_x2

    score = 0
    score += 0.3 if pitch_in_line else 0
    score += 0.4 if impact_in_line else 0
    score += 0.7 if hitting else 0

    decision = "OUT" if score >= 0.8 else "NOT OUT"
    return decision, pitch_in_line, impact_in_line, hitting


# --------------- API Routes ---------------

@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    """Upload a video file, extract frames with overlays, and return processed video."""
    session_id = str(uuid.uuid4())
    session_dir = FRAMES_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    video_path = UPLOAD_DIR / f"{session_id}_{video.filename}"
    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video file")

    frame_count = 0
    saved_frames = {}
    stump_x1, stump_x2 = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        cv2.putText(
            frame, f"Frame: {frame_count}", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
        )

        # Stump region overlay
        ref_w, ref_h = 1920, 1080
        x1_ref, x2_ref = 921, 1000
        y1_ref, y2_ref = 399, 1080

        x1 = int((x1_ref / ref_w) * width)
        x2 = int((x2_ref / ref_w) * width)
        y1 = int((y1_ref / ref_h) * height)
        y2 = int((y2_ref / ref_h) * height)

        stump_x1, stump_x2 = x1, x2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        frame_path = str(session_dir / f"{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_frames[frame_count] = frame_path
        frame_count += 1

    cap.release()

    if frame_count == 0:
        raise HTTPException(status_code=400, detail="No frames extracted from video")

    # Create output video
    output_video_name = f"{session_id}_output.mp4"
    output_video_path = PROCESSED_DIR / output_video_name

    # Use ffmpeg to create a web-compatible H.264 mp4 video
    input_pattern = str(session_dir / "%d.jpg")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "20",
            "-i", input_pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(output_video_path)
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg error:", e.stderr.decode('utf-8'))
        raise HTTPException(status_code=500, detail="Video encoding failed")

    # Store session data
    sessions[session_id] = {
        "frame_count": frame_count,
        "saved_frames": saved_frames,
        "stump_x1": stump_x1,
        "stump_x2": stump_x2,
        "video_url": f"/processed/{output_video_name}",
    }

    return JSONResponse({
        "session_id": session_id,
        "frame_count": frame_count,
        "video_url": f"/processed/{output_video_name}",
    })


@app.get("/api/frame/{session_id}/{frame_number}")
async def get_frame(session_id: str, frame_number: int):
    """Get a specific processed frame with stump detection."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    if frame_number < 0 or frame_number >= session["frame_count"]:
        raise HTTPException(status_code=400, detail="Frame number out of range")

    frame_path = session["saved_frames"][frame_number]

    # Run stump detection
    annotated, detections = detect_stump(frame_path)

    # Save annotated frame
    annotated_path = str(FRAMES_DIR / session_id / f"annotated_{frame_number}.jpg")
    cv2.imwrite(annotated_path, annotated)

    return JSONResponse({
        "frame_url": f"/frames/{session_id}/annotated_{frame_number}.jpg",
        "stump_x1": int(session["stump_x1"]),
        "stump_x2": int(session["stump_x2"]),
    })


@app.post("/api/trajectory/{session_id}")
async def compute_trajectory(session_id: str, data: dict):
    """Compute ball trajectory and LBW decision."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    pitch_x, pitch_y = data["pitch_x"], data["pitch_y"]
    impact_x, impact_y = data["impact_x"], data["impact_y"]
    impact_frame_num = data["impact_frame_number"]

    dx = impact_x - pitch_x
    dy = impact_y - pitch_y
    length = np.sqrt(dx ** 2 + dy ** 2)
    unit_dx, unit_dy = (dx / length, dy / length) if length != 0 else (0, 1)

    pred_x = int(impact_x + unit_dx * 80)
    pred_y = int(impact_y + unit_dy * 80)

    # Draw trajectory on impact frame
    frame_path = session["saved_frames"][impact_frame_num]
    image = cv2.imread(frame_path)

    # Line from pitch to impact (yellow)
    cv2.line(image, (pitch_x, pitch_y), (impact_x, impact_y), (0, 255, 255), 3)
    # Predicted line (red)
    cv2.line(image, (impact_x, impact_y), (pred_x, pred_y), (0, 0, 255), 3)

    # Dots
    cv2.circle(image, (pitch_x, pitch_y), 8, (255, 0, 0), -1)        # Blue = pitch
    cv2.circle(image, (impact_x, impact_y), 8, (0, 255, 0), -1)      # Green = impact
    cv2.circle(image, (pred_x, pred_y), 8, (0, 0, 255), -1)          # Red = predicted

    # LBW Decision
    decision, pitch_ok, impact_ok, hitting = lbw_decision(
        (pitch_x, pitch_y),
        (impact_x, impact_y),
        (pred_x, pred_y),
        session["stump_x1"],
        session["stump_x2"],
    )

    traj_path = str(FRAMES_DIR / session_id / "trajectory.jpg")
    cv2.imwrite(traj_path, image)

    return JSONResponse({
        "trajectory_url": f"/frames/{session_id}/trajectory.jpg",
        "decision": decision,
        "pitch_in_line": pitch_ok,
        "impact_in_line": impact_ok,
        "hitting_stumps": hitting,
        "predicted_point": {"x": pred_x, "y": pred_y},
    })


# --------------- Serve Frontend ---------------
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.get("/")
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/styles.css")
async def serve_css():
    return FileResponse(str(FRONTEND_DIR / "styles.css"))

@app.get("/app.js")
async def serve_js():
    return FileResponse(str(FRONTEND_DIR / "app.js"))

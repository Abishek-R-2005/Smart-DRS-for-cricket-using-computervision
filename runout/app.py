import os
import threading
import time
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from ultralytics import YOLO

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Global state ─────────────────────────────────────────────────────────────
model = YOLO("best.pt")

state = {
    "frame": None,
    "decision": None,
    "decision_color": None,
    "processing": False,
    "progress": 0,
    "total_frames": 0,
    "banner_frames_left": 0,
    "video_path": None,
    "done": False,
    "error": None,
}
state_lock = threading.Lock()

BANNER_HOLD = 120  # ~4 s at 30 fps


# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_banner(frame, text, color):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    banner_h = 110
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    font = cv2.FONT_HERSHEY_DUPLEX
    fs = 2.2
    th = 4
    tw, _ = cv2.getTextSize(text, font, fs, th)[0], 0
    tx = (w - tw) // 2
    ty = h - banner_h + (banner_h + 30) // 2
    cv2.putText(frame, text, (tx + 3, ty + 3), font, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), font, fs, color, th, cv2.LINE_AA)
    return frame



def process_video(video_path):
    with state_lock:
        state["processing"] = True
        state["decision"] = None
        state["progress"] = 0
        state["done"] = False
        state["error"] = None
        state["banner_frames_left"] = 0

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with state_lock:
        state["total_frames"] = max(total, 1)

    decision_latched = False
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({"class": name, "coords": coords, "conf": conf})

        annotated = results[0].plot()

        with state_lock:
            if not decision_latched:
                for d in detections:
                    if d["class"] == "Dis_Wicket":
                        state["decision"] = "MANUAL_REQUIRED"
                        state["decision_color"] = (0, 180, 255)
                        decision_latched = True
                        break

            if state["banner_frames_left"] > 0:
                annotated = draw_banner(annotated, state["decision"], state["decision_color"])
                state["banner_frames_left"] -= 1
            elif state["decision"] and state["decision"] not in ["MANUAL_REQUIRED", "NO RUNOUT"]:
                col = state["decision_color"]
                cv2.putText(annotated, f"VERDICT: {state['decision']}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            state["frame"] = buf.tobytes()
            state["progress"] = frame_idx + 1

        frame_idx += 1

    cap.release()
    with state_lock:
        state["processing"] = False
        state["done"] = True


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/set_decision", methods=["POST"])
def set_decision():
    dec = request.form.get("decision", "OUT")
    with state_lock:
        state["decision"] = dec
        state["decision_color"] = (0, 50, 220) if dec == "OUT" else (0, 200, 60)
        state["banner_frames_left"] = BANNER_HOLD
    return jsonify({"status": "success"})


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".mp4"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"input_video{ext}")
    file.save(save_path)

    # Reset and start processing thread
    with state_lock:
        state["frame"]  = None
        state["done"]   = False
        state["decision"] = None

    t = threading.Thread(target=process_video, args=(save_path,), daemon=True)
    t.start()

    return jsonify({"status": "started"})


def generate_frames():
    last_prog = -1
    while True:
        with state_lock:
            raw = state["frame"]
            done = state["done"]
            prog = state["progress"]

        if raw and prog != last_prog:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + raw + b"\r\n")
            last_prog = prog
        elif done:
            break

        time.sleep(0.01)


@app.route("/video_feed")
def video_feed():
    return Response(stream_with_context(generate_frames()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        total = state["total_frames"]
        prog  = state["progress"]
        pct   = int((prog / total) * 100) if total else 0
        return jsonify({
            "processing":  state["processing"],
            "done":        state["done"],
            "decision":    state["decision"],
            "progress":    pct,
            "error":       state["error"],
        })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000, threaded=True)

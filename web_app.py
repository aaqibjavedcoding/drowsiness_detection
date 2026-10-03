"""
web_app.py
-----------
Browser version of the drowsiness-detection dashboard.

Why this exists
===============
`main.py` opens a Tkinter desktop window and grabs the webcam with
`cv2.VideoCapture(0)`. That only works on a real desktop machine that has a
display server **and** a physically attached camera. Inside a remote
sandbox / preview (or on a headless server, or over SSH) there is no display
and no camera device, so the window either never appears or appears frozen on
the "Press START to begin monitoring" placeholder with no working buttons.

This module keeps *exactly the same detection pipeline* (MediaPipe Face Mesh ->
EAR / MAR / head pose -> DrowsinessEngine) but moves the UI into the browser:

    browser webcam (getUserMedia)  ->  JPEG frame  ->  Flask  ->  pipeline
                                   <-  JSON state  <-

So the camera comes from the machine running the *browser*, and the buttons are
real HTML buttons that always work.

Run with:
    python web_app.py            # then open http://localhost:5000
"""

from __future__ import annotations

import base64
import os
import threading
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import config
from detection.drowsiness_engine import DrowsinessEngine
from detection.face_detector import FaceDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "web", "templates"),
    static_folder=os.path.join(BASE_DIR, "web", "static"),
)

# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
SESSION_TTL_SECONDS = 120.0
_sessions: dict[str, "Session"] = {}
_sessions_lock = threading.Lock()


class Session:
    """One browser tab == one detector + one scoring engine."""

    def __init__(self):
        self.id = uuid.uuid4().hex
        self.detector = FaceDetector()
        self.engine = DrowsinessEngine()
        self.lock = threading.Lock()
        self.last_seen = time.time()

    def close(self):
        try:
            self.detector.close()
        except Exception:
            pass


def _get_session(session_id: str | None, create: bool = False) -> "Session | None":
    now = time.time()
    with _sessions_lock:
        # Drop sessions from tabs that were closed without saying goodbye.
        for stale_id in [
            sid for sid, s in _sessions.items() if now - s.last_seen > SESSION_TTL_SECONDS
        ]:
            _sessions.pop(stale_id).close()

        session = _sessions.get(session_id) if session_id else None
        if session is None and create:
            session = Session()
            _sessions[session.id] = session
        if session is not None:
            session.last_seen = now
        return session


# ---------------------------------------------------------------------------
# Face-mesh drawing data (sent once, the browser draws the overlay itself)
# ---------------------------------------------------------------------------
def _mesh_connections():
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh
    return {
        "contours": sorted({tuple(sorted(c)) for c in face_mesh.FACEMESH_CONTOURS}),
        "left_eye": sorted({tuple(sorted(c)) for c in face_mesh.FACEMESH_LEFT_EYE}),
        "right_eye": sorted({tuple(sorted(c)) for c in face_mesh.FACEMESH_RIGHT_EYE}),
        "lips": sorted({tuple(sorted(c)) for c in face_mesh.FACEMESH_LIPS}),
    }


_MESH_CACHE: dict | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.after_request
def _no_store(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/")
def index():
    return render_template(
        "index.html",
        thresholds={
            "warning": config.WARNING_SCORE_THRESHOLD,
            "drowsy": config.DROWSY_SCORE_THRESHOLD,
            "ear": config.EAR_THRESHOLD,
            "mar": config.MAR_THRESHOLD,
            "long_closure": config.LONG_CLOSURE_SECONDS,
        },
    )


@app.route("/api/mesh")
def api_mesh():
    global _MESH_CACHE
    if _MESH_CACHE is None:
        _MESH_CACHE = _mesh_connections()
    return jsonify(_MESH_CACHE)


@app.route("/api/session/start", methods=["POST"])
def api_session_start():
    session = _get_session(None, create=True)
    return jsonify({"session_id": session.id, "state": session.engine.state})


@app.route("/api/session/reset", methods=["POST"])
def api_session_reset():
    payload = request.get_json(silent=True) or {}
    session = _get_session(payload.get("session_id"))
    if session is None:
        return jsonify({"error": "unknown session"}), 404
    with session.lock:
        session.engine.reset_session()
        stats = session.engine.get_stats()
    return jsonify({"ok": True, "stats": stats})


@app.route("/api/session/stop", methods=["POST"])
def api_session_stop():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session is not None:
        session.close()
    return jsonify({"ok": True})


@app.route("/api/frame", methods=["POST"])
def api_frame():
    payload = request.get_json(silent=True) or {}
    session = _get_session(payload.get("session_id"))
    if session is None:
        return jsonify({"error": "unknown session", "code": "no_session"}), 404

    frame = _decode_image(payload.get("image", ""))
    if frame is None:
        return jsonify({"error": "could not decode frame", "code": "bad_frame"}), 400

    height, width = frame.shape[:2]

    with session.lock:
        try:
            landmarks, mp_results = session.detector.process(frame)
            result = session.engine.update(landmarks, frame.shape)
        except Exception as exc:  # never let one bad frame kill the session
            return jsonify({"error": f"processing error: {exc}", "code": "process"}), 500

    points = []
    if mp_results is not None and mp_results.multi_face_landmarks:
        points = [
            [round(float(lm.x), 4), round(float(lm.y), 4)]
            for lm in mp_results.multi_face_landmarks[0].landmark
        ]

    eye = result["eye"] or {}
    mouth = result["mouth"] or {}
    head = result["head"] or {}

    return jsonify(
        {
            "face_found": result["face_found"],
            "no_face_timeout": result["no_face_timeout"],
            "score": round(float(result["score"]), 1),
            "state": result["state"],
            "eye": {
                "ear": round(float(eye.get("ear", 0.0)), 3),
                "closed": bool(eye.get("eye_closed", False)),
                "long_closure": bool(eye.get("is_long_closure", False)),
                "closure_duration": round(float(eye.get("closure_duration", 0.0)), 2),
            },
            "mouth": {
                "mar": round(float(mouth.get("mar", 0.0)), 3),
                "open": bool(mouth.get("mouth_open", False)),
                "yawning": bool(mouth.get("is_yawning", False)),
            },
            "head": {
                "label": head.get("label", "--"),
                "pitch": round(float(head.get("pitch", 0.0)), 1),
                "yaw": round(float(head.get("yaw", 0.0)), 1),
                "down": bool(head.get("looking_down", False)),
                "away": bool(head.get("looking_away", False)),
            },
            "stats": {k: round(float(v), 2) for k, v in result["stats"].items()},
            "landmarks": points,
            "frame": {"width": width, "height": height},
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_image(data_url: str):
    """Decode a `data:image/jpeg;base64,...` string into a BGR numpy frame."""
    if not data_url:
        return None
    try:
        encoded = data_url.split(",", 1)[-1]
        raw = base64.b64decode(encoded)
        buffer = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:
        return None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f" * Drowsiness dashboard -> http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

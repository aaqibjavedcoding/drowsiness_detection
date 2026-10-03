"""
config.py
----------
Central place for every tunable constant used by the drowsiness detection
application. Keeping thresholds here (instead of hard-coding them inside the
detection modules) makes the whole project easy to calibrate for a different
camera, lighting condition or user without touching the detection logic.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
ALARM_SOUND_PATH = os.path.join(ASSETS_DIR, "alarm.wav")

# ---------------------------------------------------------------------------
# Camera / performance
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
UI_REFRESH_MS = 30          # ~33 FPS target for the Tkinter "after" loop
CAMERA_OPEN_RETRIES = 1     # extra attempts if the first VideoCapture open fails

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh
# ---------------------------------------------------------------------------
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
REFINE_LANDMARKS = True

# ---------------------------------------------------------------------------
# Eye landmark indices (MediaPipe FaceMesh 468/478 point topology)
# Order per eye: [outer corner, top-1, top-2, inner corner, bottom-1, bottom-2]
# This is the standard 6-point layout used for the classic EAR formula.
# ---------------------------------------------------------------------------
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# ---------------------------------------------------------------------------
# Mouth landmark indices (inner lip points) used for the Mouth Aspect Ratio
# ---------------------------------------------------------------------------
MOUTH_VERTICAL_PAIR = (13, 14)      # inner-upper-lip, inner-lower-lip
MOUTH_HORIZONTAL_PAIR = (78, 308)   # left corner, right corner

# ---------------------------------------------------------------------------
# Head pose landmark indices + 3D model points (generic face model, in mm)
# ---------------------------------------------------------------------------
HEAD_POSE_LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_left_corner": 33,
    "right_eye_right_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}

HEAD_PITCH_DOWN_THRESHOLD_DEG = 14.0   # chin-down (nodding off) angle
HEAD_YAW_THRESHOLD_DEG = 28.0          # looking away from the road/screen

# ---------------------------------------------------------------------------
# Eye closure / blink thresholds
# ---------------------------------------------------------------------------
EAR_THRESHOLD = 0.21            # below this -> eye considered closed
EAR_SMOOTHING_WINDOW = 5        # frames used for a rolling average of EAR
EAR_CONSEC_FRAMES_BLINK = 2     # min consecutive closed frames to count a blink
LONG_CLOSURE_SECONDS = 1.5      # sustained closure longer than this = microsleep

# ---------------------------------------------------------------------------
# Mouth / yawn thresholds
# ---------------------------------------------------------------------------
MAR_THRESHOLD = 0.55            # above this -> mouth considered open
YAWN_CONSEC_FRAMES = 12         # ~0.4s @ 30FPS of sustained opening = a yawn

# ---------------------------------------------------------------------------
# Drowsiness scoring model (0-100)
# Each condition adds a fixed number of points per processed frame while it
# is active; the score decays back down automatically when the driver looks
# fine again. Tweak these weights to make the system stricter or more lenient.
# ---------------------------------------------------------------------------
SCORE_RISE_EYE_CLOSED = 1.4
SCORE_RISE_LONG_CLOSURE = 3.2
SCORE_RISE_YAWN = 1.0
SCORE_RISE_HEAD_DOWN = 1.1
SCORE_RISE_HEAD_AWAY = 0.6
SCORE_DECAY_PER_FRAME = 1.6

SCORE_MIN = 0.0
SCORE_MAX = 100.0

WARNING_SCORE_THRESHOLD = 40.0
DROWSY_SCORE_THRESHOLD = 70.0

# Frames without any detected face before we report an error state
NO_FACE_GRACE_FRAMES = 15

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
STATE_NORMAL = "NORMAL"
STATE_WARNING = "WARNING"
STATE_DROWSY = "DROWSY"

STATE_COLORS = {
    STATE_NORMAL: "#2ecc71",   # green
    STATE_WARNING: "#f39c12",  # amber
    STATE_DROWSY: "#e74c3c",   # red
}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
WINDOW_TITLE = "Drowsiness Detection Dashboard"
WINDOW_BG = "#1e1e2e"
PANEL_BG = "#262637"
TEXT_COLOR = "#f0f0f5"
ACCENT_COLOR = "#4ea1ff"

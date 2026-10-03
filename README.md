# Drowsiness Detection Using Facial Features

A real-time, fully offline driver/student-drowsiness detection system built
for a college computer-vision project. It uses a laptop webcam, MediaPipe
Face Mesh landmarks, and classic computer-vision metrics (EAR, MAR, head
pose) to compute a live **Drowsiness Score (0–100)** and raise a visual +
audible alarm when the user appears drowsy — all inside a Tkinter desktop
dashboard.

No database, no backend server, no cloud services and no paid APIs are used.
Everything runs locally on your machine.

## Features

- Real-time webcam face detection with **MediaPipe Face Mesh** (468/478 landmarks).
- **Eye Aspect Ratio (EAR)** based eye-closure detection.
- Blink counting and long-closure ("microsleep") detection.
- **Mouth Aspect Ratio (MAR)** based yawning detection.
- Basic **head-pose estimation** (pitch/yaw) using `solvePnP`, flags "head down" / "looking away".
- A combined, configurable **0–100 Drowsiness Score**.
- Three states: `NORMAL`, `WARNING`, `DROWSY`, each with a distinct colour.
- Looping **audio alarm** (via `pygame`) that starts automatically in the `DROWSY` state.
- Live facial landmark overlay and on-screen EAR/MAR/head-pose/score readouts.
- Session statistics panel: blink count, long-blink count, yawn count, warning
  count, drowsiness count, max eye-closure duration, current EAR, current score.
- Graceful error handling: missing webcam, lost camera connection, no face in
  frame, or missing audio device never crash the application.
- **Start / Stop / Reset Session / Exit** controls in the dashboard.

## Tech Stack

- Python 3.9 – 3.12
- OpenCV (`opencv-contrib-python`)
- MediaPipe (Face Mesh solution)
- NumPy
- Tkinter (standard library GUI)
- Pillow (used only to bridge OpenCV frames into Tkinter widgets)
- pygame (alarm playback)

## Folder Structure

```text
drowsiness_detection/
├── main.py                     # Application entry point
├── config.py                   # All tunable thresholds and constants
├── requirements.txt
├── README.md
├── detection/
│   ├── __init__.py
│   ├── face_detector.py        # MediaPipe Face Mesh wrapper
│   ├── eye_analyzer.py         # EAR + blink/long-closure logic
│   ├── mouth_analyzer.py       # MAR + yawn detection
│   ├── head_pose.py            # solvePnP-based head pose estimation
│   └── drowsiness_engine.py    # Combines signals into score/state + stats
├── ui/
│   ├── __init__.py
│   └── dashboard.py            # Tkinter dashboard (video + stats + controls)
├── utils/
│   ├── __init__.py
│   ├── alarm.py                # pygame alarm sound manager
│   └── calculations.py         # EAR/MAR math helpers
└── assets/
    └── alarm.wav                # Generated alarm tone
```

## Installation

1. Make sure you have **Python 3.9–3.12** installed and a working webcam.
2. (Recommended) create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

From the project root:

```bash
python main.py
```

A dashboard window opens. Click **Start** to begin monitoring. The left
panel shows your live webcam feed with the detected face mesh and live
EAR/MAR/score overlay. The right panel shows the current state, drowsiness
score, detailed readouts and session statistics.

- **Start** – opens the webcam and begins real-time analysis.
- **Stop** – releases the webcam and pauses analysis (statistics are kept).
- **Reset Session** – clears all counters/statistics and resets the score.
- **Exit** – safely releases the camera/audio and closes the application.

When the system classifies you as `DROWSY` (either the score crosses the
drowsy threshold or your eyes stay closed for longer than
`LONG_CLOSURE_SECONDS`), a looping alarm sound plays automatically until you
are alert again or you press Stop/Exit.

## How It Works

1. **Face detection** (`detection/face_detector.py`): Each webcam frame is
   passed to MediaPipe's Face Mesh solution, returning 468–478 facial
   landmarks in pixel coordinates.
2. **Eye analysis** (`detection/eye_analyzer.py`): The Eye Aspect Ratio is
   computed for both eyes using 6 landmarks per eye
   (`EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)`). A rolling average smooths
   noise; sustained low EAR triggers blink/long-closure/microsleep counters.
3. **Mouth analysis** (`detection/mouth_analyzer.py`): The Mouth Aspect
   Ratio compares vertical mouth opening to mouth width. A sustained high
   MAR is classified as a yawn.
4. **Head pose** (`detection/head_pose.py`): A generic 3D face model is
   matched against 6 landmarks (nose tip, chin, eye corners, mouth corners)
   using `cv2.solvePnP` to approximate pitch/yaw. Sustained "head down" or
   "looking away" posture contributes to the score.
5. **Scoring engine** (`detection/drowsiness_engine.py`): Every frame, each
   active condition (eyes closed, long closure, yawning, head down/away)
   adds configurable points to a 0–100 score; the score decays automatically
   when the user looks alert. Crossing `WARNING_SCORE_THRESHOLD` /
   `DROWSY_SCORE_THRESHOLD` changes the state. A sustained eye closure longer
   than `LONG_CLOSURE_SECONDS` forces an immediate `DROWSY` state as a safety
   measure (independent of the score), modelling a classic PERCLOS/microsleep
   safety trigger.
6. **Dashboard** (`ui/dashboard.py`): A Tkinter `after()` loop reads a frame,
   runs the pipeline above, draws the landmarks/overlay, and updates the
   live readouts, score bar, state banner and statistics panel.

## Tuning

All thresholds (EAR/MAR cut-offs, score weights, state thresholds, head-pose
angles, camera index/resolution, etc.) live in **`config.py`** with comments
explaining each one. Typical adjustments:

- If blinks are being missed or over-counted, adjust `EAR_THRESHOLD` and
  `EAR_CONSEC_FRAMES_BLINK`.
- If yawns are detected too easily/rarely, adjust `MAR_THRESHOLD` and
  `YAWN_CONSEC_FRAMES`.
- To make the alarm trigger sooner/later, adjust `DROWSY_SCORE_THRESHOLD`,
  `LONG_CLOSURE_SECONDS`, or the `SCORE_RISE_*` / `SCORE_DECAY_PER_FRAME`
  weights.
- If you have multiple cameras, change `CAMERA_INDEX`.

## Troubleshooting

- **Webcam not detected / "Could not access the webcam"**: close any other
  app using the camera, check OS camera permissions, or try changing
  `CAMERA_INDEX` in `config.py` (0, 1, 2, ...).
- **No sound from the alarm**: confirm your system volume/output device is
  working; `utils/alarm.py` will print a warning and simply disable the
  alarm (without crashing) if no audio device is available.
- **`ImportError: libGL.so.1` on Linux**: install the system Mesa/OpenGL
  libraries, e.g. `sudo apt-get install -y libgl1`.
- **Low FPS / laggy video**: lower `FRAME_WIDTH`/`FRAME_HEIGHT` in
  `config.py`, or increase `UI_REFRESH_MS` slightly.
- **"No face detected"**: make sure your face is well lit and fully inside
  the frame; the status bar and on-screen overlay will tell you when no
  face is found instead of crashing.

## Notes

- This project is intended as an educational / college-level demonstration
  of applying classic facial-landmark geometry (EAR/MAR/head pose) to a
  real-time safety use case. It is **not** a certified safety device.
- Everything (face mesh model, alarm sound, computation) runs 100% locally;
  no internet connection, account, or paid API key is required at runtime.

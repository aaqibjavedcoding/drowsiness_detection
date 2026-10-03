# Simple Drowsiness Detector

A small, single-file OpenCV webcam demo for learning Face Mesh, eye closure and yawns. It has no score engine, head pose, Tkinter or Flask: low EAR for enough consecutive frames means **DROWSY!**, plays `assets/alarm.wav`, and counts blinks/yawns.

## Run

From the repository root (with a desktop display and webcam):

```bash
pip install opencv-contrib-python mediapipe numpy pygame
python simple/simple_detector.py
```

Press **q** in the OpenCV window to exit. If the camera cannot open or a face is not visible, the window shows a message instead of crashing.

## EAR / MAR in five lines

1. Face Mesh gives six landmark points around each eye.
2. `EAR = (two vertical eye distances) / (two times the horizontal eye distance)`.
3. A small EAR means the eyelids are close together, so the eye is probably closed.
4. `MAR = mouth height / mouth width` using upper/lower lip and mouth-corner points.
5. A large MAR that stays high for several frames is counted as one yawn.

## Numbers to tune

At the top of `simple_detector.py`, change `CAMERA_INDEX` for another webcam, `EAR_THRESHOLD` for eye-closed sensitivity, `BLINK_FRAMES` for blink filtering, `DROWSY_FRAMES` for when the alarm starts, and `MAR_THRESHOLD` / `YAWN_FRAMES` for yawn sensitivity. Try values in your own lighting and camera position; this is a learning demo, not a safety device.

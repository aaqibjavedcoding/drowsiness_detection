"""A small webcam drowsiness demo: press q in the window to quit."""
from pathlib import Path
import numpy as np
# ---- Numbers students can tune ----
CAMERA_INDEX = 0                 # webcam number; try 1 if 0 does not work
EAR_THRESHOLD = 0.21             # below this EAR = eyes closed
BLINK_FRAMES = 2                 # closed frames needed to count one blink
DROWSY_FRAMES = 45               # closed frames needed to show DROWSY
MAR_THRESHOLD = 0.55             # above this MAR = mouth open
YAWN_FRAMES = 12                 # open-mouth frames needed to count one yawn
MIN_DETECTION_CONFIDENCE = 0.5   # minimum confidence to find a new face
MIN_TRACKING_CONFIDENCE = 0.5    # minimum confidence to keep tracking a face

WINDOW_NAME = "Simple Drowsiness Detector"
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def ear(points):
    """Return Eye Aspect Ratio for six eye points."""
    vertical = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
    horizontal = 2.0 * np.linalg.norm(points[0] - points[3])
    return vertical / horizontal if horizontal else 0.0

def mar(points):
    """Return Mouth Aspect Ratio: mouth height divided by mouth width."""
    height = np.linalg.norm(points[13] - points[14])
    width = np.linalg.norm(points[78] - points[308])
    return height / width if width else 0.0

def main():
    """Read webcam frames, calculate EAR/MAR, and draw the simple status."""
    # Delayed imports let syntax/import checks run without a camera or OpenCV GUI.
    try:
        import cv2
        import mediapipe as mp
        import pygame
    except ImportError as error:
        print(f"Install the required packages first: {error}")
        return
    alarm_path = Path(__file__).resolve().parents[1] / "assets" / "alarm.wav"
    sound = None
    alarm_playing = False
    # Audio is optional, so a missing speaker never stops the program.
    try:
        pygame.mixer.init()
        sound = pygame.mixer.Sound(str(alarm_path))
    except (pygame.error, FileNotFoundError) as error:
        print(f"Alarm is unavailable: {error}")
    # Open the camera before starting Face Mesh.
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("Camera not found. Check CAMERA_INDEX or webcam permissions.")
        frame = np.zeros((240, 700, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera not found - press q to exit", (25, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        while True:
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break
        camera.release()
        pygame.mixer.quit()
        cv2.destroyAllWindows()
        return

    blinks = yawns = closed_frames = open_mouth_frames = 0
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera frame not available", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                closed_frames = 0
                ear_value = mar_value = 0.0
                status = "CAMERA ERROR"
            else:
                # Face Mesh needs RGB, while OpenCV displays BGR.
                result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ear_value = mar_value = 0.0
                status = "NO FACE"

                if result.multi_face_landmarks:
                    # Change normalized landmarks into pixel coordinates.
                    h, w = frame.shape[:2]
                    marks = result.multi_face_landmarks[0].landmark
                    points = np.array([(p.x * w, p.y * h) for p in marks], dtype=np.float32)
                    ear_value = (ear(points[LEFT_EYE]) + ear(points[RIGHT_EYE])) / 2.0
                    mar_value = mar(points)

                    # Count a blink when closed eyes open again.
                    if ear_value < EAR_THRESHOLD:
                        closed_frames += 1
                    else:
                        if closed_frames >= BLINK_FRAMES:
                            blinks += 1
                        closed_frames = 0

                    # Count a yawn once after the mouth has stayed open.
                    if mar_value > MAR_THRESHOLD:
                        open_mouth_frames += 1
                        if open_mouth_frames == YAWN_FRAMES:
                            yawns += 1
                    else:
                        open_mouth_frames = 0

                    status = "DROWSY!" if closed_frames >= DROWSY_FRAMES else "AWAKE"
                else:
                    closed_frames = open_mouth_frames = 0

            # Play alarm only while the consecutive-frame rule says drowsy.
            drowsy = status == "DROWSY!"
            if drowsy and sound is not None and not alarm_playing:
                sound.play(loops=-1)
                alarm_playing = True
            elif not drowsy and alarm_playing:
                sound.stop()
                alarm_playing = False

            # Draw only the values needed for this small version.
            color = (0, 0, 255) if drowsy else (0, 255, 0)
            cv2.putText(frame, f"EAR: {ear_value:.2f}  MAR: {mar_value:.2f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {blinks}  Yawns: {yawns}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status, (20, 135), cv2.FONT_HERSHEY_SIMPLEX,
                        2.0 if drowsy else 0.9, color, 3)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if alarm_playing and sound is not None:
            sound.stop()
        face_mesh.close()
        camera.release()
        pygame.mixer.quit()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
detection/face_detector.py
----------------------------
Wraps MediaPipe's Face Mesh solution. Converts a BGR webcam frame into a flat
list of (x, y) pixel coordinates for every landmark, which the rest of the
pipeline (eye/mouth/head-pose analyzers) consumes.
"""

import cv2
import mediapipe as mp

import config


class FaceDetector:
    """Detects a single face and its 468/478 landmarks in a video frame."""

    def __init__(self):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self.face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=config.REFINE_LANDMARKS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def process(self, frame_bgr):
        """
        Run detection on a single BGR frame.

        Returns:
            landmarks (list[(int, int)] | None): pixel coordinates of every
                landmark for the first detected face, or None if no face
                was found.
            results: the raw MediaPipe result object (useful for drawing).
        """
        height, width = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, results

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [
            (int(lm.x * width), int(lm.y * height)) for lm in face_landmarks.landmark
        ]
        return landmarks, results

    def draw_landmarks(self, frame_bgr, results):
        """Draw the face mesh contours/tessellation onto `frame_bgr` in place."""
        if results is None or not results.multi_face_landmarks:
            return frame_bgr

        for face_landmarks in results.multi_face_landmarks:
            self._mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=self._mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._mp_styles
                .get_default_face_mesh_tesselation_style(),
            )
            self._mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=self._mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._mp_styles
                .get_default_face_mesh_contours_style(),
            )
        return frame_bgr

    def close(self):
        """Release MediaPipe resources."""
        try:
            self.face_mesh.close()
        except Exception:
            pass

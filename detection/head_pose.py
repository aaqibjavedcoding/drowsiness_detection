"""
detection/head_pose.py
------------------------
Lightweight head-pose estimation using OpenCV's solvePnP with a generic 3D
face model matched against 6 MediaPipe landmarks. This is the same classic
technique used in most OpenCV head-pose tutorials, repurposed here to flag
"head nodding down" or "looking away" behaviour that often accompanies
drowsiness.
"""

import cv2
import numpy as np

import config

# Generic 3D face model points (millimetres), nose tip at the origin.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye, left corner
        (225.0, 170.0, -135.0),   # Right eye, right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)

_LANDMARK_IDS = config.HEAD_POSE_LANDMARK_IDS


class HeadPoseEstimator:
    """Estimates pitch/yaw/roll of the head from 2D facial landmarks."""

    def estimate(self, landmarks, frame_shape):
        """
        Args:
            landmarks: list of (x, y) pixel coordinates from FaceDetector.
            frame_shape: shape tuple of the source frame (h, w, channels).

        Returns a dict with pitch/yaw/roll (degrees) and a human label.
        """
        height, width = frame_shape[:2]

        try:
            image_points = np.array(
                [
                    landmarks[_LANDMARK_IDS["nose_tip"]],
                    landmarks[_LANDMARK_IDS["chin"]],
                    landmarks[_LANDMARK_IDS["left_eye_left_corner"]],
                    landmarks[_LANDMARK_IDS["right_eye_right_corner"]],
                    landmarks[_LANDMARK_IDS["left_mouth_corner"]],
                    landmarks[_LANDMARK_IDS["right_mouth_corner"]],
                ],
                dtype=np.float64,
            )
        except (IndexError, TypeError):
            return self._fallback_result()

        focal_length = width
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

        success, rotation_vector, _ = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return self._fallback_result()

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)

        looking_down = pitch > config.HEAD_PITCH_DOWN_THRESHOLD_DEG
        looking_away = abs(yaw) > config.HEAD_YAW_THRESHOLD_DEG

        if looking_down:
            label = "Head Down"
        elif looking_away:
            label = "Looking Away"
        else:
            label = "Forward"

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "label": label,
            "looking_down": looking_down,
            "looking_away": looking_away,
        }

    @staticmethod
    def _rotation_matrix_to_euler(rotation_matrix):
        """Convert a rotation matrix to pitch/yaw/roll in degrees."""
        sy = np.sqrt(
            rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2
        )
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0

        pitch_deg = float(np.degrees(pitch))
        yaw_deg = float(np.degrees(yaw))
        roll_deg = float(np.degrees(roll))

        # With this particular 3D model-point convention, solvePnP returns a
        # raw pitch close to +-180 degrees when the head faces the camera
        # straight on. Re-centre it around 0 so that a positive value means
        # "chin down / nodding off" and a negative value means "chin up".
        pitch_deg = (pitch_deg - 180.0) if pitch_deg > 0 else (pitch_deg + 180.0)
        return pitch_deg, yaw_deg, roll_deg

    @staticmethod
    def _fallback_result():
        return {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "label": "Unknown",
            "looking_down": False,
            "looking_away": False,
        }

"""
detection/drowsiness_engine.py
---------------------------------
Combines the eye, mouth and head-pose signals into a single 0-100
"Drowsiness Score" and a 3-level state machine (NORMAL / WARNING / DROWSY).
Also keeps track of the session-wide statistics shown on the dashboard.
"""

import time

import config
from detection.eye_analyzer import EyeAnalyzer
from detection.head_pose import HeadPoseEstimator
from detection.mouth_analyzer import MouthAnalyzer
from utils.calculations import clamp


class DrowsinessEngine:
    """Runs all analyzers on each frame and derives the overall drowsiness state."""

    def __init__(self):
        self.eye_analyzer = EyeAnalyzer()
        self.mouth_analyzer = MouthAnalyzer()
        self.head_pose_estimator = HeadPoseEstimator()

        self.score = 0.0
        self.state = config.STATE_NORMAL
        self._previous_state = config.STATE_NORMAL

        self.warning_count = 0
        self.drowsy_count = 0
        self._no_face_frames = 0
        self.session_start_time = time.time()

    # ------------------------------------------------------------------
    def update(self, landmarks, frame_shape):
        """
        Process a single frame.

        Args:
            landmarks: list of (x, y) pixel coordinates, or None if no face
                was detected in this frame.
            frame_shape: shape of the current frame (h, w, channels).

        Returns a dict with everything the UI needs to render this frame.
        """
        face_found = landmarks is not None

        if not face_found:
            self._no_face_frames += 1
            # Slowly relax the score while we wait to see if the face comes back.
            self.score = clamp(
                self.score - config.SCORE_DECAY_PER_FRAME,
                config.SCORE_MIN,
                config.SCORE_MAX,
            )
            self._update_state()
            return self._build_result(
                eye_result=None,
                mouth_result=None,
                head_result=None,
                face_found=False,
            )

        self._no_face_frames = 0

        eye_result = self.eye_analyzer.analyze(landmarks)
        mouth_result = self.mouth_analyzer.analyze(landmarks)
        head_result = self.head_pose_estimator.estimate(landmarks, frame_shape)

        contribution = 0.0
        if eye_result["eye_closed"]:
            contribution += config.SCORE_RISE_EYE_CLOSED
        if eye_result["is_long_closure"]:
            contribution += config.SCORE_RISE_LONG_CLOSURE
        if mouth_result["is_yawning"]:
            contribution += config.SCORE_RISE_YAWN
        if head_result["looking_down"]:
            contribution += config.SCORE_RISE_HEAD_DOWN
        if head_result["looking_away"]:
            contribution += config.SCORE_RISE_HEAD_AWAY

        if contribution > 0:
            self.score += contribution
        else:
            self.score -= config.SCORE_DECAY_PER_FRAME

        self.score = clamp(self.score, config.SCORE_MIN, config.SCORE_MAX)

        # A sustained eye closure is a safety-critical microsleep signal:
        # force an immediate DROWSY classification regardless of the score.
        force_drowsy = eye_result["is_long_closure"]

        self._update_state(force_drowsy=force_drowsy)

        return self._build_result(
            eye_result=eye_result,
            mouth_result=mouth_result,
            head_result=head_result,
            face_found=True,
        )

    # ------------------------------------------------------------------
    def _update_state(self, force_drowsy=False):
        if force_drowsy or self.score >= config.DROWSY_SCORE_THRESHOLD:
            new_state = config.STATE_DROWSY
        elif self.score >= config.WARNING_SCORE_THRESHOLD:
            new_state = config.STATE_WARNING
        else:
            new_state = config.STATE_NORMAL

        if new_state != self._previous_state:
            if new_state == config.STATE_WARNING:
                self.warning_count += 1
            elif new_state == config.STATE_DROWSY:
                self.drowsy_count += 1

        self._previous_state = new_state
        self.state = new_state

    def _build_result(self, eye_result, mouth_result, head_result, face_found):
        no_face_timeout = self._no_face_frames >= config.NO_FACE_GRACE_FRAMES
        return {
            "face_found": face_found,
            "no_face_timeout": no_face_timeout,
            "score": self.score,
            "state": self.state,
            "eye": eye_result,
            "mouth": mouth_result,
            "head": head_result,
            "stats": self.get_stats(),
        }

    # ------------------------------------------------------------------
    def get_stats(self):
        """Session statistics shown in the right-hand dashboard panel."""
        return {
            "blink_count": self.eye_analyzer.blink_count,
            "long_blink_count": self.eye_analyzer.long_blink_count,
            "yawn_count": self.mouth_analyzer.yawn_count,
            "warning_count": self.warning_count,
            "drowsy_count": self.drowsy_count,
            "max_closure_duration": self.eye_analyzer.max_closure_duration,
            "session_duration": time.time() - self.session_start_time,
        }

    def reset_session(self):
        """Reset every counter and analyzer (used by the Reset Session button)."""
        self.eye_analyzer.reset()
        self.mouth_analyzer.reset()
        self.score = 0.0
        self.state = config.STATE_NORMAL
        self._previous_state = config.STATE_NORMAL
        self.warning_count = 0
        self.drowsy_count = 0
        self._no_face_frames = 0
        self.session_start_time = time.time()

    def close(self):
        pass

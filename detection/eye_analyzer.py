"""
detection/eye_analyzer.py
----------------------------
Computes the Eye Aspect Ratio (EAR) from real facial landmarks and turns it
into blink, long-blink (microsleep) and continuous-closure-duration signals.
"""

import time

import config
from utils.calculations import RollingAverage, eye_aspect_ratio


class EyeAnalyzer:
    """Tracks eye-closure state across frames and derives blink statistics."""

    def __init__(self):
        self._ear_smoother = RollingAverage(config.EAR_SMOOTHING_WINDOW)
        self._closed_frame_count = 0
        self._closed_since = None
        self._is_closed = False

        self.blink_count = 0
        self.long_blink_count = 0
        self.max_closure_duration = 0.0

    def analyze(self, landmarks):
        """
        Args:
            landmarks: list of (x, y) pixel coordinates from FaceDetector.

        Returns a dict with the current EAR, eye state and blink statistics.
        """
        left_ear = eye_aspect_ratio(landmarks, config.LEFT_EYE_INDICES)
        right_ear = eye_aspect_ratio(landmarks, config.RIGHT_EYE_INDICES)
        raw_ear = (left_ear + right_ear) / 2.0
        smoothed_ear = self._ear_smoother.update(raw_ear)

        now = time.time()
        eye_closed_now = smoothed_ear < config.EAR_THRESHOLD
        closure_duration = 0.0

        if eye_closed_now:
            self._closed_frame_count += 1
            if self._closed_since is None:
                self._closed_since = now
            closure_duration = now - self._closed_since
            self.max_closure_duration = max(self.max_closure_duration, closure_duration)
            self._is_closed = True
        else:
            if self._is_closed and self._closed_frame_count >= config.EAR_CONSEC_FRAMES_BLINK:
                self.blink_count += 1
                if (self._closed_since is not None) and (
                    now - self._closed_since >= config.LONG_CLOSURE_SECONDS
                ):
                    self.long_blink_count += 1
            self._closed_frame_count = 0
            self._closed_since = None
            self._is_closed = False

        is_long_closure = eye_closed_now and closure_duration >= config.LONG_CLOSURE_SECONDS

        return {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "ear": smoothed_ear,
            "eye_closed": eye_closed_now,
            "closure_duration": closure_duration,
            "is_long_closure": is_long_closure,
            "max_closure_duration": self.max_closure_duration,
            "blink_count": self.blink_count,
            "long_blink_count": self.long_blink_count,
        }

    def reset(self):
        """Reset all running statistics (used by the Reset Session button)."""
        self.__init__()

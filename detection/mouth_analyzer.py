"""
detection/mouth_analyzer.py
------------------------------
Computes the Mouth Aspect Ratio (MAR) from real facial landmarks to detect
sustained mouth opening (yawning).
"""

import config
from utils.calculations import mouth_aspect_ratio


class MouthAnalyzer:
    """Tracks mouth-opening state across frames and counts yawns."""

    def __init__(self):
        self._open_frame_count = 0
        self._yawn_in_progress = False
        self.yawn_count = 0

    def analyze(self, landmarks):
        """
        Args:
            landmarks: list of (x, y) pixel coordinates from FaceDetector.

        Returns a dict with the current MAR, mouth state and yawn count.
        """
        mar = mouth_aspect_ratio(
            landmarks, config.MOUTH_VERTICAL_PAIR, config.MOUTH_HORIZONTAL_PAIR
        )
        mouth_open = mar > config.MAR_THRESHOLD

        if mouth_open:
            self._open_frame_count += 1
        else:
            self._open_frame_count = 0
            self._yawn_in_progress = False

        is_yawning = self._open_frame_count >= config.YAWN_CONSEC_FRAMES

        if is_yawning and not self._yawn_in_progress:
            self.yawn_count += 1
            self._yawn_in_progress = True

        return {
            "mar": mar,
            "mouth_open": mouth_open,
            "is_yawning": is_yawning,
            "yawn_count": self.yawn_count,
        }

    def reset(self):
        """Reset all running statistics (used by the Reset Session button)."""
        self.__init__()

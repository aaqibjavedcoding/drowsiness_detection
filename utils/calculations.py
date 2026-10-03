"""
utils/calculations.py
----------------------
Small, dependency-light math helpers shared by the eye, mouth and head-pose
analyzers. Keeping these pure functions separate makes them trivial to unit
test without needing a webcam or a GUI.
"""

from collections import deque

import numpy as np


def euclidean_distance(point_a, point_b):
    """Euclidean distance between two (x, y) pixel coordinates."""
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(points, indices):
    """
    Classic Eye Aspect Ratio (EAR) formula (Soukupova & Cech, 2016).

    `indices` must contain 6 landmark ids in the order:
        [outer corner, top-1, top-2, inner corner, bottom-1, bottom-2]

    EAR = (||top1-bottom1|| + ||top2-bottom2||) / (2 * ||outer-inner||)
    """
    p1, p2, p3, p4, p5, p6 = (points[i] for i in indices)
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    if horizontal <= 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(points, vertical_pair, horizontal_pair):
    """
    Mouth Aspect Ratio (MAR) = vertical mouth opening / mouth width.
    Used to detect yawning.
    """
    top_idx, bottom_idx = vertical_pair
    left_idx, right_idx = horizontal_pair
    vertical = euclidean_distance(points[top_idx], points[bottom_idx])
    horizontal = euclidean_distance(points[left_idx], points[right_idx])
    if horizontal <= 1e-6:
        return 0.0
    return vertical / horizontal


def clamp(value, min_value, max_value):
    """Clamp `value` into the [min_value, max_value] range."""
    return max(min_value, min(max_value, value))


class RollingAverage:
    """Tiny fixed-size rolling average used to smooth noisy per-frame ratios."""

    def __init__(self, window_size):
        self.window_size = max(1, int(window_size))
        self._values = deque(maxlen=self.window_size)

    def update(self, value):
        self._values.append(float(value))
        return self.value

    @property
    def value(self):
        if not self._values:
            return 0.0
        return float(sum(self._values) / len(self._values))

    def reset(self):
        self._values.clear()

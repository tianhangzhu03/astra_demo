"""A visible yellow tape patch, measured before any AR drawing. No depth prediction."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Marker:
    pixel: np.ndarray
    mask: np.ndarray
    area: int


def detect_marker(frame, hand=None):
    if frame is None:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (18, 90, 65), (38, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    candidates = np.flatnonzero((areas >= 9) & (areas <= frame.shape[0]*frame.shape[1]*.015))+1
    # Require the actual index distal segment (MediaPipe DIP=7, TIP=8).
    # Never search the whole frame when landmarks are missing.
    segment = getattr(hand, 'index_distal', None)
    if segment is None or not np.isfinite(segment).all():
        return None
    a, b = np.asarray(segment, dtype=float)
    axis = b-a
    length = float(np.linalg.norm(axis))
    if length < 3 or length > hand.scale*.8:
        return None
    along = np.clip((centers[candidates]-a) @ axis / (length*length), 0., 1.)
    nearest = a + along[:,None]*axis
    radius = max(7., min(hand.scale*.22, length*.65))
    candidates = candidates[np.linalg.norm(centers[candidates]-nearest, axis=1) <= radius]
    # Ambiguous color patches must not silently switch identity.
    if len(candidates) != 1:
        return None
    key = int(candidates[0])
    selected = np.uint8(labels == key)
    return Marker(centers[key].copy(), selected, int(stats[key, cv2.CC_STAT_AREA]))

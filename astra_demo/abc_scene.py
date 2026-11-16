"""Compact A -> B -> C geometry in a tabletop frame (Z up)."""

import cv2
import numpy as np


REACH = np.array([[-5.85, -3.15, .8], [-5.85, 2.75, .8], [1.75, 2.75, .8]])
CARRY = np.array([[1.75, 2.75, .8], [1.75, -3.45, .8], [6.6, -3.45, .8],
                  [6.6, 3.12, .8], [5.35, 3.12, .8]])
# x, y, width, depth, height; same proportions as web/abc-scene/scene.html.
OBSTACLES = np.array([[-1.9, -1.6, 5.4, 5.2, 1.3], [4.275, -.25, 2.95, 4.8, 1.65],
                      [3.55, 3.14, 1.5, 1.98, 1.65]])
# Shorten travel 20% inside the unchanged workspace; keep the outer-corridor layout.
LAYOUT_SCALE = .8
REACH[:, :2] *= LAYOUT_SCALE
CARRY[:, :2] *= LAYOUT_SCALE
OBSTACLES[:, :4] *= LAYOUT_SCALE
C = CARRY[-1]
BOX_WIDTH, BOX_BOTTOM, BOX_ROOF = 1.98, 2.05, 3.12
BOX_WIDTH *= LAYOUT_SCALE
BALL_COLOR = (60, 100, 245)  # Constant orange-red in BGR; yellow is reserved for tape.


def dashed_path(image, points, color, thickness=2, dash=10., gap=7.):
    for a, b in zip(points[:-1], points[1:]):
        delta = np.asarray(b, dtype=float)-a
        length = np.linalg.norm(delta)
        if length < 1:
            continue
        direction = delta/length
        for start in np.arange(0, length, dash+gap):
            p, q = np.rint([a+direction*start, a+direction*min(start+dash, length)]).astype(int)
            cv2.line(image, tuple(p), tuple(q), color, thickness, cv2.LINE_AA)

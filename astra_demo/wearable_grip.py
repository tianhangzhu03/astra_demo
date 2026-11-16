"""Side-view whole-hand grip; Top supplies palm proximity, never finger-pose votes.

Four MCP/PIP/DIP chains contribute a robust consensus without fingertip positions.
World landmarks are model estimates, NOT calibrated camera/depth measurements.
"""

from collections import deque
from dataclasses import replace
import math

import numpy as np

from .grab_logic import GrabContext, GrabOutput, GrabState, update_grab_state


def four_finger_pose(image_points, world_lms, width, height):
    """Return aggregate openness and four individual scores (None = unusable)."""
    scores = [None] * 4
    if world_lms is None:
        return None, tuple(scores)
    world = np.array([(p.x, p.y, p.z) for p in world_lms.landmark], dtype=float)
    if world.shape != (21, 3):
        return None, tuple(scores)
    for index, base in enumerate((5, 9, 13, 17)):
        used = [0, base, base+1, base+2]
        image = np.asarray(image_points)[used]
        if (not np.isfinite(world[used]).all() or not np.isfinite(image).all()
                or (image < [0, 0]).any() or (image >= [width, height]).any()):
            continue
        vectors = np.diff(world[[0, base, base+1, base+2]], axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        # Reject collapsed/implausible inferred geometry and tiny projected joints.
        if not (.025 <= lengths[0] <= .16 and
                .15 <= lengths[1]/lengths[0] <= 1.0 and
                .08 <= lengths[2]/lengths[0] <= .75):
            continue
        projected = np.linalg.norm(np.diff(image_points[[base, base+1, base+2]], axis=0), axis=1)
        if projected.min() < 2:
            continue
        directions = vectors / lengths[:, None]
        angles = np.degrees(np.arccos(np.clip(np.sum(directions[:-1]*directions[1:], axis=1), -1, 1)))
        # Wrist-to-knuckle directions include normal finger splay. Discount its
        # first 20 degrees and emphasize PIP flexion rather than hand pitch.
        bend = .35 * max(0., angles[0]-20.) + .65 * angles[1]
        scores[index] = float(np.clip(1 - bend/90., 0, 1))
    valid = sorted(v for v in scores if v is not None)
    if len(valid) < 2:
        return None, tuple(scores)
    # 3 of 4, 2 of 3, or both of 2 usable fingers must agree. One bad chain
    # cannot block the other three, and one moving finger cannot grab/release.
    quorum = max(2, math.ceil(len(valid)*.65))
    closed, opened = valid[quorum-1], valid[-quorum]
    score = closed if closed <= .42 else opened if opened >= .60 else .5
    return score, tuple(scores)


def wearable_observation(observation, *, top=False):
    """Top uses the palm base region; it need not see any middle finger joints."""
    if observation is None:
        return observation
    anchor = observation.palm_points[1:].mean(axis=0) if top else observation.grip_xy
    return observation if anchor is None else replace(observation, pinch_xy=anchor)


def near_palm(observation, ball_xy, radius, margin=20.):
    """Sphere intersects the CURRENT detected palm hull plus the same pixel margin."""
    import cv2
    if observation is None or radius <= 0 or margin < 0:
        return False
    points = np.asarray(observation.palm_points, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all() or not np.isfinite((*ball_xy, radius, margin)).all():
        return False
    hull = cv2.convexHull(points)
    if cv2.contourArea(hull) < 20:
        return False
    return cv2.pointPolygonTest(hull, tuple(map(float, ball_xy)), True) >= -(radius+margin)


class WearableGrip:
    def __init__(self, median_window=3, **settings):
        self.settings = settings
        self.median_window = median_window
        self.reset()

    def reset(self):
        self.context = GrabContext()
        self.history = deque(maxlen=self.median_window)
        self.held = False
        self.source = ''
        self.score = None

    def update(self, top, side, *, prompt_sent=False, cancel_prompt=False,
               manual_release=False, **gates):
        if manual_release:
            self.reset()
            return GrabOutput(GrabContext(), False)
        value = side.grip_openness if side is not None else None
        # Do not allow a median's old OPEN value to finish release while the
        # current sample says CLOSED (or vice versa).
        if (prompt_sent or cancel_prompt or (not self.held and
                (not gates.get('hand_near_ball') or not gates.get('top_anchor_valid')
                 or not gates.get('setup_ready', True)))):
            self.history.clear()
        if value is None or not math.isfinite(value):
            self.history.clear()
            self.score = None
        else:
            self.history.append(value)
            median = float(np.median(self.history))
            same_band = ((value >= .60 and median >= .60) or (value <= .42 and median <= .42))
            self.score = (min(value, median) if value >= .60 else max(value, median)) if same_band else .5
        out = update_grab_state(self.context, self.score, prompt_sent=prompt_sent,
                                cancel_prompt=cancel_prompt, **self.settings, **gates)
        if self.held and not out.trigger_on:
            self.reset()
            return out
        self.context, self.held = out.context, out.trigger_on
        if self.held:
            self.source = 'side'
        labels = {'PINCH TO GRAB': 'CLOSE HAND', 'OPEN FINGERS': 'OPEN HAND',
                  'SHOW SIDE FINGERS': 'SHOW SIDE GRIP', 'SHOW TOP FINGERS': 'SHOW TOP PALM'}
        return replace(out, status=labels.get(out.status, out.status))

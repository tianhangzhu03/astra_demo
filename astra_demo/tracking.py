"""Palm-relative grasp tracking, bounded optical-flow bridging and adaptive smoothing."""

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np


@dataclass
class HandObservation:
    palm: np.ndarray
    basis: np.ndarray
    scale: float
    pinch_xy: np.ndarray
    pinch_ratio: float | None
    pinch_norm: float | None
    palm_points: np.ndarray
    grip_openness: float | None = None
    grip_xy: np.ndarray | None = None
    grip_fingers: tuple = (None, None, None, None)
    index_distal: np.ndarray | None = None


def observe_hand(hand_lms, width, height, *, allow_edge_on=False, world_lms=None):
    """Keep palm tracking usable even when fingertip geometry is implausible."""
    pts = np.array([(p.x * width, p.y * height) for p in hand_lms.landmark], dtype=float)
    if pts.shape != (21, 2):
        return None
    palm_pts = pts[[0, 5, 9, 13, 17]]
    if (not np.isfinite(palm_pts).all() or (palm_pts < [-0.1 * width, -0.1 * height]).any()
            or (palm_pts > [1.1 * width, 1.1 * height]).any()):
        return None
    across = pts[5] - pts[17]
    length = float(np.linalg.norm(pts[9] - pts[0]))
    breadth = float(np.linalg.norm(across))
    scale = max(length, breadth)
    if length < 6 or (breadth < 6 and not allow_edge_on) or scale > max(width, height) * 0.8:
        return None
    # A side-on fist can collapse the knuckle span in the image. Keep its palm
    # position usable, but do not infer a new grasp/release from that geometry.
    edge_on = breadth < 6
    long_axis = (pts[9] - pts[0]) / length
    axis = np.array([long_axis[1], -long_axis[0]]) if edge_on else across / breadth
    basis = np.column_stack((axis, [-axis[1], axis[0]]))
    palm = palm_pts.mean(axis=0)
    # Cluster midpoint is only used to attach the ball initially. While held, the palm drives it.
    thumb = pts[4] * .60 + pts[3] * .28 + pts[2] * .12
    index = pts[8] * .55 + pts[7] * .30 + pts[6] * .15
    pinch_xy = (thumb + index) * .5
    bones = [(2, 3), (3, 4), (5, 6), (6, 7), (7, 8)]
    lengths = np.array([np.linalg.norm(pts[a] - pts[b]) / scale for a, b in bones])
    tips_inside = ((pts[[4, 8]] >= [0, 0]).all() and
                   (pts[[4, 8]] <= [width - 1, height - 1]).all())
    plausible = not edge_on and tips_inside and np.all((lengths > .025) & (lengths < 1.0))
    ratio = float(np.linalg.norm(pts[4] - pts[8]) / scale) if plausible else None
    norm = float(np.linalg.norm((thumb - index) / [width, height])) if plausible else None
    from .wearable_grip import four_finger_pose
    grip, fingers = four_finger_pose(pts, world_lms, width, height)
    # A palm-base center remains available when folded middle joints/tips vanish.
    grip_xy = palm_pts[1:].mean(axis=0)
    return HandObservation(palm, basis, scale, pinch_xy, ratio, norm, palm_pts, grip, grip_xy, fingers, pts[[7, 8]].copy())


class OneEuroFilter:
    def __init__(self, min_cutoff=1.4, beta=.015, derivative_cutoff=1.0, *, axiswise=False):
        self.min_cutoff, self.beta, self.derivative_cutoff = min_cutoff, beta, derivative_cutoff
        self.axiswise = axiswise
        self.value = self.raw = self.derivative = self.time = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

    def reset(self, point, now):
        self.raw = self.value = np.asarray(point, dtype=float).copy()
        self.derivative = np.zeros_like(self.value)
        self.time = now
        return self.value.copy()

    def update(self, point, now):
        point = np.asarray(point, dtype=float)
        if self.value is None:
            return self.reset(point, now)
        dt = min(.15, max(.001, now - self.time))
        velocity = (point - self.raw) / dt
        a = self._alpha(self.derivative_cutoff, dt)
        self.derivative = a * velocity + (1 - a) * self.derivative
        speed = np.abs(self.derivative) if self.axiswise else float(np.linalg.norm(self.derivative))
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)
        self.value = a * point + (1 - a) * self.value
        self.raw, self.time = point.copy(), now
        return self.value.copy()


class PositionSmoother:
    """Small input deadband plus per-axis adaptive filtering; no prediction."""
    def __init__(self, min_cutoff=.9, beta=.012, deadband_px=1.5):
        self.filter = OneEuroFilter(min_cutoff, beta, axiswise=True)
        self.deadband_px = deadband_px
        self.target = None

    def reset(self, point, now):
        self.target = np.asarray(point, dtype=float).copy()
        return self.filter.reset(self.target, now)

    def update(self, point, now):
        point = np.asarray(point, dtype=float)
        if self.target is None:
            return self.reset(point, now)
        delta = point - self.target
        # Independent axes prevent horizontal motion from weakening vertical
        # noise suppression. Slow motion still accumulates beyond the deadband.
        self.target += np.sign(delta) * np.maximum(np.abs(delta) - self.deadband_px, 0)
        return self.filter.update(self.target, now)


class DisplayPalmTrace:
    """Wrist-translation display proxy, causal smoothing and isolated jump rejection.

    The initial palm-minus-wrist offset is frozen within a trial. It reduces
    pitch-driven knuckle-centroid motion; this is not a calibrated 3D correction.
    Raw palm positions used by accuracy metrics are untouched.
    """
    def __init__(self, min_cutoff=.65, beta=.006, deadband_px=3.):
        self.filter = PositionSmoother(min_cutoff, beta, deadband_px)
        self.last_seen = None
        self.offset = self.accepted = self.candidate = None
        self.candidate_time = None
        self.segment_id = 0
        self.status = 'missing'

    def update(self, point, now, *, wrist=None, scale=80.):
        if point is None:
            self.status = 'missing'
            return None
        point = np.asarray(point, dtype=float)
        if wrist is not None and np.isfinite(wrist).all():
            wrist = np.asarray(wrist, dtype=float)
            if self.offset is None:
                self.offset = point-wrist
            point = wrist+self.offset
        reset = self.last_seen is None or not 0 < now-self.last_seen <= .35
        if not reset and self.accepted is not None:
            limit = max(30., float(scale)*.45)
            if np.linalg.norm(point-self.accepted) > limit:
                if (self.candidate is None or now-self.candidate_time > .25 or
                        np.linalg.norm(point-self.candidate) > limit):
                    self.candidate, self.candidate_time = point.copy(), now
                    self.status = 'jump_rejected'
                    return None
                reset = True  # Reacquired elsewhere: start a new stroke, no diagonal bridge.
        self.candidate = self.candidate_time = None
        self.accepted = point.copy()
        self.last_seen = now
        self.status = 'wrist_translation' if wrist is not None else 'palm_smoothed'
        if reset:
            self.segment_id += 1
        return self.filter.reset(point, now) if reset else self.filter.update(point, now)


class PalmTracker:
    def __init__(self, flow_hold_s=.7, min_cutoff=1.4, beta=.015, *, translation_only=False, deadband_px=0.,
                 model_weight=1., anchor_offset_scale=1.):
        self.flow_hold_s = flow_hold_s
        self.translation_only = translation_only
        self.model_weight = model_weight
        self.anchor_offset_scale = anchor_offset_scale
        self.filter = (PositionSmoother(min_cutoff, beta, deadband_px) if translation_only else
                       OneEuroFilter(min_cutoff, beta))
        self.gray = self.points = self.anchor = self.palm = None
        self.last_seen = -1e9
        self.offset = None
        self.was_held = False
        self.status = 'No hand'
        self.scale = 80.0
        self.candidate = None
        self.region = None

    def _flow(self, gray):
        if self.gray is None or self.points is None or len(self.points) < 4 or self.gray.shape != gray.shape:
            return None, None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None,
                                            winSize=(21, 21), maxLevel=3)
        if nxt is None:
            return None, None
        back, bst, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, nxt, None,
                                              winSize=(21, 21), maxLevel=3)
        if back is None:
            return None, None
        keep = ((st.ravel() == 1) & (bst.ravel() == 1) &
                (np.linalg.norm(back - self.points, axis=2).ravel() < 1.5))
        old, new = self.points[keep].reshape(-1, 2), nxt[keep].reshape(-1, 2)
        if len(old) < 4:
            return None, None
        matrix, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC,
                                                     ransacReprojThreshold=2.5)
        if matrix is None or inliers.sum() < max(4, len(old) * .6):
            return None, None
        scale = np.linalg.norm(matrix[:, 0])
        if not .8 < scale < 1.25:
            return None, None
        if self.translation_only:
            # Retain the grasp offset in image pixels, including during short
            # occlusions. A noisy side-view rotation must not swing the ball.
            shift = np.median((new-old)[inliers.ravel() == 1], axis=0)
            matrix = np.array([[1., 0., shift[0]], [0., 1., shift[1]]])
        return matrix, new[inliers.ravel() == 1].reshape(-1, 1, 2)

    def _seed(self, gray, observation):
        mask = np.zeros(gray.shape, np.uint8)
        hull = cv2.convexHull(observation.palm_points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        features = cv2.goodFeaturesToTrack(gray, 36, .01, 5, mask=mask)
        self.points = (features if features is not None and len(features) >= 4 else
                       observation.palm_points.astype(np.float32).reshape(-1, 1, 2))

    def update(self, frame, observation, held, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matrix, flow_points = self._flow(gray) if held else (None, None)
        entering = held and not self.was_held
        if not held:
            self.offset = None
        if observation is not None and held and not entering and self.palm is not None:
            expected = (matrix @ np.append(self.palm, 1)) if matrix is not None else self.palm
            jump_limit = max(24, self.scale * .4) if self.translation_only else max(70, self.scale * .9)
            if np.linalg.norm(observation.palm - expected) > jump_limit:
                # Reject a one-frame detection jump; permit a consistently reacquired hand.
                if self.candidate is None or np.linalg.norm(observation.palm - self.candidate) > self.scale * .4:
                    self.candidate = observation.palm.copy()
                    observation = None
                else:
                    self.candidate = None
            else:
                self.candidate = None
        if observation is not None:
            if held and not entering and matrix is not None and self.palm is not None and self.model_weight < 1.:
                expected = matrix @ np.append(self.palm, 1)
                correction = (expected-observation.palm)*(1.-self.model_weight)
                observation = replace(observation, palm=observation.palm+correction,
                    pinch_xy=observation.pinch_xy+correction, palm_points=observation.palm_points+correction)
            if held and self.offset is None:
                offset = (observation.pinch_xy - observation.palm) * self.anchor_offset_scale
                self.offset = offset.copy() if self.translation_only else observation.basis.T @ offset / observation.scale
            if held:
                offset = self.offset if self.translation_only else observation.basis @ self.offset * observation.scale
                self.anchor = observation.palm + offset
            else:
                self.anchor = observation.pinch_xy.copy()
            self.palm, self.scale = observation.palm.copy(), observation.scale
            self.region = observation.palm_points.copy()
            self.last_seen = now
            self._seed(gray, observation)
            self.status = 'Hand tracked'
        elif held and matrix is not None and self.anchor is not None and now - self.last_seen <= self.flow_hold_s:
            self.anchor = matrix @ np.append(self.anchor, 1)
            self.palm = matrix @ np.append(self.palm, 1)
            if self.region is not None:
                self.region = np.column_stack((self.region, np.ones(len(self.region)))) @ matrix.T
            self.points = flow_points
            self.status = 'Motion tracking'
        else:
            self.points = None
            self.region = None
            self.status = 'Tracking lost - hold retained' if held else 'No hand'
            self.gray, self.was_held = gray, held
            return None
        self.anchor = np.clip(self.anchor, [0, 0], [frame.shape[1] - 1, frame.shape[0] - 1])
        self.gray, self.was_held = gray, held
        point = (self.filter.reset(self.anchor, now) if entering else self.filter.update(self.anchor, now))
        return tuple(point)


class TopHandTracker:
    """One palm-motion source for current proximity, display trail and held ball.

    Optical Flow validates the current image during brief landmark loss; a cached
    position alone never permits a new grasp. The ball's image offset is fixed
    at attachment, without hand-scale or hand-angle multiplication.
    """
    def __init__(self, flow_hold_s=.5, min_cutoff=1.1, beta=.012, deadband_px=2.):
        self.settings = dict(flow_hold_s=flow_hold_s, min_cutoff=min_cutoff, beta=beta,
                             translation_only=True, deadband_px=deadband_px, model_weight=.25)
        self.motion = PalmTracker(**self.settings)
        self.point = self.offset = None
        self.was_held = False
        self.segment_id = 0
        self.status = 'No hand'
        self.now = 0.

    def observe(self, frame, observation, now):
        previously_valid = self.point is not None
        self.now = now
        # Never drag a new detection from an expired old location.
        expired = now-self.motion.last_seen > self.motion.flow_hold_s
        if observation is not None and expired:
            self.motion = PalmTracker(**self.settings)
        center = replace(observation, pinch_xy=observation.palm.copy()) if observation is not None else None
        self.point = self.motion.update(frame, center, True, now) if frame is not None else None
        self.status = self.motion.status if frame is not None else 'Top camera unavailable'
        if self.point is not None and (not previously_valid or expired):
            self.segment_id += 1

    @property
    def palm_points(self):
        return self.motion.region if self.point is not None else None

    @property
    def valid(self):
        return self.palm_points is not None

    def follow_ball(self, ball_xy, held, *, palm_centered=False):
        if not held:
            self.offset = None
        elif self.point is not None and (not self.was_held or self.offset is None):
            # Manual ABC grabs need not occur at the original ball position.
            self.offset = np.zeros(2) if palm_centered else np.asarray(ball_xy)-self.point
        self.was_held = held
        if self.point is None:
            return None
        return tuple(np.asarray(self.point)+self.offset) if held else self.point

    def record(self):
        return dict(point=self.point, segment=self.segment_id, source=self.status,
                    age_ms=round((self.now-self.motion.last_seen)*1000) if self.valid else None,
                    valid=int(self.valid))

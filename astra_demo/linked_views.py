"""Two image-space renderings share one grasp state; no depth or geometric calibration."""

import numpy as np

from .tracking import PalmTracker
from .virtual_prop import VirtualProp, VirtualPropState


def camera_point(x, y, transform, size):
    """Map a letterboxed preview click to displayed camera pixels, ignoring its margins."""
    if transform is None or size is None:
        return None
    dx, dy, sx, sy = transform
    if sx <= 0 or sy <= 0:
        return None
    point = ((x-dx)/sx, (y-dy)/sy)
    return point if 0 <= point[0] < size[0] and 0 <= point[1] < size[1] else None


class LinkedSideView:
    def __init__(self, flow_hold_s=.7, min_cutoff=.9, beta=.012, deadband_px=1.5, palm_inset=.25):
        if not 0 <= palm_inset <= 1:
            raise ValueError('palm_inset must be between 0 and 1')
        self.tracker_settings = dict(flow_hold_s=flow_hold_s, min_cutoff=min_cutoff, beta=beta,
                                     translation_only=True, deadband_px=deadband_px,
                                     anchor_offset_scale=1.-palm_inset)
        self.tracker = PalmTracker(**self.tracker_settings)
        self.prop = VirtualProp(state=VirtualPropState.HIDDEN, size_follow=31, size_held=31)
        self.origin = self.size = None
        self.tracked = None
        self.status = 'Click Side View to place the ball'

    @property
    def ready(self):
        return self.origin is not None

    def set_frame_size(self, size):
        size = tuple(size)
        changed = self.size is not None and self.size != size
        if changed:
            self.prop.pos_xy = tuple(np.asarray(self.prop.pos_xy)*np.asarray(size)/np.asarray(self.size))
            self.origin = None
            self.prop.state = VirtualPropState.HIDDEN
            self.tracker = PalmTracker(**self.tracker_settings)
            self.tracked = None
            self.status = 'Side resolution changed; click to place the ball again'
        self.size = size
        self.prop.size_follow = self.prop.size_held = max(8, round(31*size[1]/480))
        return changed

    def place(self, point, *, locked=False):
        if locked:
            raise ValueError('End the trial and release the ball before changing its side position')
        if self.size is None or not np.isfinite(point).all() or not (0 <= point[0] < self.size[0] and 0 <= point[1] < self.size[1]):
            raise ValueError('Click inside the side camera image')
        self.origin = tuple(map(float, point))
        self.reset()

    def reset(self):
        self.tracker = PalmTracker(**self.tracker_settings)
        self.tracked = None
        if self.ready:
            self.prop.initialize_at(self.origin)
            self.status = 'Side ball ready'

    def update(self, frame, observation, held, now):
        self.tracked = (self.tracker.update(frame, observation, held, now)
                        if self.ready and frame is not None else None)
        if not self.ready:
            if held:
                self.prop.update(None, True, round(now*1000))
            else:
                self.prop.state = VirtualPropState.HIDDEN
            return
        # The main grasp state is the only authority; this view never attaches/releases itself.
        self.prop.update(self.tracked, held, round(now*1000))
        self.status = self.tracker.status if frame is not None else 'Side camera unavailable - position retained'

    def draw(self, frame):
        self.prop.draw(frame, show_label=False)

    def record(self, observation):
        palm = observation.palm if observation is not None else (None, None)
        tracked = self.tracked if self.tracked is not None else (None, None)
        ball = self.prop.pos_xy if self.ready else (None, None)
        return dict(side_palm_x_px=palm[0], side_palm_y_px=palm[1],
            side_tracked_x_px=tracked[0], side_tracked_y_px=tracked[1],
            side_ball_x_px=ball[0], side_ball_y_px=ball[1],
            side_ball_rise_px=self.origin[1]-ball[1] if self.ready else None,
            side_position_valid=int(self.tracked is not None), side_tracking_status=self.status)

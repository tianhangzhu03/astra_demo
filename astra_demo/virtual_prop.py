from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .sphere_renderer import draw_sphere


class VirtualPropState(str, Enum):
    HIDDEN = "HIDDEN"
    IDLE = "IDLE"
    HELD = "HELD"


@dataclass
class VirtualProp:
    state: VirtualPropState = VirtualPropState.IDLE
    dock_xy: Tuple[float, float] = (120.0, 120.0)
    pos_xy: Tuple[float, float] = (120.0, 120.0)

    release_return_ms: int = 200
    follow_alpha: float = 0.35
    size_follow: int = 22
    size_held: int = 28
    follow_fill_alpha: float = 0.45

    def set_dock(self, xy: Tuple[float, float]) -> None:
        self.dock_xy = xy

    def initialize_at(self, xy: Tuple[float, float], visible: bool = True) -> None:
        self.pos_xy = (float(xy[0]), float(xy[1]))
        self.state = VirtualPropState.IDLE if visible else VirtualPropState.HIDDEN

    def update(self, hand_xy: Optional[Tuple[int, int]], grab_active: bool, now_ms: int, keep_idle_visible: bool = True) -> None:
        if not grab_active:
            self.state = VirtualPropState.IDLE if keep_idle_visible else VirtualPropState.HIDDEN
            return
        if hand_xy is None:
            # During fast top-view motion or zone crossing, keep the last held pose instead of dropping immediately.
            if self.state != VirtualPropState.HELD:
                self.state = VirtualPropState.IDLE if keep_idle_visible else VirtualPropState.HIDDEN
            return

        entering_held = self.state != VirtualPropState.HELD
        self.state = VirtualPropState.HELD
        hand_f = (float(hand_xy[0]), float(hand_xy[1]))
        if entering_held:
            # Snap on the first held frame to avoid visible catch-up lag.
            self.pos_xy = hand_f
            return

        px, py = self.pos_xy
        dist = float(np.hypot(hand_f[0] - px, hand_f[1] - py))
        # Adaptive alpha: move faster when far away, preserve smoothness for small jitter.
        alpha_boost = min(0.22, dist / 220.0)
        alpha = min(0.92, max(0.0, self.follow_alpha + alpha_boost))
        self.pos_xy = (
            (1.0 - alpha) * px + alpha * hand_f[0],
            (1.0 - alpha) * py + alpha * hand_f[1],
        )

    def draw(self, frame: np.ndarray, show_label: bool = True) -> None:
        if self.state == VirtualPropState.HIDDEN:
            return
        radius = self.size_held if self.state == VirtualPropState.HELD else self.size_follow
        draw_sphere(frame, self.pos_xy, radius)

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np


class VirtualPropType(str, Enum):
    BALL = "BALL"
    CUBE = "CUBE"


class VirtualPropState(str, Enum):
    FOLLOW = "FOLLOW"
    HELD = "HELD"
    RELEASE_RETURN = "RELEASE_RETURN"


@dataclass
class VirtualProp:
    prop_type: VirtualPropType = VirtualPropType.BALL
    state: VirtualPropState = VirtualPropState.FOLLOW
    dock_xy: Tuple[float, float] = (120.0, 120.0)
    pos_xy: Tuple[float, float] = (120.0, 120.0)

    release_return_ms: int = 200
    follow_alpha: float = 0.35
    size_follow: int = 22
    size_held: int = 28
    follow_fill_alpha: float = 0.45

    _release_start_ms: int = 0
    _release_from_xy: Tuple[float, float] = (120.0, 120.0)

    def toggle_type(self) -> None:
        self.prop_type = VirtualPropType.CUBE if self.prop_type == VirtualPropType.BALL else VirtualPropType.BALL

    def set_dock(self, xy: Tuple[float, float]) -> None:
        self.dock_xy = xy

    def update(self, hand_xy: Optional[Tuple[int, int]], grab_active: bool, now_ms: int) -> None:
        if hand_xy is None:
            self.state = VirtualPropState.FOLLOW
            self.pos_xy = self.dock_xy
            self._release_start_ms = 0
            return

        hand_f = (float(hand_xy[0]), float(hand_xy[1]))

        if grab_active:
            self.state = VirtualPropState.HELD
            self.pos_xy = hand_f
            self._release_start_ms = 0
            return

        if self.state == VirtualPropState.HELD:
            self.state = VirtualPropState.RELEASE_RETURN
            self._release_start_ms = now_ms
            self._release_from_xy = self.pos_xy

        if self.state == VirtualPropState.RELEASE_RETURN and self._release_start_ms > 0:
            t = (now_ms - self._release_start_ms) / float(max(1, self.release_return_ms))
            if t >= 1.0:
                self.pos_xy = self.dock_xy
                self.state = VirtualPropState.FOLLOW
                self._release_start_ms = 0
            else:
                sx, sy = self._release_from_xy
                dx, dy = self.dock_xy
                self.pos_xy = (sx + (dx - sx) * t, sy + (dy - sy) * t)
            return

        # FOLLOW: gently chase fingertip.
        px, py = self.pos_xy
        self.pos_xy = (
            (1.0 - self.follow_alpha) * px + self.follow_alpha * hand_f[0],
            (1.0 - self.follow_alpha) * py + self.follow_alpha * hand_f[1],
        )
        self.state = VirtualPropState.FOLLOW

    def draw(self, frame: np.ndarray) -> None:
        x = int(self.pos_xy[0])
        y = int(self.pos_xy[1])

        held = self.state == VirtualPropState.HELD
        follow = self.state == VirtualPropState.FOLLOW

        size = self.size_held if held else self.size_follow
        color = (30, 80, 245) if held else (40, 220, 245)

        # Light shadow to improve depth cue in demo.
        cv2.circle(frame, (x + 6, y + 6), size, (40, 40, 40), -1)

        if self.prop_type == VirtualPropType.BALL:
            if follow:
                overlay = frame.copy()
                cv2.circle(overlay, (x, y), size, color, -1)
                cv2.addWeighted(overlay, self.follow_fill_alpha, frame, 1.0 - self.follow_fill_alpha, 0, frame)
            else:
                cv2.circle(frame, (x, y), size, color, -1)
        else:
            p1 = (x - size, y - size)
            p2 = (x + size, y + size)
            if follow:
                overlay = frame.copy()
                cv2.rectangle(overlay, p1, p2, color, -1)
                cv2.addWeighted(overlay, self.follow_fill_alpha, frame, 1.0 - self.follow_fill_alpha, 0, frame)
            else:
                cv2.rectangle(frame, p1, p2, color, -1)

        cv2.putText(
            frame,
            f"PROP:{self.prop_type.value}/{self.state.value}",
            (14, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

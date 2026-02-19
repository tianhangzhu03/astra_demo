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
    HIDDEN = "HIDDEN"
    HELD = "HELD"


@dataclass
class VirtualProp:
    prop_type: VirtualPropType = VirtualPropType.BALL
    state: VirtualPropState = VirtualPropState.HIDDEN
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
        # Demo behavior: show only while grabbing. Release => hide immediately.
        if (not grab_active) or hand_xy is None:
            self.state = VirtualPropState.HIDDEN
            return

        self.state = VirtualPropState.HELD
        hand_f = (float(hand_xy[0]), float(hand_xy[1]))
        px, py = self.pos_xy
        self.pos_xy = (
            (1.0 - self.follow_alpha) * px + self.follow_alpha * hand_f[0],
            (1.0 - self.follow_alpha) * py + self.follow_alpha * hand_f[1],
        )

    def draw(self, frame: np.ndarray) -> None:
        if self.state == VirtualPropState.HIDDEN:
            return

        x = int(self.pos_xy[0])
        y = int(self.pos_xy[1])

        held = self.state == VirtualPropState.HELD

        size = self.size_held if held else self.size_follow
        color = (30, 80, 245) if held else (40, 220, 245)

        # Light shadow to improve depth cue in demo.
        cv2.circle(frame, (x + 6, y + 6), size, (40, 40, 40), -1)

        if self.prop_type == VirtualPropType.BALL:
            cv2.circle(frame, (x, y), size, color, -1)
        else:
            p1 = (x - size, y - size)
            p2 = (x + size, y + size)
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

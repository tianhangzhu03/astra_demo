from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional, Tuple

import cv2
import numpy as np


class VirtualPropHardness(str, Enum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class VirtualPropState(str, Enum):
    HIDDEN = "HIDDEN"
    IDLE = "IDLE"
    HELD = "HELD"


@dataclass
class VirtualProp:
    hardness: VirtualPropHardness = VirtualPropHardness.MEDIUM
    state: VirtualPropState = VirtualPropState.IDLE
    dock_xy: Tuple[float, float] = (120.0, 120.0)
    pos_xy: Tuple[float, float] = (120.0, 120.0)

    release_return_ms: int = 200
    follow_alpha: float = 0.35
    size_follow: int = 22
    size_held: int = 28
    follow_fill_alpha: float = 0.45

    _HARDNESS_ORDER: ClassVar[tuple[VirtualPropHardness, ...]] = (
        VirtualPropHardness.SOFT,
        VirtualPropHardness.MEDIUM,
        VirtualPropHardness.HARD,
    )

    def cycle_hardness(self) -> None:
        idx = self._HARDNESS_ORDER.index(self.hardness)
        self.hardness = self._HARDNESS_ORDER[(idx + 1) % len(self._HARDNESS_ORDER)]

    # Backward-compatible alias (old main.py used toggle_type()).
    def toggle_type(self) -> None:
        self.cycle_hardness()

    def set_dock(self, xy: Tuple[float, float]) -> None:
        self.dock_xy = xy

    def initialize_at(self, xy: Tuple[float, float], visible: bool = True) -> None:
        self.pos_xy = (float(xy[0]), float(xy[1]))
        self.state = VirtualPropState.IDLE if visible else VirtualPropState.HIDDEN

    def update(self, hand_xy: Optional[Tuple[int, int]], grab_active: bool, now_ms: int, keep_idle_visible: bool = True) -> None:
        if (not grab_active) or hand_xy is None:
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

    def hardness_freq_hz(self, soft: int, medium: int, hard: int) -> int:
        if self.hardness == VirtualPropHardness.SOFT:
            return soft
        if self.hardness == VirtualPropHardness.MEDIUM:
            return medium
        return hard

    def _colors(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        # Keep the same visual color across hardness modes; haptics differ, visuals stay constant.
        return (40, 190, 250), (20, 140, 220)

    def draw(self, frame: np.ndarray, show_label: bool = True) -> None:
        if self.state == VirtualPropState.HIDDEN:
            return

        x = int(self.pos_xy[0])
        y = int(self.pos_xy[1])

        held = self.state == VirtualPropState.HELD

        size = self.size_held if held else self.size_follow
        idle_color, held_color = self._colors()
        color = held_color if held else idle_color

        # Light shadow to improve depth cue in demo.
        cv2.circle(frame, (x + 6, y + 6), size, (40, 40, 40), -1)
        cv2.circle(frame, (x, y), size, color, -1)

        if show_label:
            cv2.putText(
                frame,
                f"PROP:{self.hardness.value}/{self.state.value}",
                (14, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

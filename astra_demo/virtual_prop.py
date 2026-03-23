from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np


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

    def _colors(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        # Blue material for the demo prop.
        return (250, 160, 70), (170, 95, 25)

    @staticmethod
    def _blend_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, t))
        return tuple(int((1.0 - t) * av + t * bv) for av, bv in zip(a, b))

    def draw(self, frame: np.ndarray, show_label: bool = True) -> None:
        if self.state == VirtualPropState.HIDDEN:
            return

        x = int(self.pos_xy[0])
        y = int(self.pos_xy[1])

        held = self.state == VirtualPropState.HELD
        size = self.size_held if held else self.size_follow
        base_color, rim_color = self._colors()
        pad = int(size * 1.8)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame.shape[1], x + pad + 1)
        y1 = min(frame.shape[0], y + pad + 1)
        if x0 >= x1 or y0 >= y1:
            return

        patch = frame[y0:y1, x0:x1].copy()
        cx = x - x0
        cy = y - y0

        # Soft drop shadow.
        shadow_patch = patch.copy()
        cv2.circle(shadow_patch, (cx + 8, cy + 10), int(size * 1.02), (18, 18, 18), -1)
        shadow_patch = cv2.GaussianBlur(shadow_patch, (0, 0), sigmaX=max(2.0, size * 0.18))
        patch = cv2.addWeighted(shadow_patch, 0.20, patch, 0.80, 0.0)

        # Radial body shading.
        overlay = patch.copy()
        for i in range(size, 0, -1):
            t = 1.0 - (i / max(1.0, float(size)))
            shade = self._blend_color(rim_color, base_color, t)
            cv2.circle(overlay, (cx, cy), i, shade, -1, lineType=cv2.LINE_AA)

        # Directional highlight.
        highlight_center = (int(cx - size * 0.34), int(cy - size * 0.34))
        highlight_color = self._blend_color((255, 255, 255), base_color, 0.78)
        for i in range(max(2, int(size * 0.42)), 0, -1):
            cv2.circle(overlay, highlight_center, i, highlight_color, -1, lineType=cv2.LINE_AA)
            overlay[:] = cv2.addWeighted(overlay, 0.92, patch, 0.08, 0.0)

        # Subtle rim and bottom contact cue.
        cv2.circle(overlay, (cx, cy), size, (245, 245, 245), 1, lineType=cv2.LINE_AA)
        cv2.ellipse(
            overlay,
            (cx, int(cy + size * 0.18)),
            (int(size * 0.72), int(size * 0.32)),
            0,
            0,
            180,
            self._blend_color(rim_color, (255, 255, 255), 0.20),
            2,
            lineType=cv2.LINE_AA,
        )
        frame[y0:y1, x0:x1] = overlay

        if show_label:
            cv2.putText(
                frame,
                "PROP",
                (14, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

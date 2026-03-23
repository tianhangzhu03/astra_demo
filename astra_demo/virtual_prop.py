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
        # Slightly desaturated blue so the prop looks closer to a real coated/plastic sphere.
        return (228, 142, 56), (116, 66, 28)

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
        ss = 2
        patch_ss = cv2.resize(patch, None, fx=ss, fy=ss, interpolation=cv2.INTER_LINEAR)
        cx_ss = cx * ss
        cy_ss = cy * ss
        size_ss = size * ss

        # Soft drop shadow with slightly elongated perspective.
        shadow_patch = patch_ss.copy()
        cv2.ellipse(
            shadow_patch,
            (cx_ss + int(size_ss * 0.16), cy_ss + int(size_ss * 0.28)),
            (int(size_ss * 0.98), int(size_ss * 0.72)),
            0,
            0,
            360,
            (16, 16, 16),
            -1,
            lineType=cv2.LINE_AA,
        )
        shadow_patch = cv2.GaussianBlur(shadow_patch, (0, 0), sigmaX=max(2.0, size_ss * 0.10))
        patch_ss = cv2.addWeighted(shadow_patch, 0.20, patch_ss, 0.80, 0.0)

        overlay = patch_ss.copy()
        sphere_mask = np.zeros(patch_ss.shape[:2], dtype=np.uint8)
        cv2.circle(sphere_mask, (cx_ss, cy_ss), size_ss, 255, -1, lineType=cv2.LINE_AA)

        yy, xx = np.mgrid[0:patch_ss.shape[0], 0:patch_ss.shape[1]]
        nx = (xx - cx_ss) / max(1.0, float(size_ss))
        ny = (yy - cy_ss) / max(1.0, float(size_ss))
        rr = np.sqrt(nx * nx + ny * ny)

        light_x = -0.38
        light_y = -0.52
        light_term = np.clip(1.0 - np.sqrt((nx - light_x) ** 2 + (ny - light_y) ** 2), 0.0, 1.0)
        core_term = np.clip(1.0 - rr, 0.0, 1.0)
        bottom_shade = np.clip((ny + 0.25) * 0.55, 0.0, 0.55)
        fresnel = np.clip((rr - 0.18) / 0.82, 0.0, 1.0)
        shoulder_light = np.clip(1.0 - np.sqrt((nx + 0.05) ** 2 + (ny + 0.10) ** 2), 0.0, 1.0)

        color_field = np.zeros_like(patch_ss, dtype=np.float32)
        base = np.array(base_color, dtype=np.float32)
        rim = np.array(rim_color, dtype=np.float32)
        bright = np.array(self._blend_color(base_color, (250, 226, 205), 0.26), dtype=np.float32)
        cool_fill = np.array(self._blend_color(base_color, (255, 208, 156), 0.12), dtype=np.float32)

        for c in range(3):
            color_field[:, :, c] = rim[c] * np.clip(rr, 0.0, 1.0) + base[c] * core_term
            color_field[:, :, c] = color_field[:, :, c] * (1.0 - bottom_shade)
            color_field[:, :, c] = color_field[:, :, c] + cool_fill[c] * (shoulder_light * 0.22)
            color_field[:, :, c] = color_field[:, :, c] + bright[c] * (light_term * 0.48)
            color_field[:, :, c] = color_field[:, :, c] - rim[c] * (fresnel * 0.08)

        mask = sphere_mask > 0
        overlay[mask] = np.clip(color_field[mask], 0, 255).astype(np.uint8)

        # Smaller glossy highlight to keep the sphere realistic instead of cartoon-like.
        spec_color = self._blend_color(base_color, (255, 236, 220), 0.34)
        cv2.circle(
            overlay,
            (int(cx_ss - size_ss * 0.36), int(cy_ss - size_ss * 0.40)),
            max(2, int(size_ss * 0.18)),
            spec_color,
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            overlay,
            (int(cx_ss - size_ss * 0.18), int(cy_ss - size_ss * 0.15)),
            max(1, int(size_ss * 0.08)),
            self._blend_color(base_color, (255, 242, 230), 0.38),
            -1,
            lineType=cv2.LINE_AA,
        )
        highlight_overlay = cv2.GaussianBlur(overlay, (0, 0), max(1.2, size_ss * 0.035))
        overlay = cv2.addWeighted(highlight_overlay, 0.20, overlay, 0.80, 0.0)

        # Subtle curved reflection to avoid the "painted circle" look.
        cv2.ellipse(
            overlay,
            (int(cx_ss - size_ss * 0.10), int(cy_ss - size_ss * 0.02)),
            (int(size_ss * 0.28), int(size_ss * 0.52)),
            28,
            220,
            296,
            self._blend_color(base_color, (255, 228, 210), 0.18),
            max(1, int(size_ss * 0.03)),
            lineType=cv2.LINE_AA,
        )

        # Darker rim and a little bottom occlusion keep volume without a fake white edge.
        rim_outline = self._blend_color(rim_color, base_color, 0.08)
        cv2.circle(overlay, (cx_ss, cy_ss), size_ss, rim_outline, max(1, ss), lineType=cv2.LINE_AA)
        cv2.ellipse(
            overlay,
            (cx_ss, int(cy_ss + size_ss * 0.16)),
            (int(size_ss * 0.70), int(size_ss * 0.30)),
            0,
            10,
            170,
            self._blend_color(rim_color, base_color, 0.12),
            max(1, int(size_ss * 0.04)),
            lineType=cv2.LINE_AA,
        )

        patch_final = cv2.resize(overlay, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_AREA)
        frame[y0:y1, x0:x1] = patch_final

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

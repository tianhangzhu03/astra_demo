"""Cached, antialiased blue sphere; compositing never blurs the camera image."""

from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=8)
def sphere_sprite(radius: int):
    pad = int(np.ceil(radius * 1.8))
    width = pad * 2 + 1
    ss = 3
    axis = (np.arange(width * ss, dtype=np.float32) + 0.5) / ss - pad - 0.5
    x, y = np.meshgrid(axis / radius, axis / radius)
    r2 = x * x + y * y
    mask = (r2 <= 1.0).astype(np.float32)
    z = np.sqrt(np.maximum(0.0, 1.0 - r2))
    light = np.array([-0.45, -0.55, 0.78], dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.maximum(0.0, x * light[0] + y * light[1] + z * light[2])
    halfway = light + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    halfway /= np.linalg.norm(halfway)
    spec = np.maximum(0.0, x * halfway[0] + y * halfway[1] + z * halfway[2])
    # Linear-light shading: a broad softbox reflection on satin blue resin.
    base = np.array([0.32, 0.13, 0.035], dtype=np.float32)  # BGR, linear
    linear = base * (0.25 + 0.72 * diffuse[..., None])
    linear += (0.16 * spec ** 42 + 0.018 * spec ** 8)[..., None] * np.array([0.90, 0.88, 0.82])
    linear = np.clip(linear, 0.0, 1.0)
    color = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055) * 255
    shadow = 0.19 * np.exp(-2.8 * (((x - 0.25) / 0.95) ** 2 + ((y - 0.57) / 0.75) ** 2))
    alpha = mask + shadow * (1 - mask)
    premult = color * mask[..., None] + np.array([32, 29, 26]) * (shadow * (1 - mask))[..., None]
    # Downsample premultiplied color and alpha together to avoid a pale fringe.
    premult = cv2.resize(premult.astype(np.float32), (width, width), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (width, width), interpolation=cv2.INTER_AREA)[..., None]
    return pad, premult, alpha


def draw_sphere(frame: np.ndarray, center, radius: int) -> None:
    radius = max(1, int(radius))
    pad, color, alpha = sphere_sprite(radius)
    x, y = int(center[0]), int(center[1])
    left, top = x - pad, y - pad
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(frame.shape[1], x + pad + 1), min(frame.shape[0], y + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return
    sx, sy = x0 - left, y0 - top
    h, w = y1 - y0, x1 - x0
    a = alpha[sy:sy + h, sx:sx + w]
    frame[y0:y1, x0:x1] = np.clip(
        color[sy:sy + h, sx:sx + w] + frame[y0:y1, x0:x1] * (1 - a), 0, 255
    ).astype(np.uint8)

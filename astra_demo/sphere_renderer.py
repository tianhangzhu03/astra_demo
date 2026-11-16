"""Cached pastel spheres with softbox highlights; camera pixels remain sharp."""

from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=8)
def sphere_sprite(radius: int, held: bool = False, pigment=None):
    pad = int(np.ceil(radius * 1.8))
    width = pad * 2 + 1
    ss = 3
    axis = (np.arange(width * ss, dtype=np.float32) + 0.5) / ss - pad - 0.5
    x, y = np.meshgrid(axis / radius, axis / radius)
    r2 = x * x + y * y
    mask = (r2 <= 1.0).astype(np.float32)
    z = np.sqrt(np.maximum(0.0, 1.0 - r2))
    light = np.array([-0.45, -0.55, 0.70], dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.maximum(0.0, x * light[0] + y * light[1] + z * light[2])
    halfway = light + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    halfway /= np.linalg.norm(halfway)
    spec = np.maximum(0.0, x * halfway[0] + y * halfway[1] + z * halfway[2])
    # OpenCV uses BGR. Pale sky blue / mint green retain the existing state colors.
    # Convert the chosen sRGB pigment to linear light before shading.
    srgb = np.array(pigment if pigment is not None else ([169, 222, 122] if held else [250, 195, 130]), dtype=np.float32) / 255
    base = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    linear = base * (0.44 + 0.64 * diffuse[..., None])
    # Broad reflection plus a smaller bright core give the surface a rounded gloss.
    linear += (0.20 * spec ** 24 + 0.10 * spec ** 85)[..., None] * np.array([1.0, .98, .94])
    # A restrained reflected rim lifts the lower edge without flattening the sphere.
    rim = (1 - z) ** 3 * (.3 + .7 * np.clip(y, 0, 1))
    linear += (0.055 * rim)[..., None] * np.array([1.0, .91, .80])
    linear = np.clip(linear, 0.0, 1.0)
    color = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055) * 255
    # Diffuse cast shadow and a small contact shadow establish depth on the video.
    shadow = 0.16 * np.exp(-2.8 * (((x - .20) / .95) ** 2 + ((y - .88) / .43) ** 2))
    shadow += .075 * np.exp(-3.5 * (((x - .06) / .63) ** 2 + ((y - 1.01) / .15) ** 2))
    alpha = mask + shadow * (1 - mask)
    premult = color * mask[..., None] + np.array([32, 29, 26]) * (shadow * (1 - mask))[..., None]
    # Downsample premultiplied color and alpha together to avoid a pale fringe.
    premult = cv2.resize(premult.astype(np.float32), (width, width), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (width, width), interpolation=cv2.INTER_AREA)[..., None]
    return pad, premult, alpha


def draw_sphere(frame: np.ndarray, center, radius: int, held: bool = False, *, pigment=None) -> None:
    radius = max(1, int(radius))
    pad, color, alpha = sphere_sprite(radius, held, pigment)
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

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class FrameBundle:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    timestamp_ms: int


class CameraSource:
    def __init__(self, fps: int = 30):
        self.fps = fps

    def start(self) -> None:
        raise NotImplementedError

    def read(self) -> Optional[FrameBundle]:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class AstraOpenNICameraSource(CameraSource):
    def __init__(self, fps: int = 30):
        super().__init__(fps=fps)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[FrameBundle] = None
        self._error: Optional[str] = None

        self._openni2 = None
        self._device = None
        self._color_stream = None
        self._depth_stream = None

    def start(self) -> None:
        try:
            from openni import openni2  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "openni Python package is required on Windows for Astra Pro Plus. "
                "Install OpenNI2 runtime + Python binding first."
            ) from exc

        self._openni2 = openni2
        self._openni2.initialize()
        self._device = self._openni2.Device.open_any()

        self._color_stream = self._device.create_color_stream()
        self._depth_stream = self._device.create_depth_stream()

        self._color_stream.start()
        self._depth_stream.start()

        # Validate RGB/depth mapping shape once at startup.
        color = self._read_color_frame()
        depth = self._read_depth_frame()
        if color is None or depth is None:
            raise RuntimeError("Failed to read initial frames from Astra streams.")
        if color.shape[:2] != depth.shape[:2]:
            raise RuntimeError(
                f"RGB/Depth size mismatch: rgb={color.shape[:2]} depth={depth.shape[:2]}. "
                "Please configure Astra SDK to provide aligned streams."
            )

        self._latest = FrameBundle(
            color_bgr=color,
            depth_mm=depth,
            timestamp_ms=int(time.time() * 1000),
        )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _read_color_frame(self) -> Optional[np.ndarray]:
        if self._color_stream is None:
            return None
        frame = self._color_stream.read_frame()
        raw = frame.get_buffer_as_uint8()
        arr = np.frombuffer(raw, dtype=np.uint8)
        h = frame.height
        w = frame.width
        if arr.size != h * w * 3:
            return None
        # Most OpenNI color buffers are RGB; convert to BGR for OpenCV.
        rgb = arr.reshape(h, w, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _read_depth_frame(self) -> Optional[np.ndarray]:
        if self._depth_stream is None:
            return None
        frame = self._depth_stream.read_frame()
        raw = frame.get_buffer_as_uint16()
        arr = np.frombuffer(raw, dtype=np.uint16)
        h = frame.height
        w = frame.width
        if arr.size != h * w:
            return None
        return arr.reshape(h, w).astype(np.uint16)

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / max(1, self.fps)
        while self._running:
            try:
                color = self._read_color_frame()
                depth = self._read_depth_frame()
                if color is None or depth is None:
                    continue

                if color.shape[:2] != depth.shape[:2]:
                    self._error = (
                        f"Runtime RGB/Depth mismatch: rgb={color.shape[:2]} depth={depth.shape[:2]}"
                    )
                    self._running = False
                    break

                with self._lock:
                    self._latest = FrameBundle(
                        color_bgr=color,
                        depth_mm=depth,
                        timestamp_ms=int(time.time() * 1000),
                    )
            except Exception as exc:
                self._error = f"Astra capture error: {exc.__class__.__name__}: {exc}"
                self._running = False
                break

            time.sleep(frame_interval)

    def read(self) -> Optional[FrameBundle]:
        with self._lock:
            return self._latest

    def get_error(self) -> Optional[str]:
        return self._error

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._color_stream is not None:
            try:
                self._color_stream.stop()
            except Exception:
                pass
        if self._depth_stream is not None:
            try:
                self._depth_stream.stop()
            except Exception:
                pass
        if self._openni2 is not None:
            try:
                self._openni2.unload()
            except Exception:
                pass


def create_camera_source(fps: int = 30) -> CameraSource:
    return AstraOpenNICameraSource(fps=fps)

"""Independent capture workers; slow cameras cannot block the operator controls."""

from dataclasses import dataclass
import os
import threading
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray
    sequence: int
    mono: float


class CameraStream:
    def __init__(self, index, *, fps=30, on_frame=None, backend=None):
        self.index, self.fps, self.on_frame = index, fps, on_frame
        self.backend = backend if backend is not None else (cv2.CAP_MSMF if os.name == 'nt' else cv2.CAP_ANY)
        self.latest = None
        self.error = ''
        self._stop = threading.Event()
        self._thread = None
        self.closed = False

    def start(self):
        self._thread = threading.Thread(target=self._run, name=f'physical-camera-{self.index}', daemon=True)
        self._thread.start()

    def _run(self):
        # The capture object is created, read and released exclusively on this thread.
        capture = None
        try:
            capture = cv2.VideoCapture(self.index, self.backend)
            if not capture.isOpened():
                raise RuntimeError(f'Camera {self.index} could not open; check index / other running demos')
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            sequence = 0
            while not self._stop.is_set():
                ok, image = capture.read()
                received = time.monotonic()
                if not ok or image is None:
                    self.error = f'Camera {self.index}: no new frame'
                    self._stop.wait(.05)
                    continue
                sequence += 1
                frame = CameraFrame(image, sequence, received)
                self.latest, self.error = frame, ''
                if self.on_frame:
                    self.on_frame(frame)
        except Exception as exc:
            self.error = str(exc)
        finally:
            if capture is not None:
                capture.release()
            self.closed = True

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self._thread is None or not self._thread.is_alive()


class SyntheticCamera:
    """Explicitly labelled offline input for preview and integration checks."""
    def __init__(self, index, *, fps=30, on_frame=None, **kwargs):
        self.index, self.fps, self.on_frame = index, fps, on_frame
        self.latest, self.error = None, ''
        self.closed = False
        self._sequence = 0
        self._next = 0.0

    def start(self):
        pass

    def poll(self):
        now = time.monotonic()
        if now < self._next:
            return
        self._next = now + 1 / self.fps
        self._sequence += 1
        width, height = (960, 540) if self.index == 0 else (640, 480)
        frame = np.full((height, width, 3), (225, 230, 234), np.uint8)
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), (214, 220, 225), 1)
        for y in range(0, height, 80):
            cv2.line(frame, (0, y), (width, y), (214, 220, 225), 1)
        x = int(width * .40 + 24 * np.sin(self._sequence / 20))
        y = int(height * .54)
        cv2.ellipse(frame, (x, y + 39), (70, 14), 0, 0, 360, (192, 198, 203), -1)
        cv2.rectangle(frame, (x - 58, y - 31), (x + 58, y + 31), (248, 249, 250), -1)
        cv2.line(frame, (x - 58, y - 10), (x + 58, y - 10), (183, 188, 193), 2)
        cv2.circle(frame, (x, y + 9), 3, (145, 180, 144), -1)
        cv2.putText(frame, 'SYNTHETIC PREVIEW', (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (92, 92, 92), 1, cv2.LINE_AA)
        self.latest = CameraFrame(frame, self._sequence, now)
        if self.on_frame:
            self.on_frame(self.latest)

    def close(self):
        self.closed = True
        return True

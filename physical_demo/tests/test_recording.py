import csv
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from physical_demo.cameras import CameraStream
from physical_demo.recording import Sample, VideoStream
from physical_demo.session import SessionClock


class RecordingTests(unittest.TestCase):
    def test_future_frame_is_never_used_for_a_past_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = SessionClock()
            initial = Sample(np.full((48, 64, 3), 40, np.uint8), time.monotonic(), 1)
            stream = VideoStream(Path(directory)/'top.mp4', 30, clock, initial)
            start = time.monotonic()
            future = Sample(np.full((48, 64, 3), 220, np.uint8), start+10, 2)
            stream.submit(future)
            stream.start(start)
            time.sleep(.11)
            stream.request_stop(time.monotonic())
            stream.close()
            self.assertFalse(stream.error)
            with stream.path.with_suffix('.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertGreater(len(rows), 1)
            self.assertEqual({r['sample_id'] for r in rows}, {'1'})
            cap = cv2.VideoCapture(str(stream.path))
            try:
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    self.assertLess(abs(image.mean()-40), 5)
            finally:
                cap.release()

    def test_resolution_change_is_reported_and_writer_is_finalized(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = SessionClock()
            stream = VideoStream(Path(directory)/'side.mp4', 30, clock,
                                 Sample(np.zeros((48, 64, 3), np.uint8), time.monotonic(), 1))
            stream.start(time.monotonic())
            stream.submit(Sample(np.zeros((60, 80, 3), np.uint8), time.monotonic(), 2))
            deadline = time.monotonic()+2
            while stream.active and time.monotonic() < deadline:
                time.sleep(.01)
            stream.request_stop(time.monotonic())
            stream.close()
            self.assertIn('resolution changed', stream.error)
            self.assertFalse(stream.active)

    def test_bounded_source_queue_reports_discarded_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            image = np.zeros((48, 64, 3), np.uint8)
            stream = VideoStream(Path(directory)/'top.mp4', 30, SessionClock(), Sample(image, time.monotonic(), 0))
            try:
                for i in range(1, 21):
                    stream.submit(Sample(image, time.monotonic(), i))
                self.assertEqual(len(stream._pending), 8)
                self.assertEqual(stream.dropped, 12)
            finally:
                stream.close()

    def test_capture_failure_releases_its_own_handle_without_fallback_camera(self):
        class Capture:
            released = False
            def isOpened(self):
                return False
            def release(self):
                self.released = True
        capture = Capture()
        with patch('physical_demo.cameras.cv2.VideoCapture', return_value=capture) as opened:
            camera = CameraStream(3)
            camera.start()
            self.assertTrue(camera.close())
        opened.assert_called_once_with(3, camera.backend)
        self.assertTrue(capture.released)
        self.assertIn('could not open', camera.error)
        self.assertIsNone(camera.latest)

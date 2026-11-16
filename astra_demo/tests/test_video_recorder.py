import csv
from pathlib import Path
import tempfile
import time
import unittest
import cv2
import numpy as np

from astra_demo.video_recorder import DualRecorder, StreamRecorder


class RecordingTests(unittest.TestCase):
    def test_two_independent_files_keep_time_and_can_stop_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DualRecorder(directory, 30)
            frames = {'top': np.full((64, 96, 3), (220, 30, 30), np.uint8),
                      'side': np.full((64, 96, 3), (30, 220, 30), np.uint8)}
            start = time.monotonic()
            try:
                recorder.toggle('both', frames, start, 100000)
                time.sleep(.16)
                recorder.submit(frames, {'top': 100160, 'side': 100162})
                time.sleep(.10)
                stopped = time.monotonic()
                recorder.toggle('top', frames, stopped, 100260)
                self.assertFalse(recorder.streams['top'].active)
                self.assertTrue(recorder.streams['side'].active)
                top_count = recorder.streams['top'].frames_written
                self.assertAlmostEqual(top_count / 30, stopped - start, delta=.07)
                time.sleep(.10)
            finally:
                recorder.close()
            self.assertGreater(recorder.streams['side'].frames_written, top_count)
            for name, stream in recorder.streams.items():
                self.assertFalse(stream.error, stream.error)
                cap = cv2.VideoCapture(str(stream.path))
                try:
                    self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), stream.frames_written)
                    ok, frame = cap.read()
                    self.assertTrue(ok)
                    np.testing.assert_allclose(frame.mean(axis=(0, 1)), frames[name][0, 0], atol=8)
                finally:
                    cap.release()
                with stream.path.with_suffix('.csv').open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), stream.frames_written)
                self.assertTrue(any(row['repeated_frame'] == '1' for row in rows))
            # Starting again creates a new segment without overwriting the first.
            recorder.toggle('top', frames, time.monotonic(), 200000)
            recorder.close()
            self.assertEqual(len(list(Path(directory).glob('top_*.mp4'))), 2)

    def test_missing_frame_and_invalid_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = StreamRecorder(directory, 'top')
            self.assertFalse(stream.start(None))
            self.assertFalse(stream.active)
            for fps in (0, -1, float('nan')):
                with self.assertRaises(ValueError):
                    StreamRecorder(directory, 'top', fps)

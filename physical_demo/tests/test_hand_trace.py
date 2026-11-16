import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

from physical_demo.cameras import CameraFrame
from physical_demo.config import Config
from physical_demo.hand_trace import palm_observation, PalmTraceProcessor, TraceHistory, HandTraceService
from physical_demo.palm_motion import HandObservation
from physical_demo.panel import Panel
from physical_demo.haptics import Actuation
from physical_demo.session import PilotSession


def observation(x=80., y=80.):
    palm = np.array([x, y])
    return HandObservation(palm, np.eye(2), 40., palm.copy(), None, None,
                           palm+[[-15, 15], [-18, -12], [0, -17], [15, -12], [18, 10]])


class Detector:
    def __init__(self, model=None):
        self.observation = observation()
        self.closed = False
    def observe(self, frame, mono):
        return self.observation
    def close(self):
        self.closed = True


class TraceTests(unittest.TestCase):
    def test_resolution_change_starts_a_separate_stroke(self):
        processor = PalmTraceProcessor(Detector(), inference_width=160)
        first = processor.process(CameraFrame(np.zeros((160,160,3),np.uint8),1,1),view='top')
        second = processor.process(CameraFrame(np.zeros((240,320,3),np.uint8),2,1.1),view='top')
        self.assertNotEqual(first['stroke_id'],second['stroke_id'])

    def test_source_pixel_coordinates_and_missing_data_start_a_new_stroke(self):
        detector = Detector()
        processor = PalmTraceProcessor(detector, inference_width=160)
        image = np.zeros((240, 320, 3), np.uint8)
        first = processor.process(CameraFrame(image, 1, 1.), view='top')
        self.assertEqual(first['raw_palm_x_px'], 160)
        self.assertEqual(first['display_palm_x_px'], 160)
        detector.observation = None
        missing = processor.process(CameraFrame(image, 2, 2.), view='top')
        self.assertEqual(missing['valid'], 0)
        self.assertIsNone(missing['display_palm_y_px'])
        detector.observation = observation(120, 60)
        next_row = processor.process(CameraFrame(image, 3, 2.1), view='top')
        self.assertGreater(next_row['stroke_id'], first['stroke_id'])
        self.assertAlmostEqual(next_row['display_palm_x_px'], 240)
        self.assertFalse(any('z_' in name for name in next_row))

    def test_palm_geometry_does_not_use_fingertip_grip_or_model_z(self):
        points = [SimpleNamespace(x=.5, y=.5, z=999999) for _ in range(21)]
        for i, x, y in [(0, .5, .7), (5, .35, .4), (9, .5, .35), (13, .6, .4), (17, .7, .5)]:
            points[i].x, points[i].y = x, y
        points[4].x = points[8].x = float('nan')
        obs = palm_observation(points, 320, 240)
        self.assertIsNotNone(obs)
        self.assertIsNone(obs.pinch_ratio)
        points[9].x = float('nan')
        self.assertIsNone(palm_observation(points, 320, 240))

    def test_history_retains_trail_but_hides_stale_current_point(self):
        row = PalmTraceProcessor(Detector()).process(CameraFrame(np.zeros((160, 160, 3), np.uint8), 1, 10), view='top')
        history = TraceHistory()
        history.add(row)
        self.assertIsNotNone(history.snapshot(10.1)['current'])
        self.assertIsNone(history.snapshot(11)['current'])
        self.assertEqual(len(history.snapshot(11)['points']), 1)

    def test_side_trace_mirrors_with_image_and_raw_input_is_unchanged(self):
        config = Config()
        panel = Panel(config)
        image = np.full((480, 640, 3), 220, np.uint8)
        source = image.copy()
        frame = CameraFrame(image, 1, 10)
        state = dict(settings=Actuation(), mono=10, traces={'side':dict(
            points=[(1, .25, .5, 10)], current=(.25, .5), status='Hand tracked')})
        canvas = panel.render(frame, frame, state)
        # Side image spans x=920..1416: mirrored x=.75 lands near 1291.
        np.testing.assert_array_equal(canvas[355:357,1290:1292], np.full((2,2,3),[134,90,139],np.uint8))
        np.testing.assert_array_equal(image, source)

    def test_service_preserves_rows_after_display_reset_and_closes_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(Config(), log_root=Path(directory), ble_enabled=False)
            session = PilotSession(config)
            detectors = []
            def factory(path):
                detector = Detector()
                detectors.append(detector)
                return detector
            service = HandTraceService(session, config, detector_factory=factory)
            try:
                for i in (1, 2):
                    frame = CameraFrame(np.zeros((160, 160, 3), np.uint8), i, time.monotonic())
                    service.submit(dict(top=frame, side=frame), '001')
                    deadline = time.monotonic()+2
                    while time.monotonic() < deadline:
                        service.consume()
                        if all(h.latest and h.latest['source_frame_id']==i for h in service.histories.values()):
                            break
                        time.sleep(.005)
                    self.assertTrue(all(h.latest['source_frame_id']==i for h in service.histories.values()))
                    if i == 1:
                        service.reset()
                        self.assertFalse(service.snapshot()['top']['points'])
            finally:
                service.close()
                session.close()
            self.assertTrue(all(d.closed for d in detectors))
            with (session.directory/'hand_trace.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            self.assertEqual({r['trace_epoch'] for r in rows}, {'0','1'})
            self.assertEqual({r['trial_id'] for r in rows}, {'001'})

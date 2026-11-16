"""Exercise the real main loop with supplied frames; no camera, window or BLE connection."""

import csv
import json
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from astra_demo import main as demo
from astra_demo.config import AstraDemoConfig
from astra_demo.operator_panel import OperatorPanel, initial_ball_position
from astra_demo.tracking import HandObservation
from astra_demo.virtual_prop import VirtualProp


class MainFlowTests(unittest.TestCase):
    def test_abc_records_only_between_n_and_end_even_with_idle_video_keys(self):
        cursor, panels = [-1], []
        class Capture:
            def __init__(self, top): self.top = top
            def read(self):
                if self.top: cursor[0] += 1
                if cursor[0] > 20: raise AssertionError('Synthetic Q must exit the loop')
                return True, np.full((480, 640, 3), 210, np.uint8)
            def release(self): pass
        class Panel(OperatorPanel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._key_down = None
                self._app_focused = lambda: True
                self.side_place_request, self.side_box_request = (240, 310), (430, 150)
                panels.append(self)
            def open(self): pass
            def show(self, *args, **kwargs):
                states.append((cursor[0], self.ui_record_active, any(self.record_active.values())))
        palm = np.array([125., 353.])
        obs = HandObservation(palm, np.eye(2), 80., palm-[0,40], .7, .05,
                              palm+[[-25,30],[-40,-30],[0,-40],[25,-30],[40,0]])
        # U/B cannot start idle recording, including after E. Q closes an active second trial.
        keys = {0:'u', 1:'b', 3:'n', 7:'e', 8:'u', 9:'b', 11:'n', 15:'q'}
        states = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            cfg = AstraDemoConfig(session_log_dir=directory, operator_panel_fps=100000,
                                  show_side_color_window=False)
            changes = {'AstraDemoConfig': lambda: cfg, 'enable_dpi_awareness': lambda: None,
                'open_uvc_camera': lambda camera, *a: Capture(camera == 1),
                'load_hands_api': lambda: (SimpleNamespace(Hands=lambda **k: SimpleNamespace(close=lambda: None)), None),
                'detect_hand': lambda *a, **k: (obs, None), 'OperatorPanel': Panel,
                'open_video_window': lambda *a: None, 'show_video_frame': lambda *a: None}
            for name, replacement in changes.items(): stack.enter_context(patch.object(demo, name, replacement))
            stack.enter_context(patch.object(sys, 'argv', ['demo','--scenario','abc','--no-ble','--top-camera','1','--side-camera','3']))
            stack.enter_context(patch.object(demo.cv2, 'waitKeyEx', lambda _: ord(keys[cursor[0]]) if cursor[0] in keys else -1))
            stack.enter_context(patch.object(demo.cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(patch.object(demo.cv2, 'destroyAllWindows'))
            demo.main()
            self.assertTrue(all(not ui and not cameras for i, ui, cameras in states if i < 4 or 8 <= i < 12))
            self.assertTrue(any(ui and cameras for i, ui, cameras in states if 4 <= i < 8))
            session = next(Path(directory).glob('experiment_*'))
            trials = json.loads((session/'trials.json').read_text())
            self.assertEqual([t['result'] for t in trials], ['incomplete', 'interrupted'])
            self.assertEqual(len(list((session/'videos').glob('*.mp4'))), 6)
            for trial in trials:
                for view in ('top', 'side', 'operator'):
                    path = session/trial['videos'][view][0]
                    with path.with_suffix('.csv').open() as file: rows = list(csv.DictReader(file))
                    self.assertTrue(rows)
                    self.assertEqual({r['trial_id'] for r in rows}, {trial['trial_id']})
            with (session/'trajectory.csv').open() as file:
                self.assertTrue(all(r['trial_id'] for r in csv.DictReader(file)))
            self.assertIsNone(panels[0].controller._thread)

    def test_restart_held_and_finished_trials_preserves_cameras_and_side_origin(self):
        cursor = [-1]
        captures, detectors, snapshots, panels = [], [], [], []
        ball = np.array(initial_ball_position(640, 480))
        side_origin = (145., 265.)

        class Capture:
            def __init__(self, top):
                self.top, self.released = top, False
                captures.append(self)

            def read(self):
                if self.top:
                    cursor[0] += 1
                return True, np.full((480, 640, 3), 210, np.uint8)

            def release(self):
                self.released = True

        def hands(**kwargs):
            detector = SimpleNamespace(top=not detectors, close=lambda: None)
            detectors.append(detector)
            return detector

        def detect(detector, frame, scale, **kwargs):
            i = cursor[0]
            if 19 <= i <= 22:
                return None, None
            ratio = .7 if 4 <= i <= 8 or 28 <= i <= 32 else .2
            anchor = ball + ([25., 0.] if 14 <= i <= 18 else [0., 0.])
            palm = anchor + [0, 40]
            return HandObservation(palm, np.eye(2), 80, anchor, ratio, .05,
                palm + [[-25, 30], [-40, -30], [0, -40], [25, -30], [40, 0]]), None

        class Panel(OperatorPanel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._key_down = None
                self.side_place_request = side_origin
                panels.append(self)

            def open(self):
                pass

            def show(self, *args, **kwargs):
                snapshots.append((cursor[0], self.held, self.restart_pending))

        keys = {0: 'n', 2: 'w', 3: 'g', 18: '0', 20: 'g',
                25: 'w', 27: 'g', 41: 'r', 42: '0', 46: '0', 52: 'q'}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            cfg = AstraDemoConfig(session_log_dir=directory, operator_panel_fps=100000)
            changes = {'AstraDemoConfig': lambda: cfg, 'enable_dpi_awareness': lambda: None,
                'open_uvc_camera': lambda camera, *args: Capture(camera == 1),
                'load_hands_api': lambda: (SimpleNamespace(Hands=hands), None),
                'detect_hand': detect, 'open_video_window': lambda *args: None,
                'show_video_frame': lambda *args: None, 'OperatorPanel': Panel}
            for name, replacement in changes.items():
                stack.enter_context(patch.object(demo, name, replacement))
            stack.enter_context(patch.object(sys, 'argv', ['demo', '--no-ble', '--linked-views',
                '--top-camera', '1', '--side-camera', '3']))
            stack.enter_context(patch.object(demo.cv2, 'waitKeyEx',
                lambda _: ord(keys[cursor[0]]) if cursor[0] in keys else -1))
            stack.enter_context(patch.object(demo.cv2, 'setMouseCallback'))
            stack.enter_context(patch.object(demo.cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(patch.object(demo.cv2, 'destroyAllWindows'))
            demo.main()
            session = next(Path(directory).glob('experiment_*'))
            trials = json.loads((session / 'trials.json').read_text())
            self.assertEqual([t['result'] for t in trials], ['reset', 'assisted', 'reset', 'interrupted'])
            self.assertTrue(trials[0]['grasp_t_s'] is not None)
            self.assertTrue(trials[1]['grasp_t_s'] > trials[0]['end_t_s'])
            self.assertTrue(all(t['grasp_t_s'] is None and t['start_t_s'] is None for t in trials[2:]))
            self.assertEqual([t['side_initial_ball_px'] for t in trials], [list(side_origin)] * 4)
            self.assertTrue(any(i == 18 and held for i, held, pending in snapshots))
            self.assertTrue(all(not held and pending for i, held, pending in snapshots if 19 <= i <= 22))
            self.assertTrue(all(not held and not pending for i, held, pending in snapshots if 23 <= i <= 32))
            with (session / 'trajectory.csv').open() as file:
                trajectory = list(csv.DictReader(file))
            for trial in trials:
                first = next(r for r in trajectory if r['trial_id'] == trial['trial_id'])
                np.testing.assert_allclose([float(first['ball_x_px']), float(first['ball_y_px'])], ball)
                np.testing.assert_allclose([float(first['side_ball_x_px']), float(first['side_ball_y_px'])], side_origin)
                self.assertEqual(first['held'], '0')
                self.assertEqual(first['grasp_status'], 'WAIT CUE [G]')
                for name in ('top', 'side'):
                    self.assertEqual(len(trial['videos'][name]), 1)
                    path = session / trial['videos'][name][0]
                    capture = cv2.VideoCapture(str(path))
                    try:
                        self.assertTrue(capture.read()[0], str(path))
                    finally:
                        capture.release()
                    with path.with_suffix('.csv').open() as file:
                        timings = list(csv.DictReader(file))
                    self.assertTrue(timings)
                    self.assertEqual({r['trial_id'] for r in timings}, {trial['trial_id']})
            self.assertEqual(len(list((session/'videos').glob('*.mp4'))), 12)
            # Each trial owns its UI segment; no idle/reset-wait frames are recorded.
            ui_paths = {t['videos']['operator'][0] for t in trials}
            self.assertEqual(len(ui_paths), 4)
            for trial in trials:
                with (session/trial['videos']['operator'][0]).with_suffix('.csv').open() as file:
                    ui_timings = list(csv.DictReader(file))
                self.assertEqual({r['trial_id'] for r in ui_timings}, {trial['trial_id']})
                self.assertTrue(all(r['top_capture_t_s'] and r['side_capture_t_s'] for r in ui_timings))
                self.assertGreaterEqual(float(ui_timings[0]['capture_t_s']), trial['setup_t_s'])
            with (session / 'events.csv').open() as file:
                events = list(csv.DictReader(file))
            commands = [r for r in events if r['event'] == 'command_simulated']
            self.assertEqual([(r['trial_id'], r['cue']) for r in commands],
                [('001', 'Forward'), ('001', 'Vibration'), ('002', 'Forward'), ('002', 'Vibration')])
            self.assertEqual(len(captures), 2)
            self.assertTrue(all(c.released for c in captures))
            self.assertIsNone(panels[0].controller._thread)

    def test_prompt_proximity_gesture_and_release_through_main_loop(self):
        self.run_grasp_flow(wearable=False)

    def test_wearable_flow_with_bad_tips_and_short_side_occlusion(self):
        self.run_grasp_flow(wearable=True)

    def run_grasp_flow(self, *, wearable):
        cursor = [-1]
        captures, detectors, snapshots = [], [], []
        ball = np.array(initial_ball_position(640, 480))

        class Capture:
            def __init__(self, top):
                self.top, self.released = top, False
                captures.append(self)

            def read(self):
                if self.top:
                    cursor[0] += 1
                return True, np.zeros((480, 640, 3), np.uint8)

            def release(self):
                self.released = True

        def hands(**kwargs):
            detector = SimpleNamespace(top=not detectors, closed=False)
            detector.close = lambda: setattr(detector, 'closed', True)
            detectors.append(detector)
            return detector

        def detect(detector, frame, scale, **kwargs):
            i = cursor[0]
            if i == 19:
                return None, None  # Tracking loss while held must not release.
            ratio = .7 if i in (6, 7, 12, 13, 14, 21, 22, 23) else .2
            shift = np.array([220., 0.]) if 6 <= i <= 9 else np.zeros(2)
            if i == 18:
                shift = np.array([30., 0.])
            anchor = ball + shift
            palm = anchor + [0, 40]
            obs = HandObservation(palm, np.eye(2), 80, anchor, ratio, .05,
                                  palm + [[-25, 30], [-40, -30], [0, -40], [25, -30], [40, 0]])
            if wearable:
                obs.pinch_ratio = None  # Taped tips cannot supply a pinch decision.
                obs.grip_xy = anchor.copy()
                obs.grip_openness = None if detector.top or i == 15 else ratio
            return obs, None

        class Panel(OperatorPanel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._key_down = None

            def open(self):
                pass

            def show(self, *args, **kwargs):
                pass

        class Prop(VirtualProp):
            def update(self, *args, **kwargs):
                super().update(*args, **kwargs)
                snapshots.append((cursor[0], self.state.value, self.pos_xy))

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            cfg = AstraDemoConfig(session_log_dir=directory)
            changes = {
                'AstraDemoConfig': lambda: cfg,
                'enable_dpi_awareness': lambda: None,
                'open_uvc_camera': lambda camera, *args: Capture(camera == 1),
                'load_hands_api': lambda: (SimpleNamespace(Hands=hands), None),
                'detect_hand': detect,
                'open_video_window': lambda *args: None,
                'show_video_frame': lambda *args: None,
                'OperatorPanel': Panel,
                'VirtualProp': Prop,
            }
            for name, replacement in changes.items():
                stack.enter_context(patch.object(demo, name, replacement))
            stack.enter_context(patch.object(sys, 'argv', ['demo', '--no-ble', '--top-camera', '1', '--side-camera', '3',
                                                          '--grasp-mode', 'wearable' if wearable else 'pinch']))
            stack.enter_context(patch.object(demo.cv2, 'waitKeyEx',
                lambda _: {3: ord('g'), 19: ord('r'), 27: ord('q')}.get(cursor[0], -1)))
            stack.enter_context(patch.object(demo.cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(patch.object(demo.cv2, 'destroyAllWindows'))
            demo.main()
            with next(Path(directory).glob('grab_log_*.csv')).open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            with next(Path(directory).glob('experiment_*/trajectory.csv')).open() as file:
                trajectory = list(csv.DictReader(file))
            self.assertTrue(any(r['grasp_status'] == 'HELD' for r in trajectory))
            if wearable:
                self.assertTrue(any(r['grip_source'] == 'side' and r['held'] == '1' for r in trajectory))
                self.assertTrue(all(r['pinch_ratio'] == '' for r in trajectory))
            self.assertTrue(all(state == 'IDLE' and np.allclose(point, ball)
                                for i, state, point in snapshots if i < 17))
            self.assertTrue(all(state == 'HELD' for i, state, _ in snapshots if 17 <= i <= 19))
            self.assertTrue(all(state == 'IDLE' for i, state, _ in snapshots if i >= 20))
            self.assertTrue(all(capture.released for capture in captures))
            self.assertTrue(all(detector.closed for detector in detectors))
            self.assertFalse(cfg.ble_enabled)

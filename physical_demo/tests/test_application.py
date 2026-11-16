import ast
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from physical_demo.cameras import CameraFrame
from physical_demo.config import Config, PACKAGE_DIR
from physical_demo.haptics import ManualBleController
from physical_demo.main import Pilot, run, validate_log_root
from physical_demo.panel import Panel
from physical_demo.session import PilotSession


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = replace(Config(), log_root=Path(self.temp.name))
        self.session = PilotSession(self.config)
        self.addCleanup(self.session.close)
        self.controller = ManualBleController(True, 'test', 'test')
        self.group = SimpleNamespace(active=True, error='', segment_id='segment_001', trial_id='', streams={})
        self.pilot = Pilot(self.config, self.session, self.controller, self.group)
        frame = CameraFrame(np.zeros((60, 80, 3), np.uint8), 1, time.monotonic())
        self.pilot.update(frame, frame)
        self.pilot.action('new_trial')

    def test_manual_annotations_and_undo_never_submit_or_stop_haptics(self):
        with patch.object(self.controller, 'submit') as submit, patch.object(self.controller, 'stop_output') as stop:
            self.pilot.action('mark_grasp')
            self.pilot.action('mark_release')
            self.pilot.action('undo')
        submit.assert_not_called()
        stop.assert_not_called()
        self.assertEqual(self.session.object_state, 'GRASPED')
        self.assertFalse(self.controller._pending)

    def test_g_remains_one_prompt_and_manual_grasp_does_not_queue_confirmation(self):
        self.controller.running = self.controller.is_connected = True
        self.pilot.action('voltage_up')
        self.pilot.action('grasp_cue')
        pending = self.controller._pending
        self.pilot.action('mark_grasp')
        self.pilot.action('mark_release')
        self.assertIs(self.controller._pending, pending)
        self.assertEqual(pending[-1].role, 'grab_prompt')
        with (self.session.directory/'events.csv').open() as file:
            events = list(csv.DictReader(file))
        requested = [e for e in events if e['event'] == 'command_requested']
        self.assertEqual(len(requested), 1)
        self.assertEqual(requested[0]['role'], 'grab_prompt')

    def test_reset_clears_trial_and_pending_output_but_keeps_video_segment(self):
        self.controller.running = self.controller.is_connected = True
        self.pilot.action('voltage_up')
        self.pilot.action('forward')
        self.assertIsNotNone(self.controller._pending)
        self.pilot.action('reset')
        self.assertIsNone(self.controller._pending)
        self.assertIsNone(self.session.active)
        self.assertEqual(self.session.trials[0]['status'], 'reset')
        self.assertTrue(self.group.active)
        self.assertEqual(self.group.segment_id, 'segment_001')
        self.assertFalse(self.pilot.start_pending)
        self.assertEqual(self.group.trial_id, '')

    def test_new_trial_rejects_stale_frames_without_hand_detection(self):
        self.pilot.action('end_trial')
        stale = CameraFrame(np.zeros((60, 80, 3), np.uint8), 2, time.monotonic()-5)
        self.pilot.update(stale, stale)
        self.pilot.action('new_trial')
        self.assertIsNone(self.session.active)
        self.assertIn('fresh', self.pilot.message)

    def test_normal_intent_waits_for_successful_write_not_request_or_rejection(self):
        self.pilot.action('normal')
        self.assertEqual(self.pilot.next_normal, 'lower')
        self.controller.running = self.controller.is_connected = True
        self.pilot.action('normal')  # 0 mV rejected.
        self.assertEqual(self.pilot.next_normal, 'lower')
        self.pilot.action('voltage_up')
        self.pilot.action('normal')
        receipt = self.controller._pending[-1]
        self.assertEqual(receipt.role, 'normal_lower')
        self.assertEqual(self.pilot.next_normal, 'lower')
        self.controller._emit('command_sent', receipt)
        self.pilot.drain_haptics()
        self.assertEqual(self.pilot.next_normal, 'lift')
        self.controller._emit('command_completed', receipt)
        self.pilot.drain_haptics()
        self.assertEqual(self.pilot.next_normal, 'lift')

    def test_normal_phase_does_not_advance_after_cancelled_pending_cue(self):
        self.controller.running = self.controller.is_connected = True
        self.pilot.action('voltage_up')
        self.pilot.action('normal')
        receipt = self.controller._pending[-1]
        self.pilot.action('reset')
        self.controller._emit('command_sent', receipt)  # Late receipt from the previous trial.
        self.pilot.drain_haptics()
        self.assertEqual(self.pilot.next_normal, 'lower')

    def test_normal_override_changes_only_meaning_and_manual_markers_do_not_advance(self):
        with patch.object(self.controller, 'submit') as submit:
            self.pilot.action('normal_next')
            self.pilot.action('mark_grasp')
            self.pilot.action('mark_release')
        submit.assert_not_called()
        self.assertEqual(self.pilot.next_normal, 'lift')

    def test_simulated_normals_alternate_using_the_same_carrier(self):
        self.controller.enabled = False
        self.pilot.config = replace(self.config, ble_enabled=False)
        roles = []
        for expected in ('lift', 'lower', 'lift'):
            self.pilot.action('normal')
            self.assertEqual(self.pilot.next_normal, expected)
        with (self.session.directory/'events.csv').open() as file:
            commands = [row for row in csv.DictReader(file) if row['event'] == 'command_simulated']
        self.assertEqual([row['cue'] for row in commands], ['Normal force']*3)
        self.assertEqual([row['role'] for row in commands], ['normal_lower', 'normal_lift', 'normal_lower'])
        self.assertEqual([row['frequency_hz'] for row in commands], ['0']*3)

    def test_keyboard_repeat_and_resized_button_mapping(self):
        panel = Panel(self.config)
        down = {ord('G')}
        panel._key_down = lambda key: key in down
        panel.handle_key(ord('g'))
        panel.handle_key(ord('g'))
        self.assertEqual(list(panel.actions), ['grasp_cue'])
        down.clear()
        panel.handle_key(-1)
        down.add(ord('G'))
        panel.handle_key(ord('g'))
        self.assertEqual(list(panel.actions), ['grasp_cue', 'grasp_cue'])
        panel.actions.clear()
        panel.buttons = [((100, 200, 200, 240), 'mark_grasp')]
        panel.transform = (10, 20, .5, .5)
        panel.click(cv2.EVENT_LBUTTONDOWN, 85, 130)
        self.assertEqual(list(panel.actions), ['mark_grasp'])
        panel.click(cv2.EVENT_LBUTTONDOWN, 5, 5)
        self.assertEqual(len(panel.actions), 1)

    def test_recording_failure_ends_trial_and_stops_output(self):
        self.group.error = 'side: encoder failed'
        self.group.stop = lambda reason: setattr(self.group, 'streams', {})
        with patch.object(self.controller, 'stop_output') as stop:
            self.pilot.update(*self.pilot.frames.values())
        stop.assert_called_once()
        self.assertEqual(self.session.trials[0]['status'], 'recording_failed')
        self.assertFalse(self.pilot.want_record)


class IsolationTests(unittest.TestCase):
    def test_no_import_of_virtual_demo_or_depth_libraries(self):
        for path in PACKAGE_DIR.glob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                names = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                         [entry.name for entry in node.names] if isinstance(node, ast.Import) else [])
                self.assertFalse(any(n.split('.')[0] in ('astra_demo', 'openni') for n in names), str(path))

    def test_existing_demo_directories_cannot_be_output_roots(self):
        for path in (PACKAGE_DIR.parent, PACKAGE_DIR.parent/'astra_demo'/'runs',
                     PACKAGE_DIR.parent/'session_logs'/'physical', PACKAGE_DIR.parent/'output'/'new'):
            with self.assertRaises(ValueError):
                validate_log_root(path)
        self.assertEqual(validate_log_root(PACKAGE_DIR/'runs'), PACKAGE_DIR/'runs')

    def test_synthetic_and_headless_modes_cannot_open_ble(self):
        with self.assertRaises(ValueError):
            run(Config(), synthetic=True, headless=True, seconds=1)
        with self.assertRaises(ValueError):
            run(replace(Config(), ble_enabled=False), headless=True, seconds=1)


class IntegrationTests(unittest.TestCase):
    def test_recording_is_continuous_across_end_reset_and_annotations_then_resumes_new_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(Config(), log_root=Path(directory), ble_enabled=False, ui_fps=20)
            actions = ((.10, 'new_trial'), (.20, 'forward'), (.35, 'grasp_cue'),
                       (.45, 'mark_grasp'), (.60, 'mark_release'), (.75, 'end_trial'),
                       (.85, 'new_trial'), (.95, 'reset'), (1.10, 'record'),
                       (1.40, 'record'), (1.60, 'new_trial'), (1.80, 'forward'),
                       (1.95, 'mark_grasp'), (2.10, 'mark_release'))
            # Any accidental physical camera access would fail the test.
            with patch('physical_demo.cameras.cv2.VideoCapture', side_effect=AssertionError('Real camera opened')):
                result = run(config, synthetic=True, headless=True, seconds=2.4, actions=actions)
            metadata = json.loads((result/'session.json').read_text())
            trials = json.loads((result/'trials.json').read_text())
            self.assertTrue(metadata['synthetic'])
            self.assertFalse(metadata['haptics']['enabled'])
            self.assertEqual([t['status'] for t in trials], ['manually_released', 'reset', 'manually_released'])
            self.assertEqual(trials[0]['segments'], trials[1]['segments'])
            self.assertNotEqual(trials[1]['segments'], trials[2]['segments'])
            self.assertEqual(len(metadata['recording']['segments']), 2)
            self.assertEqual(len(list(result.rglob('*.mp4'))), 6)
            for segment in metadata['recording']['segments']:
                self.assertEqual(segment['status'], 'closed')
                origins, counts = [], []
                for name, video in segment['videos'].items():
                    self.assertTrue(video['finalized'])
                    self.assertFalse(video['error'])
                    with (result/video['timestamps']).open() as file:
                        rows = list(csv.DictReader(file))
                    self.assertGreater(len(rows), 0)
                    origins.append(float(rows[0]['video_t_s']))
                    self.assertTrue(all(float(r['capture_t_s']) <= float(r['video_t_s'])+1e-9 for r in rows))
                    self.assertEqual(sum(int(r['repeated_frame']) for r in rows), video['repeated_frames'])
                    self.assertEqual(int(rows[0]['frame_index']), 0)
                    if name == 'operator':
                        self.assertTrue(all(r['top_frame_id'] and r['side_frame_id'] for r in rows))
                    cap = cv2.VideoCapture(str(result/video['file']))
                    count = 0
                    try:
                        while True:
                            ok, image = cap.read()
                            if not ok:
                                break
                            count += 1
                            if name != 'operator':
                                # Clean camera data remains the synthetic input, not a UI / guide overlay.
                                np.testing.assert_allclose(image[20:30, 20:30].mean(axis=(0, 1)), [225, 230, 234], atol=7)
                    finally:
                        cap.release()
                    self.assertEqual(count, len(rows))
                    self.assertEqual(count, video['frames'])
                    counts.append(count)
                self.assertEqual(len(set(origins)), 1)
                self.assertEqual(len(set(counts)), 1)
            with (result/'events.csv').open() as file:
                events = list(csv.DictReader(file))
            self.assertFalse(any(e['event'] == 'command_sent' for e in events))
            self.assertEqual(sum(e['event'] == 'command_simulated' for e in events), 3)
            self.assertTrue(all(t['timing_source'] == 'simulated_direction' for t in (trials[0], trials[2])))
            self.assertTrue(all('rmse' not in key and 'position' not in key for key in trials[0]))

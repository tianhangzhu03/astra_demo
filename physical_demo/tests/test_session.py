import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from physical_demo.config import Config
from physical_demo.session import PilotSession


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = PilotSession(replace(Config(), log_root=Path(self.temp.name), ble_enabled=False))
        self.addCleanup(self.session.close)
        self.origin = self.session.clock.origin
        self.session.start_trial('segment_001', mono=self.origin+1)

    def send(self, event, cue, at, trial='001'):
        self.session.ingest_haptic(dict(event=event, cue=cue, mono=self.origin+at,
                                       trial_id=trial, command_id='C00001'))

    def test_only_sent_directions_start_timing(self):
        for event, cue in [('command_requested', 'Forward'), ('command_rejected', 'Left'),
                           ('command_sent', 'Vibration'), ('command_sent', 'Normal force')]:
            self.send(event, cue, 2)
        self.assertIsNone(self.session.active['first_direction_t_s'])
        self.send('command_sent', 'Forward', 3)
        self.send('command_sent', 'Left', 4)
        self.assertEqual(self.session.active['first_direction_t_s'], 3)
        self.assertEqual(self.session.active['timing_source'], 'ble_write')

    def test_manual_markers_and_undo_preserve_original_events(self):
        self.send('command_simulated', 'Forward', 2)
        self.session.mark('grasped', mono=self.origin+3, evidence={'top_frame_id': 100})
        self.session.mark('released', mono=self.origin+4)
        first_release_id = self.session.active['markers'][-1]['marker_id']
        self.session.undo(mono=self.origin+5)
        self.assertEqual(self.session.object_state, 'GRASPED')
        self.assertIsNone(self.session.active['manual_release_t_s'])
        self.session.mark('released', mono=self.origin+6)
        self.session.finish_trial(mono=self.origin+7)
        trial = self.session.trials[0]
        self.assertEqual(trial['status'], 'manually_released')
        self.assertEqual(trial['direction_to_release_s'], 4)
        self.assertEqual(trial['manual_hold_s'], 3)
        self.assertEqual(trial['timing_source'], 'simulated_direction')
        self.assertEqual(trial['markers'][0]['evidence']['top_frame_id'], 100)
        with (self.session.directory/'events.csv').open() as file:
            events = list(csv.DictReader(file))
        original = [e for e in events if e['event'] == 'manual_released' and e['marker_id'] == first_release_id]
        self.assertEqual(len(original), 1)
        self.assertTrue(any(e['event'] == 'manual_marker_undone' for e in events))
        self.assertFalse(any('success' in e['event'] for e in events))

    def test_release_without_grasp_or_direction_is_flagged_not_scored(self):
        self.assertTrue(self.session.mark('released', mono=self.origin+3)[0])
        self.session.finish_trial(mono=self.origin+4)
        trial = self.session.trials[0]
        self.assertEqual(set(trial['flags']), {'grasp_not_marked', 'direction_not_sent'})
        self.assertIsNone(trial['manual_hold_s'])
        self.assertIsNone(trial['direction_to_release_s'])
        self.assertNotIn('success', trial)

    def test_active_trial_and_duplicate_markers_are_not_silently_overwritten(self):
        self.assertFalse(self.session.start_trial('segment_001')[0])
        self.session.mark('grasped')
        self.assertFalse(self.session.mark('grasped')[0])
        self.session.mark('released')
        self.assertFalse(self.session.mark('released')[0])
        self.assertTrue(self.session.start_trial('segment_001')[0])
        self.assertEqual(len(self.session.trials), 2)
        self.assertEqual(self.session.trials[0]['status'], 'manually_released')

    def test_delayed_ble_event_is_attributed_to_its_original_trial(self):
        self.session.finish_trial('reset', mono=self.origin+4)
        self.session.start_trial('segment_001', mono=self.origin+5)
        self.send('command_sent', 'Left', 3)
        self.assertEqual(self.session.trials[0]['first_direction_t_s'], 3)
        self.assertIsNone(self.session.active['first_direction_t_s'])
        self.send('command_sent', 'Forward', 6)
        self.assertIsNone(self.session.active['first_direction_t_s'])

    def test_close_keeps_incomplete_trial_and_unknown_geometry(self):
        self.session.close()
        trial = json.loads((self.session.directory/'trials.json').read_text())[0]
        meta = json.loads((self.session.directory/'session.json').read_text())
        self.assertEqual(trial['status'], 'interrupted')
        self.assertEqual(meta['object']['name'], 'AirPods case')
        self.assertEqual(meta['object']['geometry'], 'unspecified')
        self.assertFalse(any(meta['measurements'].values()))
        self.assertFalse(meta['haptics']['automatic_confirmation'])

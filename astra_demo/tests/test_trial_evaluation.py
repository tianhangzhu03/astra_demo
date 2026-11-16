"""Known-answer metrics and a full synthetic trial through the real recording loop."""

import csv
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from astra_demo import main as demo
from astra_demo.config import AstraDemoConfig
from astra_demo.experiment import (Experiment, SessionClock, reference_route, point_path_distance,
                                  rounded_reference_path, reference_path)
from astra_demo.manual_control import ManualBleController, Actuation, Cue, CommandReceipt
from astra_demo.operator_panel import OperatorPanel, workspace_bounds
from astra_demo.tracking import HandObservation
from astra_demo.trial_report import accuracy_metrics, command_rows, generate_report, read_rows
from astra_demo.video_recorder import DualRecorder


ROUTE = {'start': [100., 100.], 'turn': [100., 20.], 'pickup': [20., 20.], 'drop': [100., 100.]}


class MetricTests(unittest.TestCase):
    def test_rounded_reference_preserves_endpoints_and_follows_tangent_arc(self):
        points=np.array(rounded_reference_path([[0,100],[0,0],[100,0]],20))
        np.testing.assert_allclose(points[[0,-1]],[[0,100],[100,0]])
        np.testing.assert_allclose(points[[1,-2]],[[0,20],[20,0]],atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(points[1:-1]-[20,20],axis=1),20)
        self.assertGreater(point_path_distance([0,0],points),8.)
        self.assertLess(point_path_distance([20-20/np.sqrt(2)]*2,points),.03)
        for custom in (None, {'start':[.9,.9],'turn':[.9,.2],'pickup':[.1,.2],'drop':[.5,.9]}):
            route=reference_route(1280,720,workspace_bounds(1280,720),custom)
            np.testing.assert_allclose(reference_path(route)[0],route['start'])
            np.testing.assert_allclose(reference_path(route)[-1],route['pickup'])
            np.testing.assert_allclose(reference_path(route,True)[-1],route['drop'])
            if custom is None:
                np.testing.assert_allclose(reference_path(route),reference_path(route,True)[::-1])

    def test_rounded_reference_is_frozen_for_accuracy_and_legacy_stays_unchanged(self):
        route=reference_route(640,480,workspace_bounds(640,480))
        with tempfile.TemporaryDirectory() as folder:
            ex=Experiment(Path(folder)/'session','P01','session',clock=SessionClock(100,1000000))
            try:
                trial=ex.begin(route,None,None,[640,480],mono=100)
                anchor=np.array(route['outbound_path'][5])
                obs=SimpleNamespace(palm=anchor+[0,30],pinch_xy=anchor,pinch_ratio=.7)
                ex.sample(mono=101,top_mono=101,side_mono=101,observation=obs,side_valid=True,
                          tracked=None,ball=route['pickup'],held=False,pinch_ratio=.7,
                          tracking_status='Detected',recordings={'top':True,'side':True})
                np.testing.assert_allclose(trial['palm_outbound'],np.array(route['outbound_path'])+[0,30])
                self.assertLess(point_path_distance(obs.palm,trial['palm_outbound']),1e-8)
                self.assertGreater(point_path_distance(obs.palm,np.array([route[k] for k in ('start','turn','pickup')])+[0,30]),.1)
                trial['start_t_s']=.5
                ex.finish('incomplete',mono=102)
            finally: ex.close()
            self.assertLess(accuracy_metrics(trial,read_rows(ex.directory/'trajectory.csv'))['trajectory_rmse_px'],1e-8)
            self.assertEqual(reference_path(ROUTE),[ROUTE[k] for k in ('start','turn','pickup')])

    def test_clock_uses_one_anchor_not_changing_wall_clock(self):
        clock = SessionClock(100., 1_000_000.)
        with patch('time.time_ns', return_value=0):
            self.assertEqual(clock.seconds(102.25), 2.25)
            self.assertEqual(clock.unix_ms(102.25), 1_002_250.)

    def test_known_geometry_and_missing_frames_do_not_look_perfect(self):
        trial = {'trial_id': '001', 'start_t_s': 0., 'end_t_s': 1., 'grasp_t_s': None,
                 'palm_outbound': [[0, 0], [100, 0]], 'palm_return': [[100, 0], [0, 0]]}
        rows = [dict(trial_id='001', top_capture_t_s=t, top_valid=valid, palm_x_px=x, palm_y_px=5,
                     top_recording=1, side_recording=1) for t, valid, x in [
                         (0., 1, 0), (.1, 1, 10), (.2, 0, ''), (.3, 1, 30), (1., 1, 100)]]
        result = accuracy_metrics(trial, rows)
        self.assertEqual(result['trajectory_rmse_px'], 5.)
        self.assertEqual(result['tracking_valid_fraction'], .8)
        self.assertAlmostEqual(result['observed_time_s'], .1)
        self.assertEqual(result['observed_path_length_px'], 10.)
        self.assertEqual(point_path_distance([13, 4], [[0, 0], [10, 0]]), 5.)
        for row in rows: row['top_valid'] = 0
        self.assertIsNone(accuracy_metrics(trial, rows)['trajectory_rmse_px'])

    def test_trial_gates_timing_and_assisted_or_unknown_results(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = Experiment(Path(folder)/'session', 'P01', 'session', clock=SessionClock(100, 1_000_000))
            controller = ManualBleController(False, '', '')
            try:
                for mode, expected in [('auto', 'success'), ('manual', 'assisted'), ('lost', 'unverified_position')]:
                    experiment.begin(ROUTE, [100, 130], [100, 100], [640, 480], mono=100)
                    controller.trial_id = experiment.trial_id
                    with patch('time.monotonic', return_value=102):
                        controller.simulate(Cue.FORWARD, Actuation(1200, .3))
                    controller.trial_id = 'different'  # Queued events retain the original trial.
                    experiment.consume_commands(controller)
                    self.assertEqual(experiment.active['start_t_s'], 2.)
                    self.assertEqual(experiment.active['palm_outbound'][0], [100., 130.])
                    experiment.grasp(104)
                    trial = experiment.release([100, 100], manual=mode == 'manual',
                                                position_valid=mode != 'lost', mono=109)
                    self.assertEqual(trial['result'], expected)
                    self.assertEqual(trial['completion_time_s'], 7. if mode == 'auto' else None)
                experiment.begin(ROUTE, [100, 130], [100, 100], [640, 480], mono=110)
                experiment.finish('incomplete', mono=112)
            finally:
                experiment.close()
            reports = generate_report(experiment.directory, figures=False)
            summary = json.loads((reports/'summary.json').read_text())
            self.assertEqual(summary['groups']['vision_only']['completed'], 1)
            self.assertEqual(len(read_rows(reports/'commands.csv')), 3)

    def test_start_marker_is_only_a_guide_and_reference_is_not_fitted_to_hand(self):
        with tempfile.TemporaryDirectory() as folder:
            ex = Experiment(Path(folder)/'session', 'P01', 'session')
            try:
                trial = ex.begin(ROUTE, [10, 40], [10, 10], [640, 480])
                self.assertEqual(trial['palm_outbound'][0], [100., 130.])
                self.assertEqual(trial['reference'], ROUTE)
                self.assertIsNone(trial['start_t_s'])
            finally:
                ex.close()

    def test_top_reference_can_arrive_later_and_is_fixed_once(self):
        with tempfile.TemporaryDirectory() as folder:
            ex = Experiment(Path(folder)/'session', 'P01', 'session', clock=SessionClock(100, 1000000))
            try:
                trial = ex.begin(ROUTE, None, None, [640, 480], mono=100)
                trial['start_t_s'] = 1.
                for t, palm, ratio in [(102, [999., 999.], None), (103, [100., 130.], .7),
                                       (104, [100., 120.], .7)]:
                    obs = SimpleNamespace(palm=np.array(palm), pinch_xy=np.array([100., 100.]), pinch_ratio=ratio)
                    ex.sample(mono=t, top_mono=t, side_mono=t, observation=obs, side_valid=True,
                              tracked=None, ball=[20, 20], held=False, pinch_ratio=.7,
                              tracking_status='Detected', recordings={'top': True, 'side': True})
                self.assertEqual(trial['palm_offset_px'], [0., 30.])
                self.assertEqual(trial['palm_offset_source'], 'first_valid_top_observation')
                self.assertEqual(trial['palm_offset_t_s'], 3.)
                ex.finish('incomplete', mono=105)
            finally:
                ex.close()
            metrics = accuracy_metrics(trial, read_rows(ex.directory/'trajectory.csv'))
            self.assertEqual(metrics['valid_palm_samples'], 3)
            self.assertEqual(metrics['trajectory_rmse_px'], 0.)  # No retrospective error for the earlier unaligned palm.
            events = read_rows(ex.directory/'events.csv')
            self.assertEqual(sum(e['event'] == 'palm_reference_initialized' for e in events), 1)

    def test_smoothed_display_is_saved_separately_and_accuracy_uses_raw_palm(self):
        with tempfile.TemporaryDirectory() as folder:
            ex = Experiment(Path(folder)/'session', 'P01', 'session', clock=SessionClock(100,1000000))
            try:
                trial=ex.begin(ROUTE,[100,130],[100,100],[640,480],mono=100)
                trial['start_t_s']=0.
                for i in range(60):
                    palm=np.array([100.+5*(-1)**i,100.])
                    obs=SimpleNamespace(palm=palm,pinch_xy=palm-[0,30],pinch_ratio=.7)
                    ex.sample(mono=100+i/15,top_mono=100+i/15,side_mono=100+i/15,
                              observation=obs,side_valid=True,tracked=None,ball=[20,20],held=False,
                              pinch_ratio=.7,tracking_status='Detected',recordings={'top':True,'side':True})
                ex.finish('incomplete',mono=104)
            finally:ex.close()
            rows=read_rows(ex.directory/'trajectory.csv')
            raw=np.array([float(r['palm_x_px']) for r in rows])
            displayed=np.array([float(r['display_palm_x_px']) for r in rows])
            self.assertLess(displayed[20:].std(),raw[20:].std()*.4)
            self.assertEqual(accuracy_metrics(trial,rows)['trajectory_rmse_px'],5.)
            metadata=json.loads((ex.directory/'session.json').read_text())
            self.assertIn('Raw palm_*',metadata['display_trajectory'])

    def test_missing_top_reference_reports_unavailable_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            ex = Experiment(Path(folder)/'session', 'P01', 'session', clock=SessionClock(100, 1000000))
            try:
                trial = ex.begin(ROUTE, None, None, [640, 480], mono=100)
                trial['start_t_s'] = 1.
                ex.sample(mono=102, top_mono=102, side_mono=102, observation=None, side_valid=True,
                          tracked=None, ball=[20, 20], held=False, pinch_ratio=.7,
                          tracking_status='Missing', recordings={'top': True, 'side': True})
                ex.finish('incomplete', mono=103)
            finally:
                ex.close()
            reports = generate_report(ex.directory)
            self.assertEqual(read_rows(reports/'trial_summary.csv')[0]['trajectory_rmse_px'], '')
            for name in ('trajectory', 'timeline'):
                self.assertGreater((reports/f'trial_001_{name}.png').stat().st_size, 1000)

    def test_shared_motion_display_does_not_hide_raw_palm_error_in_report(self):
        with tempfile.TemporaryDirectory() as folder:
            ex = Experiment(Path(folder)/'session', 'P01', 'session', clock=SessionClock(100, 1000000))
            try:
                trial = ex.begin(ROUTE, [100, 130], [100, 100], [640, 480], mono=100)
                trial['start_t_s'] = 0.
                for i in range(40):
                    palm = np.array([100.+20*np.sin(i*.3), 100.])
                    obs = SimpleNamespace(palm=palm, pinch_xy=palm-[0,30], pinch_ratio=.7,
                                          palm_points=np.array([[100., 180.]]*5), scale=100.)
                    ex.sample(mono=100+i/15, top_mono=100+i/15, side_mono=100+i/15,
                              observation=obs, side_valid=True, tracked=None, ball=[20,20], held=False,
                              pinch_ratio=.7, tracking_status='Detected', recordings={'top':True,'side':True},
                              top_tracking=dict(point=(100.,100.), segment=1, source='Hand tracked', age_ms=0, valid=1))
                ex.finish('incomplete', mono=103)
            finally:
                ex.close()
            rows = read_rows(ex.directory/'trajectory.csv')
            self.assertTrue(all(float(r['wrist_x_px']) == 100. for r in rows))
            self.assertTrue(all(abs(float(r['display_palm_x_px'])-100.) < 1e-6 for r in rows))
            self.assertGreater(accuracy_metrics(trial, rows)['trajectory_rmse_px'], 10.)

    def test_command_settings_are_snapshots_and_roles_and_rejections_survive(self):
        c = ManualBleController(True, '', '')
        c.trial_id = '001'
        receipt = CommandReceipt(role='grab_prompt')
        self.assertFalse(c.submit(Cue.GRASP, Actuation(1500, .4, 90), receipt=receipt)[0])
        c.trial_id = '002'
        events = list(c.drain_events())
        self.assertEqual([e['event'] for e in events], ['command_requested', 'command_rejected'])
        self.assertTrue(all(e['trial_id'] == '001' and e['role'] == 'grab_prompt' for e in events))
        self.assertEqual(events[-1]['commanded_voltage_mv'], 1500)
        self.assertEqual(events[-1]['commanded_duration_s'], .4)
        normalized = [{**e, 't_s': e['mono'], 'session_id': 's'} for e in events]
        self.assertEqual(command_rows(normalized)[0]['status'], 'rejected')

    def test_video_frame_alignment_uses_capture_time_not_record_start(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = SessionClock.start()
            recorder = DualRecorder(folder, 30, clock, 'S01')
            frames = {name: np.zeros((64, 96, 3), np.uint8) for name in ('top', 'side')}
            now = time.monotonic()
            captures = {'top': now-.05, 'side': now-.02}
            epochs = {name: clock.unix_ms(t) for name, t in captures.items()}
            recorder.toggle('both', frames, now, clock.unix_ms(now), capture_times=captures,
                            timestamps=epochs, trial_id='007')
            time.sleep(.09)
            recorder.close()
            first = {}
            for name, stream in recorder.streams.items():
                row = read_rows(stream.path.with_suffix('.csv'))[0]
                self.assertEqual(row['session_id'], 'S01')
                self.assertEqual(row['trial_id'], '007')
                self.assertAlmostEqual(float(row['capture_t_s']), clock.seconds(captures[name]))
                first[name] = float(row['video_t_s'])
            self.assertEqual(first['top'], first['side'])

    def test_incomplete_wrong_target_and_late_timer_are_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            ex = Experiment(Path(folder)/'s', 'P01', 's', clock=SessionClock(100, 1000000))
            try:
                for mode, expected in [('wrong', 'off_target'), ('late', 'invalid_sequence'), ('no_cue', 'not_started')]:
                    ex.begin(ROUTE, [100,130], [100,100], [640,480], mono=100)
                    ex.active['start_t_s'] = None if mode == 'no_cue' else 5 if mode == 'late' else 1
                    ex.grasp(103)
                    result = ex.release([20,20] if mode == 'wrong' else [100,100],
                                        manual=False, position_valid=True, mono=108)
                    self.assertEqual(result['result'], expected)
                    self.assertIsNone(result['completion_time_s'])
                    self.assertFalse(result['success'])
            finally:
                ex.close()

    def test_trial_video_start_does_not_partially_record(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = DualRecorder(folder)
            frame = np.zeros((64,96,3),np.uint8)
            self.assertFalse(recorder.start_both({'top':frame,'side':None}, time.monotonic(),1000))
            self.assertFalse(any(s.active for s in recorder.streams.values()))
            with patch.object(recorder.streams['side'], 'start', return_value=False):
                self.assertFalse(recorder.start_both({'top':frame,'side':frame}, time.monotonic(),1000))
            self.assertFalse(any(s.active for s in recorder.streams.values()))


def run_synthetic_trial(directory, record_content='display', linked_mode=False, startup_case=None):
    """Safe test fixture: synthetic frames, no real windows, no BLE or cameras."""
    cursor = [-1]
    route = reference_route(640, 480, workspace_bounds(640, 480))
    start, turn, pickup = [np.array(route[k]) for k in ('start', 'turn', 'pickup')]
    outbound=np.array(reference_path(route))
    def along(points,fraction):
        lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
        cumulative=np.r_[0,np.cumsum(lengths)]
        distance=np.clip(fraction,0,1)*cumulative[-1]
        return np.array([np.interp(distance,cumulative,points[:,axis]) for axis in range(2)])
    detectors = []
    def location(i):
        if i < 3: return start
        if i <= 8: return along(outbound[:2],(i-3)/4)
        if i <= 16: return along(outbound[1:],(i-9)/6)
        if i <= 25: return pickup
        if i <= 32: return along(outbound[-2:][::-1],(i-26)/5)
        if i <= 39: return along(outbound[:-1][::-1],(i-33)/5)
        return start
    class Capture:
        def __init__(self, top): self.top = top
        def read(self):
            if self.top:
                cursor[0] += 1
                time.sleep(.035)
            frame = np.full((480, 640, 3), 235 if self.top else 215, np.uint8)
            cv2.putText(frame, 'SYNTHETIC TEST - NO HARDWARE', (80, 460), cv2.FONT_HERSHEY_SIMPLEX, .6, (40,40,40), 1)
            return True, frame
        def release(self): pass
    def hands(**kwargs):
        obj = SimpleNamespace(close=lambda: None)
        detectors.append(obj)
        return obj
    def detect(detector, frame, scale, **kwargs):
        i = cursor[0]
        if i < 6 and ((startup_case == 'top_missing' and detector is detectors[0]) or
                      (startup_case == 'side_missing' and detector is detectors[1])):
            return None, None
        anchor = location(i)
        palm = anchor + [0, 40]
        if linked_mode and detector is detectors[1]:
            if i == 28: return None,None
            lift = max(0,1-abs(i-31)/8)*140 if 23 <= i <= 39 else 0
            anchor=np.array([200+location(i)[0]*.35,300-lift])
            palm=anchor+[0,40]
        ratio = .7 if i < 22 else .2 if i < 41 else 1.1
        if startup_case == 'top_missing' and i < 6 and detector is detectors[1]:
            ratio = .2  # A detected closed hand can start a trial; open-to-pinch is a grasp rule.
        return HandObservation(palm, np.eye(2), 80, anchor, ratio, .05,
            palm + [[-25,30],[-40,-30],[0,-40],[25,-30],[40,0]]), None
    class Panel(OperatorPanel):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._key_down = None
            if linked_mode:
                self.side_place_request=(200+pickup[0]*.35,300.)
        def open(self): pass
        def show(self, *a, **kw): pass
    keys = {0:'n', 2:'w', 8:' ', 9:'a', 16:' ', 17:'g', 25:'d', 32:' ', 33:'s', 40:' ', 57:'q'}
    if startup_case == 'side_missing':
        keys = {0: 'n', 6: 'q'}
    cfg = AstraDemoConfig(session_log_dir=str(Path(directory).resolve()))
    with ExitStack() as stack:
        for name, value in {
            'AstraDemoConfig': lambda: cfg, 'enable_dpi_awareness': lambda: None,
            'open_uvc_camera': lambda camera, *a: Capture(camera == 1),
            'load_hands_api': lambda: (SimpleNamespace(Hands=hands), None), 'detect_hand': detect,
            'open_video_window': lambda *a: None, 'show_video_frame': lambda *a: None,
            'OperatorPanel': Panel,
        }.items(): stack.enter_context(patch.object(demo, name, value))
        arguments = ['demo','--no-ble','--subject','SYNTHETIC','--top-camera','1','--side-camera','3',
                     '--record-content', record_content]
        if linked_mode:
            arguments += ['--linked-views']
            stack.enter_context(patch.object(demo.cv2,'setMouseCallback'))
        stack.enter_context(patch.object(sys, 'argv', arguments))
        stack.enter_context(patch.object(demo.cv2, 'waitKeyEx', lambda _: ord(keys[cursor[0]]) if cursor[0] in keys else -1))
        stack.enter_context(patch.object(demo.cv2, 'getWindowProperty', return_value=1))
        stack.enter_context(patch.object(demo.cv2, 'destroyAllWindows'))
        demo.main()
    return sorted(Path(directory).glob('experiment_*'))[-1]


class IntegrationTests(unittest.TestCase):
    def test_n_starts_both_videos_with_only_side_hand_and_top_recovers(self):
        with tempfile.TemporaryDirectory() as folder:
            session = run_synthetic_trial(folder, linked_mode=True, startup_case='top_missing')
            trial = json.loads((session/'trials.json').read_text())[0]
            self.assertEqual(trial['result'], 'success')
            self.assertEqual(trial['palm_offset_source'], 'first_valid_top_observation')
            self.assertGreater(trial['palm_offset_t_s'], trial['start_t_s'])
            rows = read_rows(session/'trajectory.csv')
            self.assertTrue(any(r['trial_id'] == '001' and r['top_valid'] == '0' and
                                r['top_recording'] == r['side_recording'] == '1' for r in rows))
            events = read_rows(session/'events.csv')
            self.assertFalse(any(e['event'] in ('trial_start_rejected', 'command_sent') for e in events))
            commands = command_rows(events)
            self.assertEqual(commands[0]['cue'], 'Forward')  # N itself never emits a cue.
            self.assertTrue(any(c['cue'] == 'Left' for c in commands))  # A remains left guidance.
            for name in ('top', 'side'):
                video = cv2.VideoCapture(str(next((session/'videos').glob(name+'*.mp4'))))
                try:
                    self.assertGreater(video.get(cv2.CAP_PROP_FRAME_COUNT), 10)
                finally:
                    video.release()
            self.assertGreater((session/'reports/trial_001_timeline.png').stat().st_size, 1000)

    def test_n_requires_side_palm_and_fingers(self):
        with tempfile.TemporaryDirectory() as folder:
            session = run_synthetic_trial(folder, linked_mode=True, startup_case='side_missing')
            self.assertEqual(json.loads((session/'trials.json').read_text()), [])
            rejected = [e for e in read_rows(session/'events.csv') if e['event'] == 'trial_start_rejected']
            self.assertEqual(len(rejected), 1)
            self.assertIn('Side View', rejected[0]['reason'])
            self.assertEqual(list((session/'videos').glob('top_*.mp4')), [])
            self.assertEqual(list((session/'videos').glob('side_*.mp4')), [])
            self.assertEqual(len(list((session/'videos').glob('operator_*.mp4'))), 1)

    def test_linked_trial_records_both_balls_pixel_trajectory_and_no_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            session=run_synthetic_trial(folder,linked_mode=True)
            rows=read_rows(session/'trajectory.csv')
            self.assertEqual(rows[28]['held'],'1')
            self.assertEqual(rows[28]['side_position_valid'],'0')
            self.assertEqual(rows[28]['side_ball_y_px'],rows[27]['side_ball_y_px'])
            self.assertGreater(max(float(r['side_ball_rise_px']) for r in rows),90)
            self.assertTrue(all(r['ball_world_z_mm']=='' for r in rows))
            trial=json.loads((session/'trials.json').read_text())[0]
            self.assertEqual(trial['result'],'success')
            self.assertTrue((session/'reports/trial_001_side_trajectory.png').stat().st_size>1000)
            self.assertTrue((session/'reports/trial_001_timeline.svg').stat().st_size>1000)
            meta=json.loads((session/'session.json').read_text())
            self.assertFalse(meta['linked_views']['depth_enabled'])
            for name in ('top','side'):
                path=next((session/'videos').glob(name+'*.mp4'))
                video=cv2.VideoCapture(str(path))
                try:
                    self.assertGreater(video.get(cv2.CAP_PROP_FRAME_COUNT),10)
                    ok,frame=video.read();self.assertTrue(ok)
                    x,y=trial['reference']['pickup'] if name=='top' else trial['side_initial_ball_px']
                    color=frame[round(y),round(x)].astype(float)
                    self.assertGreater(color[0]-color[2],30)  # The blue sphere is recorded in BOTH views.
                finally: video.release()

    def test_linked_raw_recording_excludes_both_spheres(self):
        with tempfile.TemporaryDirectory() as folder:
            session=run_synthetic_trial(folder,record_content='raw',linked_mode=True)
            trial=json.loads((session/'trials.json').read_text())[0]
            for name in ('top','side'):
                video=cv2.VideoCapture(str(next((session/'videos').glob(name+'*.mp4'))))
                try:
                    ok,frame=video.read();self.assertTrue(ok)
                    x,y=trial['reference']['pickup'] if name=='top' else trial['side_initial_ball_px']
                    region=frame[round(y)-15:round(y)+15,round(x)-15:round(x)+15]
                    self.assertLess(np.std(region,axis=(0,1)).max(),2)
                    self.assertGreater(region.mean(),200)
                finally:video.release()


    def test_full_trial_videos_events_metrics_and_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshots = []
            grasp_labels = []
            draw = demo.draw_haptic_status
            def capture_overlay(frame, snapshot, clock):
                snapshots.append(snapshot)
                right = draw(frame, snapshot, clock)
                if snapshot.trigger and snapshot.trigger.label == 'GRASP CUE':
                    grasp_labels.append(frame[12:34, 20:165].copy())
                return right
            with patch.object(demo, 'draw_haptic_status', capture_overlay):
                session = run_synthetic_trial(folder)
            self.assertTrue(all(snapshots[i] is snapshots[i+1] for i in range(0, len(snapshots), 2)))
            self.assertTrue(any(s.trigger and s.trigger.label == 'GRASP CUE' for s in snapshots))
            self.assertTrue(all(s.simulated for s in snapshots))
            trials = json.loads((session/'trials.json').read_text())
            self.assertEqual(len(trials), 1)
            self.assertEqual(trials[0]['result'], 'success')
            self.assertTrue(trials[0]['simulated'])
            self.assertGreater(trials[0]['completion_time_s'], 1.)
            events = read_rows(session/'events.csv')
            self.assertEqual(sum(e['event'] == 'task_started' for e in events), 1)
            self.assertTrue(any(e['role'] == 'grab_prompt' for e in events))
            for name in ('top', 'side'):
                paths = list((session/'videos').glob(name+'*.mp4'))
                self.assertEqual(len(paths), 1)
                self.assertGreater(len(read_rows(paths[0].with_suffix('.csv'))), 10)
                # Match the rendered GRASP CUE lettering after MP4 encoding in BOTH
                # recordings. This checks the label itself, independent of its theme.
                video = cv2.VideoCapture(str(paths[0]))
                active_frames = 0
                try:
                    while True:
                        ok, frame = video.read()
                        if not ok:
                            break
                        error = np.abs(frame[12:34, 20:165].astype(float) - grasp_labels[0])
                        active_frames += bool(error.mean() < 6)
                finally:
                    video.release()
                self.assertGreater(active_frames, 0)
            ui_paths=list((session/'videos').glob('operator_*.mp4'))
            self.assertEqual(len(ui_paths),1)
            self.assertEqual(trials[0]['videos']['operator'],[str(ui_paths[0].relative_to(session))])
            ui_times=read_rows(ui_paths[0].with_suffix('.csv'))
            self.assertLess(float(ui_times[0]['video_t_s']),trials[0]['setup_t_s'])
            self.assertTrue(all(float(r['capture_t_s']) >= float(r['top_capture_t_s']) and
                                float(r['capture_t_s']) >= float(r['side_capture_t_s']) for r in ui_times))
            ui_capture=cv2.VideoCapture(str(ui_paths[0]))
            try:
                ok,ui_frame=ui_capture.read()
                self.assertTrue(ok)
                self.assertEqual(ui_frame.shape,(900,1180,3))
                expected=OperatorPanel().render()
                for rect in [(24,30,470,70),(836,337,1140,449)]:
                    x0,y0,x1,y1=rect
                    self.assertLess(np.abs(ui_frame[y0:y1,x0:x1].astype(float)-expected[y0:y1,x0:x1]).mean(),6.)
            finally:
                ui_capture.release()
            report = read_rows(session/'reports'/'trial_summary.csv')[0]
            self.assertGreater(float(report['tracking_valid_fraction']), .9)
            self.assertLess(float(report['trajectory_rmse_px']), 1e-5)
            self.assertTrue(all((session/'reports'/f'trial_001_{name}.{ext}').stat().st_size > 1000
                                for name in ('trajectory', 'timeline') for ext in ('png', 'svg')))

    def test_raw_videos_exclude_on_screen_trigger(self):
        with tempfile.TemporaryDirectory() as folder:
            session = run_synthetic_trial(folder, record_content='raw')
            for path in (session / 'videos').glob('*.mp4'):
                if path.name.startswith('operator_'):
                    continue  # UI always includes labels/controls, even when camera files are Raw.
                video = cv2.VideoCapture(str(path))
                frames = 0
                try:
                    while True:
                        ok, frame = video.read()
                        if not ok:
                            break
                        frames += 1
                        # Raw synthetic input has uniform gray here, including during cues.
                        band = frame[50:110, 8:620].astype(float)
                        self.assertLess(np.std(band, axis=(0, 1)).max(), 3)
                        self.assertGreater(band.mean(), 200)
                finally:
                    video.release()
                self.assertGreater(frames, 10)


if __name__ == '__main__':
    unittest.main()

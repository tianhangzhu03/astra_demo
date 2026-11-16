"""Independent AirPods-case pilot: manual BLE cues, video and manual annotations."""

import argparse
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import time

import cv2

from .cameras import CameraStream, SyntheticCamera
from .config import Config, PACKAGE_DIR
from .haptics import Actuation, CommandReceipt, Cue, ManualBleController
from .hand_trace import HandTraceService
from .panel import Panel
from .recording import RecordingGroup, Sample
from .session import PilotSession


CUES = dict(forward=Cue.FORWARD, back=Cue.BACK, left=Cue.LEFT, right=Cue.RIGHT,
            normal=Cue.NORMAL, grasp_cue=Cue.GRASP)


class Pilot:
    """Application policy; annotation actions have no path to BLE output."""
    def __init__(self, config, session, controller, recordings, *, route=True, traces=None):
        self.config, self.session = config, session
        self.controller, self.recordings = controller, recordings
        self.settings = Actuation()
        self.route, self.want_record = route, True
        self.message = 'Place the case at PICKUP. N: new trial. F/R: manual grasp/release.'
        self.quit_requested = self.start_pending = False
        self.frames = dict(top=None, side=None)
        self.evidence = {}
        self._camera_states = {}
        self._ui_sequence = 0
        self.traces = traces
        self.next_normal = 'lower'
        self._normal_receipt = None

    def drain_haptics(self):
        for event in self.controller.drain_events():
            self.session.ingest_haptic(event)
            receipt = self._normal_receipt
            if (receipt is not None and event.get('command_id') == receipt.log_data.get('command_id')
                    and event['event'] in ('command_sent', 'command_simulated')):
                previous = receipt.role.removeprefix('normal_')
                self.next_normal = 'lift' if previous == 'lower' else 'lower'
                self._normal_receipt = None
                self.session.event('normal_intent_advanced', mono=event['mono'],
                    source='simulation' if event['event'] == 'command_simulated' else 'ble',
                    from_command_id=event['command_id'], delivered_intent=previous, next_intent=self.next_normal)
            elif receipt is not None and event.get('command_id') == receipt.log_data.get('command_id') and event['event'] in (
                    'command_rejected', 'command_cancelled', 'command_expired', 'command_failed'):
                self._normal_receipt = None

    def _sync_trial(self):
        self.controller.trial_id = self.recordings.trial_id = self.session.trial_id

    def cameras_ready(self, now=None):
        now = time.monotonic() if now is None else now
        return all(frame is not None and 0 <= now-frame.mono <= self.config.stale_after_s
                   for frame in self.frames.values())

    def update(self, top, side, now=None):
        now = time.monotonic() if now is None else now
        self.frames = dict(top=top, side=side)
        self.drain_haptics()
        if self.traces:
            self.traces.submit(self.frames, self.session.trial_id)
            self.traces.consume()
        for name, frame in self.frames.items():
            live = frame is not None and now-frame.mono <= self.config.stale_after_s
            size = list(frame.image.shape[1::-1]) if frame is not None else None
            current = (live, tuple(size) if size else None)
            if self._camera_states.get(name) != current:
                self._camera_states[name] = current
                self.session.event('camera_live' if live else 'camera_stale', mono=now, source='camera',
                                   view=name, size=size, frame_id=frame.sequence if frame else None,
                                   capture_t_s=self.session.clock.seconds(frame.mono) if frame else None)
        if self.recordings.error:
            error = self.recordings.error
            self.controller.stop_output()
            self._normal_receipt = None
            self.session.finish_trial('recording_failed')
            self.recordings.stop('recording_failed')
            self.want_record = self.start_pending = False
            self._sync_trial()
            self.message = 'Recording stopped: ' + error

    def state(self):
        active = self.session.active
        detail = 'Place the case at PICKUP, then start a trial.'
        if active:
            start = active['first_direction_t_s']
            if start is None:
                detail = 'Ready - waiting for the first sent direction.'
            elif active['direction_to_release_s'] is not None:
                detail = f'First direction to manual release: {active["direction_to_release_s"]:.2f} s'
            else:
                detail = f'Elapsed from first direction: {self.session.clock.seconds()-start:.2f} s'
            if active['timing_source'] == 'simulated_direction':
                detail += ' (simulated cue)'
        return dict(settings=self.settings, recording=self.recordings.active, route=self.route,
                    mono=time.monotonic(), session_s=self.session.clock.seconds(),
                    feedback=self.controller.feedback_snapshot(), ble_status=self.controller.status_msg,
                    trial_id=self.session.trial_id, object_state=self.session.object_state,
                    trial_detail=detail, message=('Hand trace unavailable: '+self.traces.error if self.traces and self.traces.error else self.message),
                    traces=self.traces.snapshot() if self.traces else {}, next_normal=self.next_normal)

    def _start_trial(self):
        if not self.cameras_ready():
            self.message = 'Both cameras must have fresh frames before N.'
            return
        if self.controller.busy:
            self.message = 'Wait for the active pulse to finish, or stop [X], before N.'
            return
        if not self.recordings.active:
            self.want_record = self.start_pending = True
            self.message = 'Starting three-video recording before the new trial.'
            return
        started, self.message = self.session.start_trial(self.recordings.segment_id, evidence=self.evidence)
        if started:
            self.next_normal, self._normal_receipt = 'lower', None
            self.session.event('normal_intent_selected', intent='lower', reason='new_trial')
            if self.traces:
                self.traces.reset()
        self.start_pending = False
        self._sync_trial()

    def publish_canvas(self, canvas):
        mono = time.monotonic()
        self._ui_sequence += 1
        sources = {}
        for name, frame in self.frames.items():
            sources[f'{name}_frame_id'] = frame.sequence if frame else ''
            sources[f'{name}_capture_t_s'] = self.session.clock.seconds(frame.mono) if frame else ''
        self.evidence = dict(display_t_s=self.session.clock.seconds(mono), **sources)
        sample = Sample(canvas, mono, self._ui_sequence, self.session.trial_id, sources)
        if self.want_record and not self.recordings.streams and self.cameras_ready(mono):
            ok, self.message = self.recordings.start(self.frames['top'], self.frames['side'], sample)
            if not ok:
                self.want_record = self.start_pending = False
            elif self.start_pending:
                self._start_trial()
        self.recordings.operator_frame(sample)

    def action(self, action):
        self.drain_haptics()
        self.session.event('operator_action', action=action, evidence=self.evidence)
        if action in CUES:
            cue = CUES[action]
            role = ('normal_'+self.next_normal if cue == Cue.NORMAL else
                    'grab_prompt' if cue == Cue.GRASP else 'operator')
            receipt = CommandReceipt(role=role)
            send = self.controller.submit if self.config.ble_enabled else self.controller.simulate
            accepted, self.message = send(cue, self.settings, receipt=receipt)
            if accepted and cue == Cue.NORMAL:
                self._normal_receipt = receipt
                self.message = f'{self.next_normal.upper()} cue requested using Normal mode 0.'
            self.drain_haptics()
        elif action == 'normal_next':
            if self.controller.busy or self._normal_receipt is not None:
                self.message = 'Wait for the pending pulse or stop [X] before changing its meaning.'
            else:
                self.next_normal = 'lift' if self.next_normal == 'lower' else 'lower'
                self.session.event('normal_intent_selected', intent=self.next_normal, reason='operator_override')
                self.message = f'Next Normal means {self.next_normal.upper()}. No output sent.'
        elif action in ('mark_grasp', 'mark_release'):
            _, self.message = self.session.mark('grasped' if action == 'mark_grasp' else 'released', evidence=self.evidence)
        elif action == 'undo':
            _, self.message = self.session.undo()
        elif action == 'new_trial':
            self._start_trial()
        elif action in ('stop', 'end_trial', 'reset', 'quit'):
            self.message = self.controller.stop_output()
            if action != 'stop':
                self._normal_receipt = None
            self.start_pending = False
            if action == 'end_trial':
                self.session.finish_trial()
                self.message = 'Trial ended. Recording continues. Reposition the case, then N.'
            elif action == 'reset':
                self.session.finish_trial('reset')
                self.next_normal = 'lower'
                if self.traces:
                    self.traces.reset()
                self.message = 'Trial reset. Physically replace the case at PICKUP, then press N.'
            elif action == 'quit':
                self.quit_requested = True
            self._sync_trial()
            self.drain_haptics()
        elif action == 'record':
            if self.recordings.streams or self.want_record:
                self.controller.stop_output()
                self._normal_receipt = None
                self.session.finish_trial('recording_paused')
                self.recordings.stop('operator_pause')
                self.want_record = self.start_pending = False
                self._sync_trial()
                self.message = 'All recordings paused. B resumes; N resumes and starts a trial.'
            else:
                self.want_record = True
                self.message = 'Waiting for fresh camera frames to resume all recordings.'
        elif action == 'route':
            self.route = not self.route
            self.session.event('guide_changed', visible=self.route)
        elif action.rsplit('_', 1)[0] in ('voltage', 'duration', 'frequency'):
            name, direction = action.rsplit('_', 1)
            field, step, low, high = {'voltage': ('volts_mv', 25, 0, 5000),
                                      'duration': ('duration_s', .1, .1, 5),
                                      'frequency': ('frequency_hz', 10, 10, 200)}[name]
            value = min(high, max(low, round(getattr(self.settings, field) + (step if direction == 'up' else -step), 2)))
            self.settings = replace(self.settings, **{field: value})
            self.session.event('settings_changed', commanded_voltage_mv=self.settings.volts_mv,
                               commanded_duration_s=self.settings.duration_s, frequency_hz=self.settings.frequency_hz)
            self.message = 'Settings updated; no output until a cue is pressed.'


def validate_log_root(path):
    path = Path(path).expanduser().resolve()
    repo = PACKAGE_DIR.parent.resolve()
    protected = [repo / name for name in ('astra_demo', 'session_logs', 'output', 'calibration')]
    if path == repo or any(path == p or path.is_relative_to(p) for p in protected):
        raise ValueError('Physical pilot output must stay outside existing demo code, logs and output directories')
    return path


def run(config, *, synthetic=False, headless=False, seconds=None, actions=(), route=True, backend=None):
    if synthetic and config.ble_enabled:
        raise ValueError('Synthetic mode requires BLE disabled')
    if headless and (not synthetic or seconds is None or seconds <= 0):
        raise ValueError('Headless runs require bounded synthetic input')
    if not synthetic and config.top_camera == config.side_camera:
        raise ValueError('Top and Side must use distinct camera indices')
    config = replace(config, log_root=validate_log_root(config.log_root), hand_trace=config.hand_trace and not synthetic)
    outcome = 'normal'
    with ExitStack() as cleanup:
        session = PilotSession(config, synthetic=synthetic)
        cleanup.callback(lambda: session.close(outcome))
        print(f'[PHYSICAL] {session.directory}', flush=True)
        recordings = RecordingGroup(session, config.recording_fps)
        controller = ManualBleController(config.ble_enabled, config.ble_address, config.ble_uuid)
        pilot = Pilot(config, session, controller, recordings, route=route)
        panel = Panel(config)
        if not headless:
            cleanup.callback(cv2.destroyAllWindows)

        def close_camera(name, camera):
            closed = camera.close()
            session.event('camera_closed', source='camera', view=name, completed=closed, error=camera.error)
            if not closed:
                print(f'[PHYSICAL] Camera {name} is still closing; release is unconfirmed.', flush=True)

        factory = SyntheticCamera if synthetic else CameraStream
        cameras = {}
        for name, index in (('top', 0 if synthetic else config.top_camera), ('side', 1 if synthetic else config.side_camera)):
            camera = factory(index, fps=config.camera_fps, backend=backend,
                             on_frame=lambda frame, view=name: recordings.camera_frame(view, frame))
            cameras[name] = camera
            cleanup.callback(close_camera, name, camera)
        if config.hand_trace:
            pilot.traces = HandTraceService(session, config)
            cleanup.callback(pilot.traces.close)
        cleanup.callback(recordings.stop, 'shutdown')
        cleanup.callback(pilot.drain_haptics)
        # Hardware stop runs before potentially slow encoder/camera cleanup.
        cleanup.callback(controller.stop)
        try:
            for camera in cameras.values():
                camera.start()
            if not headless:
                panel.open()
            controller.start()
            session.event('guide_changed', source='system', visible=route)
            print('[PHYSICAL] N trial; WASD direction; Space lower/lift (same Normal 0); Tab changes next meaning; G cue; F/R marks; X stop; Q exit.', flush=True)
            print('[PHYSICAL] ' + ('BLE enabled; initial voltage 0 mV.' if config.ble_enabled else 'BLE disabled; cues only simulated.'), flush=True)
            origin, pending, camera_errors = time.monotonic(), list(actions), {}
            while not pilot.quit_requested:
                loop = time.monotonic()
                for camera in cameras.values():
                    if synthetic:
                        camera.poll()
                pilot.update(cameras['top'].latest, cameras['side'].latest, loop)
                for name, camera in cameras.items():
                    if camera.error != camera_errors.get(name, ''):
                        camera_errors[name] = camera.error
                        session.event('camera_error' if camera.error else 'camera_recovered', source='camera',
                                      view=name, reason=camera.error)
                        if camera.error:
                            pilot.message = camera.error
                frame = panel.render(pilot.frames['top'], pilot.frames['side'], pilot.state())
                pilot.publish_canvas(frame)
                if not headless:
                    panel.show(frame)
                    panel.handle_key(cv2.waitKeyEx(1))
                    if cv2.getWindowProperty(panel.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        panel.actions.append('quit')
                while pending and loop-origin >= pending[0][0]:
                    panel.actions.append(pending.pop(0)[1])
                while panel.actions:
                    pilot.action(panel.actions.popleft())
                    if pilot.quit_requested:
                        break
                if seconds is not None and loop-origin >= seconds:
                    pilot.action('quit')
                delay = 1/config.ui_fps - (time.monotonic()-loop)
                if delay > 0 and not pilot.quit_requested:
                    time.sleep(delay)
        except KeyboardInterrupt:
            outcome = 'keyboard_interrupt'
        except Exception as exc:
            outcome = 'error'
            session.event('application_error', source='system', reason=str(exc))
            raise
    print(f'[PHYSICAL] Saved: {session.directory}', flush=True)
    return session.directory


def preview(config, output):
    config = replace(config, ble_enabled=False)
    cameras = [SyntheticCamera(i) for i in (0, 1)]
    for camera in cameras:
        camera.poll()
    controller = ManualBleController(False, '', '')
    state = dict(settings=Actuation(), recording=False, route=True, mono=time.monotonic(),
                 session_s=0, feedback=controller.feedback_snapshot(), message='Synthetic preview. No cameras or BLE opened.')
    frame = Panel(config).render(cameras[0].latest, cameras[1].latest, state)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f'Could not write {output}')
    print(output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--top-camera', type=int, default=2)
    parser.add_argument('--side-camera', type=int, default=3)
    parser.add_argument('--no-ble', action='store_true')
    parser.add_argument('--ble-address', default=Config.ble_address)
    parser.add_argument('--ble-uuid', default=Config.ble_uuid)
    parser.add_argument('--subject', default='physical_pilot')
    parser.add_argument('--object-name', default='AirPods case')
    parser.add_argument('--object-notes', default='')
    parser.add_argument('--output', type=Path, default=PACKAGE_DIR/'runs')
    parser.add_argument('--side-no-mirror', action='store_true')
    parser.add_argument('--hide-guide', action='store_true')
    parser.add_argument('--no-hand-trace', action='store_true', help='Disable the separate 2D palm trails')
    parser.add_argument('--hand-model', type=Path, default=Config.hand_model)
    parser.add_argument('--backend', choices=('msmf', 'dshow', 'any'))
    parser.add_argument('--preview', action='store_true', help='Save a synthetic UI preview; no camera / BLE access')
    parser.add_argument('--smoke', action='store_true', help='Run a 3-second synthetic recording check; no camera / BLE access')
    args = parser.parse_args()
    config = Config(top_camera=args.top_camera, side_camera=args.side_camera,
                    ble_enabled=not (args.no_ble or args.preview or args.smoke),
                    ble_address=args.ble_address, ble_uuid=args.ble_uuid, subject=args.subject,
                    object_name=args.object_name, object_notes=args.object_notes,
                    log_root=args.output, side_mirror=not args.side_no_mirror,
                    hand_trace=not args.no_hand_trace, hand_model=args.hand_model)
    if args.preview:
        preview(config, PACKAGE_DIR/'output'/'operator_preview.png')
        return
    actions = ((.2, 'new_trial'), (.4, 'forward'), (.7, 'grasp_cue'), (1.1, 'mark_grasp'),
               (1.6, 'mark_release'), (2.0, 'end_trial')) if args.smoke else ()
    backend = {'msmf': cv2.CAP_MSMF, 'dshow': cv2.CAP_DSHOW, 'any': cv2.CAP_ANY}.get(args.backend)
    run(config, synthetic=args.smoke, headless=args.smoke, seconds=3 if args.smoke else None,
        actions=actions, route=not args.hide_guide, backend=backend)


if __name__ == '__main__':
    main()

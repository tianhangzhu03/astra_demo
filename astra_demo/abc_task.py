"""Observation-driven ABC trial. Timers debounce observations; they never animate the hand."""

import time

from .manual_control import Cue, CommandReceipt, TORSION_CHANNELS, TORSION_DIRECTION, direction_code


FORCE_ACTIONS = {'normal_up': (Cue.NORMAL, 'normal_up'), 'normal_down': (Cue.NORMAL, 'normal_down'),
                'torsion_cw': (Cue.TORSION, 'torsion_cw'), 'torsion_ccw': (Cue.TORSION, 'torsion_ccw')}


class AbcTask:
    hints = dict(idle='Place hand at A, then N', reach='W / D: guide to B',
        pick='G: grab the ball', carry='S / D / W / A: follow the outer corridor',
        flip_cw='J: CW cue | flip downward, 90 deg',
        lift='Tap I / Space: Normal Up | lift into the downward opening',
        release='G: drop the ball', lower='Tap K / Space: Normal Down | lower the empty hand',
        flip_ccw='L: CCW cue | flip back, 90 deg',
        complete='Complete | N: next trial', failed='Trial failed | 0: restart')

    def __init__(self, scene, experiment, controller, *, manual_grasp=False):
        self.scene, self.experiment, self.controller = scene, experiment, controller
        self.manual_grasp = manual_grasp
        self.reset()

    def reset(self):
        self.cancel_output()
        self.phase, self.waypoint = 'idle', 1
        self.was_held = self.captured = self.collision = self.collision_seen = False
        self.baseline = self.angle = None
        self.flip_ready = self.entered_from_below = False
        self.last_ball = None
        self.stable_since = None
        self.drop_cued = False
        self.result = None
        self.flip_confirmations = []
        self.observed_at = -1e9
        self.lower_z = None

    def cancel_output(self):
        receipt = getattr(self, 'drop_receipt', None)
        if receipt is not None:
            self.controller.cancel(receipt)
        self.drop_receipt = None
        self.drop_cued = False


    def start(self, spatial, now):
        self.reset()
        self._change('reach', now)
        trial = self.experiment.active
        trial.update(task='abc', scene=self.scene.snapshot(),
                     torsion_available=self.controller.enabled and self.controller.torsion_firmware_ready,
                     torsion_channels=list(TORSION_CHANNELS), torsion_packet_status='defined',
                     torsion_direction_code=TORSION_DIRECTION,
                     torsion_firmware_ready=self.controller.torsion_firmware_ready,
                     torsion_output_direction='CW=5, CCW=6 on CEL v0.2.1; physical rotation is not measured',
                     flip_direction_verified=False, collision_seen=False,
                     success_scope='Linked RGB task completion; not verification of haptic output or CW/CCW rotation direction',
                     completion_rule='Capture in C, lower empty hand, then restore initial palm orientation')
        self.experiment._save_trials()
        if self.manual_grasp:
            trial.update(flip_confirmation='none; J/L record intent', flip_confirmations=[],
                         task_tracking='linked_2d',
                         grasp_control='operator G toggle; no gesture or proximity gate',
                         completion_rule='Manual grab/drop, box entry, lower empty hand and operator-requested flips')
            self.experiment._save_trials()

    def _change(self, phase, now):
        old, self.phase = self.phase, phase
        self.stable_since = None
        self.experiment.event('abc_phase', mono=now, previous=old, phase=phase,
                              intended_torsion={'flip_cw': 'CW', 'flip_ccw': 'CCW'}.get(phase))

    def _stable(self, condition, now):
        if not condition:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = now
        return now-self.stable_since >= .25

    @property
    def grasp_enabled(self):
        return self.experiment.active is not None and self.phase == 'pick'

    def force_command(self, action):
        """Resolve operator intent without sending; idle controls permit a BLE bench check."""
        if action == Cue.NORMAL:
            action = 'normal_down' if self.phase == 'lower' else 'normal_up'
        if action == Cue.TORSION:
            action = 'torsion_ccw' if self.phase == 'flip_ccw' else 'torsion_cw'
        cue, role = FORCE_ACTIONS.get(action, (action, 'operator'))
        if cue == Cue.GRASP:
            raise ValueError('G is a single pickup/drop cue')
        if self.experiment.active and not self.manual_grasp:
            if self.phase in ('failed', 'complete'):
                raise ValueError('End or reset the completed trial first')
            required = {'normal_up':'lift', 'normal_down':'lower', 'torsion_cw':'flip_cw', 'torsion_ccw':'flip_ccw'}
            if role in required and self.phase != required[role]:
                raise ValueError(f'{role.replace("_", " ")} applies at {required[role]}; current stage is {self.phase}')
        return cue, role

    def select(self, cue, panel):
        """None delegates to the existing G/WASD handler; other results are handled here."""
        if cue != Cue.GRASP:
            try:
                command, role = self.force_command(cue)
            except ValueError as exc:
                return False, str(exc)
            if command == Cue.TORSION:
                return self.request_torsion(panel.settings, role=role)
            receipt = CommandReceipt(role=role)
            send = self.controller.submit if self.controller.enabled else self.controller.simulate
            return send(command, panel.settings, receipt=receipt)
        if not self.experiment.active or self.phase in ('failed', 'complete'):
            return False, 'Start an ABC trial at A with N first'
        if self.manual_grasp:
            if not panel.held and not getattr(panel, 'grab_available', False):
                return False, 'No valid hand position; show marker/hand and press G again'
            panel.manual_request = 'release' if panel.held else 'grab'
            role = 'release_prompt' if panel.held else 'grab_prompt'
            receipt = CommandReceipt(role=role)
            if self.controller.enabled:
                accepted, message = self.controller.submit(cue, panel.settings, receipt=receipt, replace_output=True)
            else:
                accepted, message = self.controller.simulate(cue, panel.settings, receipt=receipt)
            if panel.held:
                self.drop_receipt = receipt if accepted else None
            self.experiment.event('manual_grasp_requested', action=panel.manual_request,
                                  feedback_accepted=accepted, feedback_status=message)
            return True, f'G: {panel.manual_request} / {message}'
        if cue == Cue.GRASP and self.phase == 'pick':
            return None
        if self.phase != 'release':
            return False, 'G is available at B for pickup or inside C for release'
        receipt = CommandReceipt(role='release_prompt')
        send = self.controller.submit if self.controller.enabled else self.controller.simulate
        accepted, message = send(cue, panel.settings, receipt=receipt)
        if accepted and cue == Cue.GRASP:
            self.drop_receipt, self.drop_cued = receipt, False
        return accepted, message

    def request_torsion(self, settings, *, role=None):
        try:
            _, role = self.force_command(role or Cue.TORSION)
        except ValueError as exc:
            return False, str(exc)
        receipt = CommandReceipt(role=role)
        if self.controller.enabled:
            accepted, reason = self.controller.submit(Cue.TORSION, settings, receipt=receipt, replace_output=self.manual_grasp)
        else:
            accepted, reason = self.controller.simulate(Cue.TORSION, settings, receipt=receipt)
        if not accepted:
            self.experiment.event('cue_unavailable', cue=Cue.TORSION.value, role=role,
                direction_code=direction_code(Cue.TORSION, role), reason=reason)
        return accepted, reason

    def poll_output(self, observed_at):
        receipt = self.drop_receipt
        if receipt is None:
            return
        invalid = (receipt.cancelled or receipt.status in ('failed', 'expired', 'cancelled')
                   or (self.controller.enabled and (not self.controller.is_connected
                       or receipt.connection_id != self.controller.connection_id)))
        if invalid:
            self.drop_cued = False
            self.drop_receipt = None
        elif receipt.sent_at is not None and receipt.status in ('sent', 'completed'):
            self.drop_cued = observed_at >= receipt.sent_at



    def record(self):
        return dict(task_phase=self.phase, task_waypoint=self.waypoint,
                    hand_flip_angle_deg=self.angle, flip_direction_verified=0,
                    obstacle_contact=int(self.collision), ball_captured=int(self.captured),
                    drop_prompt_valid=int(self.drop_cued), box_entry_from_below=int(self.entered_from_below))

    @property
    def status(self):
        if self.phase == 'flip_cw' and not self.flip_ready and not self.manual_grasp:
            return 'AT C: restore the pickup palm orientation below the box before flipping'
        if self.phase in ('flip_cw', 'flip_ccw') and self.controller.enabled and not self.controller.torsion_firmware_ready:
            return f'{self.phase.upper()}: waiting for CEL v0.2.1 verification'
        return f'{self.phase.upper()}: {self.hints[self.phase]}'

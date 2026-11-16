"""Trial-first English dashboard. Display only; it does not control recording or BLE."""
from collections import deque
import time

import cv2
import numpy as np

from .manual_control import Cue
from .operator_panel import text, fit_camera, PAPER, SURFACE, LINE, INK, MUTED, ACCENT, STOP_COLOR


class OperatorDashboard:
    def __init__(self, panel):
        self.panel = panel
        self.background = np.full((900, 1180, 3), PAPER, dtype=np.uint8)
        self.history = deque(maxlen=1800)
        self.trial_id = None
        self.active = False
        self.elapsed = self.task_elapsed = None
        self.visible = (False, False)
        self.values = [None]*3
        self.phase = 'idle'
        self.advanced_rect = (500, 91, 218, 40)
        panel.trial_status = 'Place ball + box, show the yellow tape, then Start trial & record [N].'
        panel.trial_rect = (24, 91, 235, 40)
        panel.end_trial_rect = (271, 91, 217, 40)
        panel.restart_rect = (866, 96, 150, 32)
        panel.exit_rect = (1028, 96, 128, 32)
        panel.side_ball_rect, panel.side_box_rect = (938, 163, 102, 28), (1050, 163, 106, 28)
        panel.record_buttons = {'top': (40, 635, 198, 40), 'side': (250, 635, 198, 40),
                                'both': (460, 635, 198, 40)}
        panel.ui_record_rect = (40, 690, 198, 40)
        panel.record_mode_rect = (250, 690, 198, 40)
        panel.buttons = {Cue.FORWARD: (704, 586, 106, 40), Cue.LEFT: (818, 586, 106, 40),
            Cue.BACK: (932, 586, 106, 40), Cue.RIGHT: (1046, 586, 110, 40),
            'normal_up': (704, 632, 222, 40), 'normal_down': (934, 632, 222, 40),
            'torsion_cw': (704, 678, 222, 40), 'torsion_ccw': (934, 678, 222, 40),
            Cue.GRASP: (704, 724, 222, 42)}
        panel.stop_rect = (934, 724, 222, 42)
        self.settings = [('volts_mv', 'Voltage', 'mV', 704),
                         ('duration_s', 'Duration', 's', 858),
                         ('frequency_hz', 'Vibration', 'Hz', 1012)]
        panel.adjust_buttons = [(field, direction, (x+offset, 798, 30, 30))
            for field, _, _, x in self.settings for direction, offset in [(-1, 0), (1, 114)]]

    def update(self, trial, marker, now):
        self.active = trial is not None
        marker = marker or {}
        self.visible = tuple(bool(marker.get(f'rgb_marker_{v}_valid')) for v in ('top', 'side'))
        self.values = [marker.get(f'rgb_marker_delta_{a}_au') for a in 'xyz']
        self.phase = marker.get('rgb_marker_phase', 'idle')
        if not trial:
            return  # Retain the last trial plot after E, instead of replacing it with idle data.
        if self.trial_id != trial['trial_id']:
            self.trial_id = trial['trial_id']
            self.history.clear()
        self.elapsed = max(0., now-trial['setup_t_s'])
        self.task_elapsed = None if trial['start_t_s'] is None else max(0., now-trial['start_t_s'])
        self.history.append((self.elapsed, *[np.nan if v is None else v for v in self.values], self.phase == 'z'))
        while self.history and self.history[0][0] < self.elapsed-30:
            self.history.popleft()

    def _plot(self, canvas):
        text(canvas, 'LIVE TAPE MOTION', (24, 590), .40, bold=True)
        colors = [(85, 85, 85), (51, 74, 226), (189, 138, 52)]
        for label, color, x in zip(('X / Top', 'Y / Top', 'Z proxy / Side'), colors, (288, 400, 506)):
            text(canvas, label, (x, 590), .34, color, bold=True)
        x, y, w, h = 68, 616, 600, 160
        cv2.rectangle(canvas, (x, y), (x+w, y+h), SURFACE, -1)
        if not self.history:
            text(canvas, 'Press Start trial & record [N] to begin.', (128, 695), .43, MUTED)
        else:
            data = np.asarray(self.history, dtype=float)
            end = max(30., data[-1, 0])
            start = end-30
            finite = data[:, 1:4][np.isfinite(data[:, 1:4])]
            span = max(.1, float(np.max(np.abs(finite)))*1.1) if len(finite) else .1
            px = x+(data[:, 0]-start)/30*w
            py = y+h/2-data[:, 1:4]/span*h/2
            normal = data[:, 4].astype(bool)
            edges = np.flatnonzero(np.diff(np.r_[False, normal, False])).reshape(-1, 2)
            for a, b in edges:
                cv2.rectangle(canvas, (round(px[a]), y),
                    (round(px[b] if b < len(px) else px[-1]), y+h), (237, 246, 255), -1)
            cv2.line(canvas, (x, y+h//2), (x+w, y+h//2), LINE, 1)
            for axis, color in enumerate(colors):
                valid = np.isfinite(py[:, axis])
                breaks = np.r_[False, np.diff(data[:, 0]) > 1.5]
                starts = np.flatnonzero(valid & (~np.r_[False, valid[:-1]] | breaks))
                stops = np.flatnonzero(valid & (~np.r_[valid[1:], False] | np.r_[breaks[1:], False]))+1
                segments = [np.rint(np.c_[px[a:b], py[a:b, axis]]).astype(np.int32) for a, b in zip(starts, stops)]
                if segments:
                    cv2.polylines(canvas, segments, False, color, 2, cv2.LINE_AA)
            text(canvas, f'+{span:.2f}', (24, y+9), .27, MUTED)
            text(canvas, '0', (48, y+h//2+5), .27, MUTED)
            text(canvas, f'-{span:.2f}', (24, y+h), .27, MUTED)
            text(canvas, f'{start:.0f}s', (x, 796), .28, MUTED)
            text(canvas, f'{end:.0f}s from trial start', (x+w-153, 796), .28, MUTED)
        text(canvas, 'Relative image motion (a.u.) | fixed trial zero | gaps = missing tape', (24, 818), .32, MUTED)
        text(canvas, 'Last 30 s | orange = Normal | different view scales; not metric XYZ', (24, 837), .29, MUTED)

    def _advanced(self, canvas):
        p = self.panel
        cv2.rectangle(canvas, (24, 574), (684, 839), SURFACE, -1)
        text(canvas, 'ADVANCED RECORDING', (40, 607), .43, bold=True)
        for view, rect in p.record_buttons.items():
            active = all(p.record_active.values()) if view == 'both' else p.record_active[view]
            suffix = {'top': 'Top [T]', 'side': 'Side [Y]', 'both': 'Both [B]'}[view]
            p._plain_button(canvas, rect, ('Stop ' if active else 'Record ') + suffix, active)
        p._plain_button(canvas, p.ui_record_rect, 'Stop UI [U]' if p.ui_record_active else 'Record UI [U]', p.ui_record_active)
        p._plain_button(canvas, p.record_mode_rect, f'Video: {p.record_mode} [V]')
        text(canvas, 'These controls change videos only. XYZ recording follows the trial.', (40, 762), .34, MUTED)
        text(canvas, 'Trial only: N starts recording; End trial stops all videos.', (40, 787), .34, MUTED)
        text(canvas, 'Normal workflow: Start [N] -> End [E] -> next Start [N].', (40, 817), .34, MUTED)

    def render(self, top, side, *, fps=0.):
        p = self.panel
        canvas = self.background.copy()
        text(canvas, 'Haptic guidance', (24, 44), .82, bold=True)
        subtitle = (p.trial_status if not self.active or p.restart_pending else
                    'Place ball + box  /  Start trial  /  Perform task  /  End & save')
        text(canvas, subtitle[:135], (24, 72), .36, MUTED)
        mode = 'VISION ONLY' if p.controller and not p.controller.enabled else 'MANUAL BLE'
        text(canvas, mode, (1024, 44), .36, MUTED, bold=True)
        p._plain_button(canvas, p.trial_rect, 'Start trial & record [N]', not self.active)
        p._plain_button(canvas, p.end_trial_rect, 'End trial & save [E]', self.active)
        p._plain_button(canvas, self.advanced_rect, 'Hide recording options' if p.advanced_recording else 'Advanced recording', p.advanced_recording)
        p._plain_button(canvas, p.restart_rect, 'Restart trial [0]', p.restart_pending)
        p._plain_button(canvas, p.exit_rect, 'Exit app [Q]')
        if self.active:
            timer = 'waiting for first WASD' if self.task_elapsed is None else f'{self.task_elapsed:.1f}s'
            status = f'Trial {self.trial_id} | XYZ recording: {self.elapsed:.1f}s | Task timer: {timer}'
        else:
            status = f'Trial {self.trial_id} ended | last plot retained | N starts a new trial' if self.trial_id else 'Preview only | N starts trial videos and XYZ recording'
        text(canvas, status, (24, 154), .42, ACCENT, bold=True)
        text(canvas, 'TOP VIEW', (24, 183), .40, bold=True)
        text(canvas, 'SIDE VIEW', (704, 183), .40, bold=True)
        text(canvas, 'Ball held' if p.held else 'Ball released', (550, 183), .33, MUTED)
        p._plain_button(canvas, p.side_ball_rect, 'Ball [P]', p.side_placement_mode == 'ball')
        p._plain_button(canvas, p.side_box_rect, 'Box [O]', p.side_placement_mode == 'box')
        fit_camera(canvas, top, (24, 196, 660, 340), 'Waiting for Top camera')
        p._side_camera_transform = fit_camera(canvas, side, (704, 196, 452, 339), 'Waiting for Side camera')
        p._side_camera_size = side.shape[1::-1] if side is not None else None
        for name, visible, x in zip(('top', 'side'), self.visible, (24, 704)):
            state = 'RECORDING' if p.record_active[name] else 'OFF'
            text(canvas, f'{name.title()} tape: {"visible" if visible else "missing"}  |  Video: {state}', (x, 558), .37, ACCENT if visible else MUTED)
        if p.advanced_recording:
            self._advanced(canvas)
        else:
            self._plot(canvas)
        age = time.perf_counter()-p.cue_time
        repeating = p.controller and p.controller.repeat_action is not None
        for cue, rect in p.buttons.items():
            p._button(canvas, cue, rect, cue == p.last_cue and (age < .8 or repeating))
        x, y, w, h = p.stop_rect
        cv2.rectangle(canvas, (x, y), (x+w, y+h), STOP_COLOR, 1)
        text(canvas, 'Stop output [X / Esc]', (x+12, y+26), .39, STOP_COLOR, bold=True)
        for field, label, unit, x in self.settings:
            value = getattr(p.settings, field)
            text(canvas, label, (x, 787), .32, MUTED)
            text(canvas, f'{value:g} {unit}', (x+36, 818), .32, bold=True)
        for _, direction, (x, y, w, h) in p.adjust_buttons:
            cv2.rectangle(canvas, (x, y), (x+w, y+h), LINE, 1)
            text(canvas, '+' if direction > 0 else '-', (x+10, y+22), .42)
        cv2.line(canvas, (24, 847), (1156, 847), LINE, 1)
        text(canvas, p.task_controls.status[:132], (24, 867), .34)
        text(canvas, p.message[:124], (24, 890), .30, MUTED)
        text(canvas, f'UI video: {"REC" if p.ui_record_active else "OFF"} | {fps:.1f} fps', (924, 890), .30, MUTED)
        return canvas

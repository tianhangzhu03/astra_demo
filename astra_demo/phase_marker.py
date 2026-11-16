"""Continuous independent image motion with trial-fixed zeros and cue annotations."""

from collections import deque
from .yellow_marker import detect_marker


SOURCE = 'yellow_tape_continuous_rgb'
PHASE_FIELDS = ['rgb_marker_phase', 'rgb_marker_z_segment', 'rgb_marker_overlap',
          'rgb_marker_top_valid', 'rgb_marker_side_valid', 'rgb_marker_z_reference_t_s']
FIELDS = ['rgb_marker_source', 'rgb_marker_units', 'rgb_marker_valid', 'rgb_marker_status',
          'rgb_marker_t_s', 'rgb_marker_skew_ms', 'rgb_marker_residual', 'rgb_marker_extrapolated',
          'rgb_marker_reference_t_s', 'rgb_marker_model_warning'] + [f'rgb_marker_{view}_{a}_px' for view in ('top', 'side') for a in 'xy'] + [
          f'rgb_marker_{prefix}{a}_au' for prefix in ('', 'delta_', 'reference_') for a in 'xyz'] + [
          f'rgb_marker_{prefix}{a}_mm' for prefix in ('', 'delta_', 'reference_') for a in 'xyz'] + [
          'rgb_marker_reprojection_px', 'rgb_marker_ray_angle_deg']
FIELDS += PHASE_FIELDS


class PhaseMarkerRecorder:
    model = None
    units = 'a.u.'
    source = SOURCE
    setup_status = 'Image trends ready; no board calibration'

    def __init__(self):
        self.events = deque()
        self.markers = [None, None]
        self.sizes = [None, None]
        self.times = [0., 0.]
        self.reference_id = None
        self._reset()

    def _reset(self):
        self.phase, self.previous = 'idle', 'idle'
        self.phase_start = float('-inf')
        self.xy_zero = self.z_zero = None
        self.xy_time = self.z_time = None
        self.segment = 0
        self.delta = None

    def command(self, event, mono, item):
        # Sent is a BLE write, not measured force. Simulation is labelled in events.
        if event not in ('command_sent', 'command_simulated'):
            return
        cue = item.get('cue')
        phase = 'xy' if cue in ('Forward', 'Back', 'Left', 'Right') else 'z' if cue == 'Normal force' else None
        if phase:
            self.events.append((mono, item.get('trial_id', ''), phase))

    def observe(self, top, side, top_mono, side_mono, *, top_hand=None, side_hand=None):
        self.markers = [detect_marker(top, top_hand),
                        detect_marker(side, side_hand)]
        self.sizes = [None if f is None else f.shape[1::-1] for f in (top, side)]
        self.times = [top_mono, side_mono]

    def record(self, clock, trial_id='', setup_t_s=None):
        if trial_id != self.reference_id:
            self.reference_id = trial_id
            self._reset()
        now = max(self.times)
        while self.events and self.events[0][0] <= now:
            when, identifier, phase = self.events.popleft()
            if identifier != trial_id or phase == self.phase:
                continue
            self.previous, self.phase, self.phase_start = self.phase, phase, when
            if phase == 'z':
                self.segment += 1
        points = []
        for marker, size, when in zip(self.markers, self.sizes, self.times):
            good = (marker is not None and size is not None and now-when <= .25
                    and (setup_t_s is None or clock.seconds(when) >= setup_t_s))
            # Both Top axes use the same image-height scale. Side has its own scale.
            points.append(marker.pixel / size[1] if good else None)
        top, side = points
        if top is not None and self.xy_zero is None:
            self.xy_zero, self.xy_time = top.copy(), clock.seconds(self.times[0])
        if side is not None and self.z_zero is None:
            self.z_zero, self.z_time = float(side[1]), clock.seconds(self.times[1])
        xy = ((top-self.xy_zero)*[1., -1.] if top is not None and self.xy_zero is not None
              else None)
        z = self.z_zero-float(side[1]) if side is not None and self.z_zero is not None else None
        self.delta = [None if xy is None else float(xy[0]), None if xy is None else float(xy[1]), z]
        row = dict(rgb_marker_source=self.source, rgb_marker_units=self.units,
                   rgb_marker_valid=int(xy is not None or z is not None),
                   rgb_marker_status=f'Continuous image trends / cue {self.phase}; not world XYZ',
                   rgb_marker_t_s=clock.seconds(now), rgb_marker_skew_ms=(self.times[1]-self.times[0])*1000,
                   rgb_marker_reference_t_s=self.xy_time, rgb_marker_z_reference_t_s=self.z_time,
                   rgb_marker_phase=self.phase, rgb_marker_z_segment=self.segment,
                   rgb_marker_overlap=0, rgb_marker_top_valid=int(top is not None),
                   rgb_marker_side_valid=int(side is not None), rgb_marker_model_warning=0,
                   rgb_marker_extrapolated=0)
        for view, marker in zip(('top', 'side'), self.markers):
            for i, axis in enumerate('xy'):
                row[f'rgb_marker_{view}_{axis}_px'] = float(marker.pixel[i]) if marker else None
        row.update({f'rgb_marker_delta_{a}_au': v for a, v in zip('xyz', self.delta)})
        row['rgb_marker_reference_x_au'] = float(self.xy_zero[0]) if self.xy_zero is not None else None
        row['rgb_marker_reference_y_au'] = -float(self.xy_zero[1]) if self.xy_zero is not None else None
        row['rgb_marker_reference_z_au'] = -self.z_zero if self.z_zero is not None else None
        return row

    def display_status(self):
        visible = ' / '.join(f'{v} {"OK" if m else "missing"}' for v, m in zip(('Top', 'Side'), self.markers))
        if self.delta is None or all(v is None for v in self.delta):
            return f'Tape: {visible} | Waiting for valid tape samples'
        values = '  '.join(f'{a} {v:+.3f}' if v is not None else f'{a} --' for a, v in zip('XYZ', self.delta))
        return f'Image trends [a.u.] {values} | {visible}'

"""Trial records on a shared host monotonic clock; no hardware calls or plotting here."""

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from .tracking import DisplayPalmTrace


@dataclass
class SessionClock:
    origin: float
    unix_origin_ms: float

    @classmethod
    def start(cls):
        return cls(time.monotonic(), time.time_ns() / 1_000_000)

    def seconds(self, mono=None):
        return (time.monotonic() if mono is None else mono) - self.origin

    def unix_ms(self, mono=None):
        return self.unix_origin_ms + self.seconds(mono) * 1000


def finite_point(point):
    return (point is not None and len(point) == 2 and
            all(v is not None and math.isfinite(float(v)) for v in point))


def rounded_reference_path(points, radius):
    """Tangent circular fillet at one corner, sampled at <= 2 px arc intervals."""
    a, corner, c = np.asarray(points, dtype=float)
    before, after = a-corner, c-corner
    lengths = np.linalg.norm(before), np.linalg.norm(after)
    if min(lengths) < 1e-6 or radius <= 0:
        return [a.tolist(), corner.tolist(), c.tolist()]
    u, v = before/lengths[0], after/lengths[1]
    angle = float(np.arccos(np.clip(np.dot(u,v),-1,1)))
    if angle < 1e-3 or math.pi-angle < 1e-3:
        return [a.tolist(), corner.tolist(), c.tolist()]
    # Limit the fillet to preserve straight guidance on short/custom routes.
    inset = min(radius/math.tan(angle/2), min(lengths)*.3)
    actual_radius = inset*math.tan(angle/2)
    entry, exit_point = corner+u*inset, corner+v*inset
    center = corner+(u+v)/np.linalg.norm(u+v)*actual_radius/math.sin(angle/2)
    first_angle = math.atan2(*(entry-center)[::-1])
    last_angle = math.atan2(*(exit_point-center)[::-1])
    sweep = (last_angle-first_angle+math.pi)%(2*math.pi)-math.pi
    count = max(3, math.ceil(abs(sweep)*actual_radius/2)+1)
    angles = np.linspace(first_angle,first_angle+sweep,count)
    arc = center+actual_radius*np.column_stack((np.cos(angles),np.sin(angles)))
    arc[0],arc[-1] = entry,exit_point
    return np.vstack((a,arc,c)).tolist()


def reference_path(route, returning=False):
    """Saved geometry is authoritative; legacy/custom anchor-only records stay sharp."""
    key = 'return_path' if returning else 'outbound_path'
    anchors = ('pickup','turn','drop') if returning else ('start','turn','pickup')
    return route.get(key, [route[k] for k in anchors])


def reference_route(width, height, bounds, custom=None):
    """Reference points are in the displayed (possibly mirrored) top image."""
    if custom is not None:
        points = np.asarray([custom[k] for k in ('start', 'turn', 'pickup', 'drop')], dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
            raise ValueError('Reference needs start, turn, pickup, drop as normalized [x, y] in 0..1')
        points *= [width - 1, height - 1]
    else:
        x0, x1, y0, y1 = bounds
        start = [x0 + .75 * (x1 - x0), y0 + .75 * (y1 - y0)]
        pickup = [x0 + .25 * (x1 - x0), y0 + .25 * (y1 - y0)]
        points = np.array([start, [start[0], pickup[1]], pickup, start])
    if np.linalg.norm(points[0] - points[2]) < 40:
        raise ValueError('Start and pickup must be at least 40 pixels apart')
    route = dict(zip(('start', 'turn', 'pickup', 'drop'), points.tolist()))
    route['geometry'] = 'rounded_corner'
    route['corner_radius_px'] = min(width,height)*.06
    route['outbound_path'] = rounded_reference_path(points[:3],route['corner_radius_px'])
    route['return_path'] = (list(reversed(route['outbound_path'])) if np.allclose(points[0],points[3]) else
                            rounded_reference_path(points[[2,1,3]],route['corner_radius_px']))
    return route


def point_path_distance(point, route):
    p, a, b = np.asarray(point), np.asarray(route[:-1]), np.asarray(route[1:])
    delta = b - a
    length2 = np.sum(delta * delta, axis=1)
    projection = np.clip(np.sum((p - a) * delta, axis=1) / np.maximum(length2, 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + projection[:, None] * delta), axis=1).min())


class Experiment:
    from .phase_marker import FIELDS as rgb_marker_fields
    grasp_fields = ['grasp_status', 'grasp_near', 'grasp_top_valid', 'side_pinch_raw',
                    'grip_top_openness', 'grip_side_openness', 'grip_source',
                    'grip_side_index', 'grip_side_middle', 'grip_side_ring', 'grip_side_little']
    display_fields = ['wrist_x_px', 'wrist_y_px', 'display_segment', 'display_trace_status',
                      'top_region_valid', 'top_region_source', 'top_region_age_ms']
    linked_fields = ['side_palm_x_px', 'side_palm_y_px', 'side_tracked_x_px', 'side_tracked_y_px',
                     'side_ball_x_px', 'side_ball_y_px', 'side_ball_rise_px',
                     'side_position_valid', 'side_tracking_status']
    spatial_fields = ['ball_world_x_mm', 'ball_world_y_mm', 'ball_world_z_mm',
                      'hand_world_x_mm', 'hand_world_y_mm', 'hand_world_z_mm',
                      'spatial_valid', 'spatial_status', 'depth_capture_t_s',
                      'depth_device_timestamp_us', 'depth_frame_index', 'depth_side_skew_ms',
                      'hand_pose_valid', 'hand_pose_source', 'hand_position_source', 'hand_position_t_s',
                      'ball_position_valid', 'marker_side_x_px', 'marker_side_y_px', 'marker_area_px',
                      'marker_depth_valid_pixels', 'marker_depth_mad_mm'] + [f'hand_basis_{a}{b}' for a in 'xyz' for b in 'xyz'] + [f'{prefix}_{axis}_mm' for prefix in ('goal_world', 'goal_from_hand') for axis in 'xyz']
    task_fields = ['task_phase', 'task_waypoint', 'hand_flip_angle_deg', 'flip_direction_verified',
                   'obstacle_contact', 'ball_captured', 'drop_prompt_valid', 'box_entry_from_below']
    trajectory_fields = ['session_id', 'trial_id', 't_s', 'top_capture_t_s', 'side_capture_t_s',
                         'phase', 'top_valid', 'side_valid', 'palm_x_px', 'palm_y_px',
                         'anchor_x_px', 'anchor_y_px', 'tracked_x_px', 'tracked_y_px',
                         'ball_x_px', 'ball_y_px', 'held', 'pinch_ratio', 'tracking_status',
                         'top_recording', 'side_recording'] + spatial_fields + linked_fields + ['display_palm_x_px', 'display_palm_y_px'] + grasp_fields + display_fields + ['operator_recording'] + task_fields + rgb_marker_fields
    event_fields = ['session_id', 'trial_id', 't_s', 'unix_ms', 'event', 'command_id', 'cue',
                    'role', 'commanded_voltage_mv', 'commanded_duration_s', 'frequency_hz',
                    'connection_id', 'direction_code', 'commanded_channels_json',
                    'repeat_id', 'repeat_index', 'repeat_interval_s', 'reason', 'detail']

    def __init__(self, directory, subject_id, session_id, *, clock=None, settings=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.subject_id, self.session_id = subject_id, session_id
        self.clock = clock or SessionClock.start()
        self.active = None
        self.trials = []
        self.samples = []
        settings = settings or {}
        self.trace_settings = dict(min_cutoff=settings.get('trail_min_cutoff', .65),
                                   beta=settings.get('trail_beta', .006), deadband_px=settings.get('trail_deadband_px', 3.))
        self.display_trace = DisplayPalmTrace(**self.trace_settings)
        self.closed = False
        self.command_observer = None
        self._last_flush = self.clock.origin
        self._last_recorded_s = 0.
        self._trajectory_file = (self.directory / 'trajectory.csv').open('w', newline='', encoding='utf-8')
        self._event_file = (self.directory / 'events.csv').open('w', newline='', encoding='utf-8')
        self._trajectory = csv.DictWriter(self._trajectory_file, self.trajectory_fields)
        self._events = csv.DictWriter(self._event_file, self.event_fields)
        self._trajectory.writeheader()
        self._events.writeheader()
        self.metadata = {
            'schema_version': 15, 'subject_id': subject_id, 'session_id': session_id,
            'clock_origin_monotonic_s': self.clock.origin, 'unix_origin_ms': self.clock.unix_origin_ms,
            'timestamp_basis': 'Host monotonic time. Camera timestamps are read-completion times, not exposure times.',
            'synchronization': 'Software alignment only; cameras are not hardware synchronized.',
            'coordinate_system': 'Displayed top image, x right, y down; pixels, no physical calibration.',
            'trajectory_anchor': 'Raw detected palm center; no filtered/frozen/optical-flow points in accuracy metrics.',
            'display_trajectory': 'display_palm_* uses the same filtered palm motion as ball translation, with image-flow validation and bounded landmark-loss bridging. No fixed wrist proxy; no 3D perspective/pitch correction. Raw palm_* remains the accuracy source. top_region_* distinguishes detected/flow/missing input.',
            'grasp_diagnostics': 'grasp_status/near/top_valid record the actual decision gates; side_pinch_raw precedes the median filter. Brief Side tip loss can retain a confirmed opening, never complete a grasp.',
            'reference_alignment': 'Fixed palm-minus-grasp-anchor offset at setup, or first valid top observation if absent at setup; never fitted to the path.',
            'reference_geometry': 'New trials freeze the rounded centerline in reference.outbound_path/return_path; display and evaluation use the same sampled curve. Stroke width is not an error tolerance.',
            'completion_time': 'First sent directional command (or simulated cue in vision-only mode) to detected release.',
            'voltage_duration': 'Commanded settings only; BLE writes do not measure physical actuator output.',
            'feedback_1hz': 'ABC only: one row per [t,t+1) host-clock second, generated at close from events.csv. Includes idle seconds, all short/multiple cues, failed/unavailable events and the final partial window. Logging never schedules hardware output.',
            'settings': settings or {},
        }
        self._save_metadata()
        self._save_trials()

    @property
    def trial_id(self):
        return self.active['trial_id'] if self.active else ''

    def _save_metadata(self):
        self._json('session.json', self.metadata)

    def _json(self, name, value):
        target = self.directory / name
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(target)

    def _save_trials(self):
        self._json('trials.json', self.trials)

    def event(self, event, *, mono=None, trial_id=None, **data):
        mono = time.monotonic() if mono is None else mono
        row = dict(session_id=self.session_id, trial_id=self.trial_id if trial_id is None else trial_id,
                   t_s=self.clock.seconds(mono), unix_ms=self.clock.unix_ms(mono), event=event)
        self._last_recorded_s = max(self._last_recorded_s, row['t_s'])
        extras = {k: v for k, v in data.items() if k not in self.event_fields}
        row.update({k: v for k, v in data.items() if k in self.event_fields})
        if extras:
            row['detail'] = json.dumps(extras, allow_nan=False)
        self._events.writerow(row)
        self._event_file.flush()

    def consume_commands(self, controller):
        for item in controller.drain_events():
            event, mono = item.pop('event'), item.pop('mono')
            self.event(event, mono=mono, **item)
            if self.command_observer is not None:
                self.command_observer(event, mono, item)
            if (self.active and item.get('trial_id') == self.trial_id and
                    event in ('command_sent', 'command_simulated') and
                    item.get('cue') in ('Forward', 'Back', 'Left', 'Right') and
                    self.active['start_t_s'] is None):
                self.active['start_t_s'] = self.clock.seconds(mono)
                self.active['simulated'] = event == 'command_simulated'
                self.event('task_started', mono=mono)
                self._save_trials()

    def begin(self, route, palm, anchor, frame_size, *, tolerance_px=45., mono=None):
        if self.active:
            raise ValueError('A trial is already active; finish it or press E first')
        mono = time.monotonic() if mono is None else mono
        offset = np.asarray(palm)-anchor if finite_point(palm) and finite_point(anchor) else None
        outbound = reference_path(route)
        inbound = reference_path(route, returning=True)
        trial = {'trial_id': f'{len(self.trials)+1:03d}', 'setup_t_s': self.clock.seconds(mono),
                 'start_t_s': None, 'end_t_s': None, 'grasp_t_s': None, 'release_t_s': None,
                 'result': 'active', 'success': False, 'simulated': False,
                 'reference': route, 'palm_offset_px': offset.tolist() if offset is not None else None,
                 'palm_offset_source': 'trial_setup' if offset is not None else 'pending_top_observation',
                 'palm_offset_t_s': self.clock.seconds(mono) if offset is not None else None,
                 'palm_outbound': (np.array(outbound)+offset).tolist() if offset is not None else None,
                 'palm_return': (np.array(inbound)+offset).tolist() if offset is not None else None,
                 'frame_size': list(frame_size), 'drop_tolerance_px': float(tolerance_px),
                 'videos': {}, 'manual_interventions': 0}
        self.active = trial
        self.trials.append(trial)
        self.clear_display_trace()
        self.event('trial_armed', mono=mono, reference=route, palm_offset_px=trial['palm_offset_px'])
        self._save_trials()
        return trial

    def clear_display_trace(self):
        """Clear the live path without deleting any recorded samples or trials."""
        self.samples = []
        self.display_trace = DisplayPalmTrace(**self.trace_settings)

    def set_videos(self, recorders, *, operator=None):
        if self.active:
            streams = dict(recorders.streams)
            if operator is not None and operator.active:
                streams['operator'] = operator
            for name, stream in streams.items():
                if stream.path:
                    path = str(stream.path.relative_to(self.directory))
                    paths = self.active['videos'].setdefault(name, [])
                    if path not in paths:
                        paths.append(path)
            self._save_trials()

    def sample(self, *, mono, top_mono, side_mono, observation, side_valid, tracked,
               ball, held, pinch_ratio, tracking_status, recordings, spatial=None, linked=None, grasp=None,
               anchor_valid=None, top_tracking=None, operator_recording=False, task=None, rgb_marker=None):
        if anchor_valid is None:
            anchor_valid = observation is not None and observation.pinch_ratio is not None
        if (self.active and self.active['palm_offset_px'] is None and observation is not None
                and anchor_valid and finite_point(observation.palm)
                and finite_point(observation.pinch_xy)):
            # Starting a trial does not require top detection. Capture the first usable
            # hand offset once for evaluation, without moving/fitting the reference route.
            offset = np.zeros(2) if self.active.get('task') == 'abc' else observation.palm-observation.pinch_xy
            self.active['palm_offset_px'] = offset.tolist()
            self.active['palm_offset_source'] = 'abc_fixed_palm_reference' if self.active.get('task') == 'abc' else 'first_valid_top_observation'
            self.active['palm_offset_t_s'] = self.clock.seconds(top_mono)
            for field, returning in [('palm_outbound', False), ('palm_return', True)]:
                self.active[field] = (np.array(reference_path(self.active['reference'], returning))+offset).tolist()
            self.event('palm_reference_initialized', mono=top_mono, palm_offset_px=offset.tolist())
            self._save_trials()
        phase = 'idle' if not self.active else ('ready' if self.active['start_t_s'] is None
                    else 'return' if self.active['grasp_t_s'] is not None else 'outbound')
        palm = observation.palm if observation is not None else (None, None)
        anchor = observation.pinch_xy if observation is not None else (None, None)
        tracked = tracked if tracked is not None else (None, None)
        row = dict(zip(self.trajectory_fields, [self.session_id, self.trial_id, self.clock.seconds(mono),
            self.clock.seconds(top_mono), self.clock.seconds(side_mono), phase,
            int(observation is not None), int(side_valid), *palm, *anchor, *tracked, *ball,
            int(held), pinch_ratio, tracking_status, int(recordings['top']), int(recordings['side'])]))
        row['operator_recording'] = int(operator_recording)
        palm_points = getattr(observation, 'palm_points', None)
        wrist = palm_points[0] if palm_points is not None else (None, None)
        # Production passes the already-filtered palm motion used by the ball.
        # Do not filter twice or substitute the wrist position for palm motion.
        if top_tracking is None:
            display = self.display_trace.update(palm if finite_point(palm) else None, top_mono,
                                               scale=getattr(observation, 'scale', 80.))
            segment, source = self.display_trace.segment_id, self.display_trace.status
        else:
            display = top_tracking['point']
            segment, source = top_tracking['segment'], top_tracking['source']
            row.update(top_region_valid=top_tracking['valid'], top_region_source=source,
                       top_region_age_ms=top_tracking['age_ms'])
        row.update(display_palm_x_px=display[0] if display is not None else None,
                   display_palm_y_px=display[1] if display is not None else None,
                   wrist_x_px=wrist[0], wrist_y_px=wrist[1],
                   display_segment=segment, display_trace_status=source)
        if spatial:
            row.update({k: spatial.get(k) for k in self.spatial_fields})
        if linked:
            row.update({k: linked.get(k) for k in self.linked_fields})
        if grasp:
            row.update({k: grasp.get(k) for k in self.grasp_fields})
        if task:
            row.update({k: task.get(k) for k in self.task_fields})
        if rgb_marker:
            row.update({k: rgb_marker.get(k) for k in self.rgb_marker_fields})
        self._trajectory.writerow(row)
        self._last_recorded_s = max(self._last_recorded_s, row['t_s'])
        if self.active:
            self.samples.append(row)
        if mono - self._last_flush >= 1:
            self._trajectory_file.flush()
            self._last_flush = mono

    def grasp(self, mono):
        self.event('grasp_detected', mono=mono)
        if self.active and self.active['grasp_t_s'] is None:
            self.active['grasp_t_s'] = self.clock.seconds(mono)
            self._save_trials()

    def release(self, ball, *, manual, position_valid, mono):
        self.event('release_detected', mono=mono, reason='manual' if manual else 'open_hand', ball=list(ball))
        if not self.active:
            return None
        trial = self.active
        trial['release_t_s'] = self.clock.seconds(mono)
        trial['placement_error_px'] = float(np.linalg.norm(np.asarray(ball) - trial['reference']['drop'])) if position_valid else None
        if manual:
            trial['manual_interventions'] += 1
        result = ('assisted' if manual else 'not_started' if trial['start_t_s'] is None
                  else 'no_grasp' if trial['grasp_t_s'] is None
                  else 'invalid_sequence' if trial['grasp_t_s'] < trial['start_t_s']
                  else 'unverified_position' if not position_valid
                  else 'success' if trial['placement_error_px'] <= trial['drop_tolerance_px'] else 'off_target')
        return self.finish(result, mono=mono)

    def finish(self, result='incomplete', *, mono=None):
        if not self.active:
            return None
        mono = time.monotonic() if mono is None else mono
        trial = self.active
        trial['end_t_s'], trial['result'] = self.clock.seconds(mono), result
        trial['success'] = result == 'success'
        completion = trial['end_t_s'] if trial.get('task') == 'abc' else trial['release_t_s']
        trial['completion_time_s'] = (completion - trial['start_t_s']
                                     if trial['success'] and trial['start_t_s'] is not None else None)
        trial['attempt_duration_s'] = (trial['end_t_s'] - trial['start_t_s']
                                      if trial['start_t_s'] is not None else None)
        self.event('trial_ended', mono=mono, reason=result)
        self.active = None
        self._trajectory_file.flush()
        self._save_trials()
        return trial

    def close(self):
        if self.closed:
            return
        self.finish('interrupted')
        self._trajectory_file.close()
        self._event_file.close()
        self.closed = True
        if self.metadata['settings'].get('scenario') == 'abc':
            from .feedback_log import write_feedback_seconds
            write_feedback_seconds(self.directory, self.session_id, self._last_recorded_s)


def draw_trial_overlay(frame, route, experiment, tolerance_px=45.):
    """Top-view reference and measured palm path, included in Display recordings."""
    import cv2
    from .operator_panel import (text, INK, MUTED, PAPER, LINE, _text_mask, draw_start_hand,
                                 REFERENCE_COLOR, TRAJECTORY_COLOR, rounded_rect, route_legend_width)
    scale = max(.85, min(1.35, min(frame.shape[1] / 640, frame.shape[0] / 480)))
    def path(points, color, width):
        points = np.rint(points).astype(np.int32)
        # White separation keeps both colors legible over light or dark camera scenes.
        cv2.polylines(frame, [points], False, PAPER, width+4, cv2.LINE_AA)
        cv2.polylines(frame, [points], False, color, width, cv2.LINE_AA)
    def label(message, center, baseline, pixels=12):
        width = _text_mask(message, pixels, True)[0].shape[1]
        x = max(8, min(frame.shape[1]-width-12, round(center-width/2)))
        y = min(frame.shape[0]-20, round(baseline))
        rounded_rect(frame, (x-7, y-pixels-4, width+14, pixels+10), PAPER, radius=7)
        text(frame, message, (x, y), pixels/34, INK, bold=True)
    def cross(point):
        x, y = np.round(point).astype(int)
        for a, b in [((x-7, y), (x+7, y)), ((x, y-7), (x, y+7))]:
            cv2.line(frame, a, b, PAPER, 7, cv2.LINE_AA)
            cv2.line(frame, a, b, REFERENCE_COLOR, 3, cv2.LINE_AA)
    path(reference_path(route), REFERENCE_COLOR, round(8*scale))
    separate_drop = not np.allclose(route['start'], route['drop'])
    if separate_drop:
        path(reference_path(route, returning=True), REFERENCE_COLOR, round(8*scale))
        cross(route['drop'])
        label('DROP', route['drop'][0], route['drop'][1]+28)
    segments, segment, previous_time, previous_segment = [], [], None, None
    for sample in experiment.samples[-1800:]:
        if not sample['top_valid'] and not sample.get('top_region_valid'):
            if len(segment) > 1:
                segments.append(segment)
            segment, previous_time = [], None
            continue
        display = (sample.get('display_palm_x_px', sample['palm_x_px']),
                   sample.get('display_palm_y_px', sample['palm_y_px']))
        if not finite_point(display):
            if len(segment) > 1:
                segments.append(segment)
            segment, previous_time = [], None
            continue
        current = tuple(np.round(display).astype(int))
        current_time = sample['top_capture_t_s']
        segment_id = sample.get('display_segment')
        if previous_time is not None and (not 0 <= current_time - previous_time <= .25
                                         or segment_id != previous_segment):
            if len(segment) > 1:
                segments.append(segment)
            segment = []
        segment.append(current)
        previous_time = current_time
        previous_segment = segment_id
    if len(segment) > 1:
        segments.append(segment)
    # Draw a continuous stroke per observed segment, so the white edging cannot
    # overwrite the previous segment's colored end cap and create a dotted trail.
    for segment in segments:
        path(segment, TRAJECTORY_COLOR, round(2*scale))
    # The open hand replaces the ambiguous start/drop circle. Its thumb-index gap
    # is placed at the route endpoint for demonstration, never as a start gate.
    x, y, w, h = draw_start_hand(frame, route['start'], height=round(96*scale))
    cross(route['start'])
    label('START' if separate_drop else 'START / DROP', x+w/2, y+h+10)
    label('PICKUP', route['pickup'][0], route['pickup'][1]+61)
    label('TURN', route['turn'][0], route['turn'][1]-15)
    # One header row above the workspace, with matching reference/observed colors.
    width = route_legend_width(frame.shape[1])
    x = frame.shape[1] - width - 8
    rounded_rect(frame, (x, 8, width, 30), PAPER, LINE, radius=10)
    names = ('Reference', 'Hand (tracked)') if width == 208 else ('Ref.', 'Hand')
    for offset, color, name in [(10, REFERENCE_COLOR, names[0]), (width//2+4, TRAJECTORY_COLOR, names[1])]:
        path([(x+offset, 23), (x+offset+18, 23)], color, 3)
        text(frame, name, (x+offset+25, 27), 11/34, MUTED)

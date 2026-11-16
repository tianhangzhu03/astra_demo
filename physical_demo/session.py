"""Append-only events and manual observations, separate from virtual-ball trials."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time


class SessionClock:
    def __init__(self):
        self.origin = time.monotonic()
        self.epoch = time.time()

    def seconds(self, mono=None):
        return (time.monotonic() if mono is None else mono) - self.origin

    def utc(self, mono=None):
        return datetime.fromtimestamp(self.epoch + self.seconds(mono), timezone.utc).isoformat()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class PilotSession:
    EVENT_FIELDS = ['event_id', 'session_t_s', 'utc', 'trial_id', 'event', 'source',
                    'command_id', 'cue', 'role', 'commanded_voltage_mv', 'commanded_duration_s',
                    'frequency_hz', 'marker_id', 'details_json']

    def __init__(self, config, *, synthetic=False):
        self.clock = SessionClock()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        subject = re.sub(r'[^A-Za-z0-9_.-]+', '_', config.subject).strip('._') or 'physical_pilot'
        self.directory = Path(config.log_root) / f'physical_{subject}_{stamp}'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.trials, self.active, self.segments = [], None, []
        self.closed = False
        self._event_count = self._marker_count = 0
        self._events_file = (self.directory / 'events.csv').open('w', newline='', encoding='utf-8')
        self._events = csv.DictWriter(self._events_file, fieldnames=self.EVENT_FIELDS)
        self._events.writeheader()
        self.metadata = dict(schema_version=1, mode='physical_object_manual',
            session_id=self.directory.name, started_utc=self.clock.utc(),
            synthetic=synthetic, subject=config.subject,
            object=dict(name=config.object_name, notes=config.object_notes, geometry='unspecified'),
            cameras=dict(top=config.top_camera, side=config.side_camera,
                         raw_video_mirrored=False, ui_top_mirrored=False, ui_side_mirrored=config.side_mirror),
            haptics=dict(enabled=config.ble_enabled, address=config.ble_address, uuid=config.ble_uuid,
                         protocol='CEL_GUI 9-byte', initial_voltage_mv=0,
                         annotation_triggers_output=False, automatic_confirmation=False,
                         normal_semantics=dict(carrier='Same Static Normal direction=0 packet for LOWER and LIFT',
                             sequence='LOWER then LIFT, alternate only after command_sent or command_simulated',
                             override='Tab selects next meaning without output',
                             interpretation='Learned movement cue; not opposite force polarity or measured Z'),
                         sent_means='BLE write succeeded; not a measurement of physical force or response'),
            recording=dict(fps=config.recording_fps, segments=self.segments,
                           camera_content='Unannotated camera pixels encoded as lossy MP4; not sensor RAW',
                           ui_content='Full operator canvas; side mirror and optional route guide only here'),
            timing=dict(origin='Host monotonic clock', wall_clock='UTC epoch sampled at session start',
                        capture='Host frame-receipt time, not synchronized camera exposure',
                        markers='Operator key/click time; reaction delay is not measured',
                        alignment='Use session timestamps and per-video CSV; each segment has its own origin'),
            measurements=dict(object_detection=False, hand_detection=False, trajectory=False,
                              depth=False, position_error=False, automatic_success=False))
        self.save()
        self.event('session_started', source='system')

    @property
    def trial_id(self):
        return self.active['trial_id'] if self.active else ''

    @property
    def object_state(self):
        if not self.active:
            return 'UNMARKED'
        marks = [m for m in self.active['markers'] if not m['undone']]
        return marks[-1]['kind'].upper() if marks else 'UNMARKED'

    def event(self, event, *, mono=None, trial_id=None, source='operator', **data):
        if self.closed:
            return
        mono = time.monotonic() if mono is None else mono
        self._event_count += 1
        row = dict(event_id=f'E{self._event_count:06d}', session_t_s=self.clock.seconds(mono),
                   utc=self.clock.utc(mono), trial_id=self.trial_id if trial_id is None else trial_id,
                   event=event, source=source)
        for key in self.EVENT_FIELDS:
            if key in data:
                row[key] = data.pop(key)
        row['details_json'] = json.dumps(data, ensure_ascii=False, allow_nan=False)
        self._events.writerow(row)
        self._events_file.flush()

    def save(self):
        write_json(self.directory / 'session.json', self.metadata)
        write_json(self.directory / 'trials.json', self.trials)
        fields = ['trial_id', 'status', 'setup_t_s', 'first_direction_t_s', 'timing_source',
                  'manual_grasp_t_s', 'manual_release_t_s', 'end_t_s',
                  'direction_to_release_s', 'manual_hold_s', 'flags', 'segments']
        temporary = self.directory / 'trials.csv.tmp'
        with temporary.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for trial in self.trials:
                row = {key: trial.get(key, '') for key in fields}
                row['flags'] = '|'.join(row['flags'])
                row['segments'] = '|'.join(row['segments'])
                writer.writerow(row)
        temporary.replace(self.directory / 'trials.csv')

    def start_trial(self, segment_id, *, mono=None, evidence=None):
        mono = time.monotonic() if mono is None else mono
        if self.active:
            if self.object_state != 'RELEASED':
                return False, 'Trial active. End [E] or reset [0] before starting again.'
            self.finish_trial('manually_released', mono=mono)
        trial = dict(trial_id=f'{len(self.trials)+1:03d}', status='active',
                     setup_t_s=self.clock.seconds(mono), first_direction_t_s=None, timing_source=None,
                     manual_grasp_t_s=None, manual_release_t_s=None, end_t_s=None,
                     direction_to_release_s=None, manual_hold_s=None,
                     segments=[segment_id], flags=[], markers=[])
        self.active = trial
        self.trials.append(trial)
        self.event('trial_started', mono=mono, evidence=evidence or {})
        self.save()
        return True, f'Trial {trial["trial_id"]} ready. First sent direction starts timing.'

    def _refresh(self, trial):
        marks = [m for m in trial['markers'] if not m['undone']]
        grasp = next((m['t_s'] for m in marks if m['kind'] == 'grasped'), None)
        release = next((m['t_s'] for m in marks if m['kind'] == 'released'), None)
        start = trial['first_direction_t_s']
        trial.update(manual_grasp_t_s=grasp, manual_release_t_s=release,
                     manual_hold_s=release-grasp if release is not None and grasp is not None and release >= grasp else None,
                     direction_to_release_s=release-start if release is not None and start is not None and release >= start else None)
        trial['flags'] = [label for missing, label in ((grasp is None, 'grasp_not_marked'),
                          (release is None, 'release_not_marked'), (start is None, 'direction_not_sent'),
                          (start is not None and release is not None and start > release, 'release_before_direction')) if missing]

    def mark(self, kind, *, mono=None, evidence=None):
        mono = time.monotonic() if mono is None else mono
        if kind not in ('grasped', 'released'):
            raise ValueError('Unknown manual marker')
        if not self.active:
            self.event('annotation_rejected', kind=kind, reason='No active trial')
            return False, 'Press N to start a trial before marking.'
        if self.object_state == kind.upper() or self.object_state == 'RELEASED':
            return False, 'Already marked. Undo [Z] to correct the last annotation.'
        self._marker_count += 1
        marker = dict(marker_id=f'M{self._marker_count:05d}', kind=kind,
                      t_s=self.clock.seconds(mono), undone=False, evidence=evidence or {})
        self.active['markers'].append(marker)
        self.event('manual_' + kind, mono=mono, marker_id=marker['marker_id'], evidence=evidence or {})
        self._refresh(self.active)
        self.save()
        return True, f'{kind.title()} marked by operator. No haptic output from this mark.'

    def undo(self, *, mono=None):
        mono = time.monotonic() if mono is None else mono
        marks = [m for m in self.active['markers'] if not m['undone']] if self.active else []
        if not marks:
            return False, 'No manual annotation to undo in this trial.'
        mark = marks[-1]
        mark.update(undone=True, undone_t_s=self.clock.seconds(mono))
        self.event('manual_marker_undone', mono=mono, marker_id=mark['marker_id'], kind=mark['kind'])
        self._refresh(self.active)
        self.save()
        return True, 'Last manual annotation undone; original event retained.'

    def ingest_haptic(self, item):
        item = dict(item)
        event, mono = item.pop('event'), item.pop('mono')
        trial_id = item.pop('trial_id', '')
        self.event(event, mono=mono, trial_id=trial_id, source='ble' if self.metadata['haptics']['enabled'] else 'simulation', **item)
        if event not in ('command_sent', 'command_simulated') or item.get('cue') not in ('Forward', 'Back', 'Left', 'Right'):
            return
        trial = next((t for t in self.trials if t['trial_id'] == trial_id), None)
        at = self.clock.seconds(mono)
        if (trial and trial['first_direction_t_s'] is None and at >= trial['setup_t_s']
                and (trial['end_t_s'] is None or at <= trial['end_t_s'])):
            trial['first_direction_t_s'] = at
            trial['timing_source'] = 'ble_write' if event == 'command_sent' else 'simulated_direction'
            self._refresh(trial)
            self.save()

    def finish_trial(self, reason=None, *, mono=None):
        if not self.active:
            return
        mono = time.monotonic() if mono is None else mono
        trial = self.active
        self._refresh(trial)
        trial.update(status=reason or ('manually_released' if trial['manual_release_t_s'] is not None else 'incomplete'),
                     end_t_s=self.clock.seconds(mono))
        self.event('trial_ended', mono=mono, status=trial['status'])
        self.active = None
        self.save()

    def close(self, reason='normal'):
        if self.closed:
            return
        if self.active:
            self.finish_trial(None if self.object_state == 'RELEASED' and reason == 'normal' else 'interrupted')
        self.event('session_ended', source='system', reason=reason)
        self.metadata.update(ended_utc=self.clock.utc(), end_reason=reason, event_count=self._event_count)
        self.save()
        self._events_file.close()
        self.closed = True

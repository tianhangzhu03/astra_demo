"""Continuous three-video segments with explicit source-frame provenance."""

from collections import deque
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class Sample:
    image: np.ndarray
    mono: float
    sequence: int
    trial_id: str = ''
    sources: dict = field(default_factory=dict)


class VideoStream:
    FIELDS = ['frame_index', 'video_t_s', 'capture_t_s', 'sample_id', 'repeated_frame',
              'trial_id', 'write_t_s', 'top_frame_id', 'top_capture_t_s',
              'side_frame_id', 'side_capture_t_s', 'dropped_updates']

    def __init__(self, path, fps, clock, initial):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('FPS must be positive and finite')
        self.path, self.fps, self.clock = Path(path), fps, clock
        self.size = initial.image.shape[1::-1]
        if any(d % 2 for d in self.size):
            raise ValueError('MP4 dimensions must be even')
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, self.size)
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError(f'Cannot open video encoder: {path}')
        try:
            self._file = self.path.with_suffix('.csv').open('x', newline='', encoding='utf-8')
        except Exception:
            self._writer.release()
            raise
        self._condition = threading.Condition()
        self._pending = deque()
        self._current = initial
        self._last_submitted = initial.sequence
        self._thread = None
        self._stop_at = None
        self.frames = self.repeated = self.dropped = 0
        self.error = ''

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, mono):
        self._start = mono
        self._thread = threading.Thread(target=self._run, name=f'physical-video-{self.path.stem}', daemon=True)
        self._thread.start()

    def submit(self, sample):
        with self._condition:
            if self._stop_at is not None or sample.sequence == self._last_submitted:
                return
            self._last_submitted = sample.sequence
            if len(self._pending) >= 8:
                self._pending.popleft()
                self.dropped += 1
            # Images are immutable after capture/render; no unnecessary full-size copy.
            self._pending.append(sample)
            self._condition.notify_all()

    def request_stop(self, mono):
        with self._condition:
            self._stop_at = mono
            self._condition.notify_all()

    def close(self):
        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                self.error = 'Video writer still closing; final file may be incomplete'
        else:
            self._writer.release()
            self._file.close()

    def _run(self):
        writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        previous = None
        try:
            writer.writeheader()
            while True:
                target = self._start + self.frames / self.fps
                with self._condition:
                    while time.monotonic() < target and self._stop_at is None:
                        self._condition.wait(timeout=max(.001, target-time.monotonic()))
                    if self._stop_at is not None and target >= self._stop_at:
                        break
                    # A later source frame must never be backfilled into an earlier video slot.
                    while self._pending and self._pending[0].mono <= target:
                        self._current = self._pending.popleft()
                    sample = self._current
                    dropped = self.dropped
                if sample.image.shape[1::-1] != self.size:
                    raise RuntimeError('Camera resolution changed during recording')
                self._writer.write(sample.image)
                repeated = sample.sequence == previous
                self.repeated += int(repeated)
                row = dict(frame_index=self.frames, video_t_s=self.clock.seconds(target),
                           capture_t_s=self.clock.seconds(sample.mono), sample_id=sample.sequence,
                           repeated_frame=int(repeated), trial_id=sample.trial_id,
                           write_t_s=self.clock.seconds(), dropped_updates=dropped)
                row.update(sample.sources)
                writer.writerow(row)
                previous = sample.sequence
                self.frames += 1
                if self.frames % 30 == 0:
                    self._file.flush()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self._writer.release()
            self._file.close()

    def describe(self, root):
        return dict(file=self.path.relative_to(root).as_posix(),
                    timestamps=self.path.with_suffix('.csv').relative_to(root).as_posix(),
                    size=list(self.size), fps=self.fps, frames=self.frames,
                    repeated_frames=self.repeated, dropped_updates=self.dropped,
                    error=self.error, finalized=not self.active)


class RecordingGroup:
    def __init__(self, session, fps=30):
        self.session, self.fps = session, fps
        self.streams = {}
        self.trial_id = ''
        self.segment = None

    @property
    def active(self):
        return bool(self.streams) and all(s.active for s in self.streams.values())

    @property
    def error(self):
        return '; '.join(f'{name}: {s.error or "writer stopped"}' for name, s in self.streams.items()
                         if s.error or not s.active)

    @property
    def segment_id(self):
        return self.segment['segment_id'] if self.active else ''

    def start(self, top, side, operator):
        if self.streams:
            return False, 'Recording already active or needs closing'
        if top is None or side is None:
            return False, 'Both camera frames are required'
        number = 1
        videos = self.session.directory / 'videos'
        while (videos / f'segment_{number:03d}').exists():
            number += 1
        segment_id = f'segment_{number:03d}'
        directory = videos / segment_id
        directory.mkdir(parents=True, exist_ok=False)
        samples = dict(top=Sample(top.image, top.mono, top.sequence, self.trial_id),
                       side=Sample(side.image, side.mono, side.sequence, self.trial_id), operator=operator)
        streams = {}
        try:
            for name, sample in samples.items():
                streams[name] = VideoStream(directory / f'{name}.mp4', self.fps, self.session.clock, sample)
            origin = time.monotonic()
            for stream in streams.values():
                stream.start(origin)
        except Exception as exc:
            for stream in streams.values():
                stream.request_stop(time.monotonic())
                stream.close()
            self.session.event('recording_failed', source='system', reason=str(exc), segment_id=segment_id)
            return False, str(exc)
        self.streams = streams
        self.segment = dict(segment_id=segment_id, start_t_s=self.session.clock.seconds(origin), end_t_s=None,
                            status='recording', videos={name: s.describe(self.session.directory) for name, s in streams.items()})
        self.session.segments.append(self.segment)
        self.session.event('recording_started', mono=origin, source='system', segment_id=segment_id)
        self.session.save()
        return True, f'Recording all three videos: {segment_id}'

    def camera_frame(self, name, frame):
        stream = self.streams.get(name)
        if stream:
            stream.submit(Sample(frame.image, frame.mono, frame.sequence, self.trial_id))

    def operator_frame(self, sample):
        stream = self.streams.get('operator')
        if stream:
            stream.submit(sample)

    def stop(self, reason='operator'):
        if not self.streams:
            return
        streams, self.streams = self.streams, {}
        now = time.monotonic()
        for stream in streams.values():
            stream.request_stop(now)
        for stream in streams.values():
            stream.close()
        videos = {name: s.describe(self.session.directory) for name, s in streams.items()}
        self.segment.update(end_t_s=self.session.clock.seconds(now),
                            status='error' if any(v['error'] for v in videos.values()) else 'closed',
                            stop_reason=reason, videos=videos)
        self.session.event('recording_stopped', mono=now, source='system',
                           segment_id=self.segment['segment_id'], reason=reason, videos=videos)
        self.session.save()

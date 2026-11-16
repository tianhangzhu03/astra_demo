"""Independent video writers. Window resizing/occlusion never affects the saved frames."""

import csv
import math
from pathlib import Path
import threading
import time

import cv2


class StreamRecorder:
    def __init__(self, directory, name, fps=30.0, clock=None, session_id=''):
        self.directory, self.name, self.fps = Path(directory), name, float(fps)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Recording FPS must be positive and finite')
        self.path = None
        self.error = ''
        self.frames_written = 0
        self.duplicates = 0
        self._condition = threading.Condition()
        self._thread = None
        self._latest = None
        self._stop_at = None
        self._started = 0.0
        self._epoch_ms = 0
        self.clock, self.session_id = clock, session_id

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def status(self):
        if self.error:
            return f'Error: {self.error}'
        if self.active:
            return f'REC {max(0, time.monotonic() - self._started):.0f}s'
        return f'Saved {self.path.name}' if self.path else 'Ready'

    def start(self, frame, now=None, epoch_ms=None, *, capture_mono=None, capture_epoch_ms=None, trial_id='', source_times=None):
        if self.active:
            return False
        if frame is None:
            self.error = 'No camera frame'
            return False
        self.directory.mkdir(parents=True, exist_ok=True)
        number = 1
        while (self.directory / f'{self.name}_{number:03d}.mp4').exists():
            number += 1
        self.path = self.directory / f'{self.name}_{number:03d}.mp4'
        self.size = (frame.shape[1], frame.shape[0])
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        if not writer.isOpened():
            writer.release()
            self.error = 'Video encoder unavailable'
            return False
        try:
            timing_file = self.path.with_suffix('.csv').open('w', newline='', encoding='utf-8')
        except OSError as exc:
            writer.release()
            self.error = str(exc)
            return False
        self._started = time.monotonic() if now is None else now
        self._epoch_ms = int(time.time() * 1000) if epoch_ms is None else epoch_ms
        self.frames_written = self.duplicates = 0
        self.error, self._stop_at = '', None
        self._latest = (frame.copy(), self._epoch_ms if capture_epoch_ms is None else capture_epoch_ms,
                        self._started if capture_mono is None else capture_mono, trial_id, dict(source_times or {}))
        self._thread = threading.Thread(target=self._run, args=(writer, timing_file),
                                        name=f'record-{self.name}', daemon=False)
        self._thread.start()
        return True

    def submit(self, frame, capture_epoch_ms, capture_mono=None, trial_id='', source_times=None):
        if not self.active or frame is None:
            return
        with self._condition:
            self._latest = (frame.copy(), capture_epoch_ms,
                            time.monotonic() if capture_mono is None else capture_mono, trial_id, dict(source_times or {}))
            self._condition.notify()

    def stop(self, now=None):
        if self._thread is None:
            return
        with self._condition:
            self._stop_at = time.monotonic() if now is None else now
            self._condition.notify()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            self.error = 'Writer is still closing; do not exit yet'

    def _run(self, writer, timing_file):
        csv_writer = csv.writer(timing_file)
        previous_timestamp = None
        try:
            csv_writer.writerow(['frame_index', 'video_unix_ms', 'capture_unix_ms', 'repeated_frame',
                                 'session_id', 'trial_id', 'video_t_s', 'capture_t_s',
                                 'top_capture_t_s', 'side_capture_t_s'])
            while True:
                target = self._started + self.frames_written / self.fps
                with self._condition:
                    while time.monotonic() < target and self._stop_at is None:
                        self._condition.wait(timeout=max(.001, target - time.monotonic()))
                    if self._stop_at is not None and target >= self._stop_at:
                        break
                    frame, timestamp, capture_mono, trial_id, source_times = self._latest
                if frame.shape[1::-1] != self.size:
                    raise RuntimeError('Camera resolution changed; recording stopped')
                writer.write(frame)
                repeated = timestamp == previous_timestamp
                self.duplicates += int(repeated)
                csv_writer.writerow([self.frames_written,
                                     round(self._epoch_ms + self.frames_written * 1000 / self.fps, 3),
                                     timestamp, int(repeated), self.session_id, trial_id,
                                     self.clock.seconds(target) if self.clock else '',
                                     self.clock.seconds(capture_mono) if self.clock else '',
                                     source_times.get('top', ''), source_times.get('side', '')])
                previous_timestamp = timestamp
                self.frames_written += 1
                if self.frames_written % 30 == 0:
                    timing_file.flush()
        except Exception as exc:
            self.error = str(exc)
        finally:
            writer.release()
            timing_file.close()


class DualRecorder:
    def __init__(self, directory, fps=30.0, clock=None, session_id=''):
        self.streams = {name: StreamRecorder(directory, name, fps, clock, session_id) for name in ('top', 'side')}

    def start_both(self, frames, now, epoch_ms, *, capture_times=None, timestamps=None, trial_id=''):
        """Trial startup is all-or-nothing; never silently keep an older segment."""
        if any(stream.active for stream in self.streams.values()):
            return False
        if any(frames.get(name) is None for name in self.streams):
            return False
        started = [stream.start(frames[name], now, epoch_ms,
                     capture_mono=(capture_times or {}).get(name),
                     capture_epoch_ms=(timestamps or {}).get(name), trial_id=trial_id)
                   for name, stream in self.streams.items()]
        if not all(started):
            self.close()
            return False
        return True

    def toggle(self, which, frames, now, epoch_ms, *, capture_times=None, timestamps=None, trial_id=''):
        names = ('top', 'side') if which == 'both' else (which,)
        stop = all(self.streams[name].active for name in names)
        for name in names:
            stream = self.streams[name]
            if stop:
                stream.stop(now)
            elif not stream.active:
                stream.start(frames.get(name), now, epoch_ms,
                             capture_mono=(capture_times or {}).get(name),
                             capture_epoch_ms=(timestamps or {}).get(name), trial_id=trial_id)

    def submit(self, frames, timestamps, capture_times=None, trial_id=''):
        for name, stream in self.streams.items():
            stream.submit(frames.get(name), timestamps.get(name), (capture_times or {}).get(name), trial_id)

    def close(self):
        now = time.monotonic()
        for stream in self.streams.values():
            stream.stop(now)

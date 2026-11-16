"""Two independent image-plane hand trails; no grasp decision or world XYZ."""

from collections import deque
import csv
from pathlib import Path
from queue import SimpleQueue, Empty
import threading
import time

import cv2
import numpy as np

from .palm_motion import HandObservation, PalmPathTracker


def palm_observation(landmarks, width, height):
    if landmarks is None or len(landmarks) != 21:
        return None
    points = np.asarray([(p.x*width, p.y*height) for p in landmarks], dtype=float)[[0, 5, 9, 13, 17]]
    if (not np.isfinite(points).all() or (points < [-.1*width, -.1*height]).any()
            or (points > [1.1*width, 1.1*height]).any()):
        return None
    length = float(np.linalg.norm(points[2]-points[0]))
    scale = max(length, float(np.linalg.norm(points[1]-points[4])))
    if length < 6 or scale > max(width, height)*.8:
        return None
    palm = points.mean(axis=0)
    return HandObservation(palm, np.eye(2), scale, palm.copy(), None, None, points)


class HandDetector:
    def __init__(self, model_path):
        import mediapipe as mp
        self.mp = mp
        self.last_ms = -1
        self.legacy = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')
        if self.legacy:
            self.detector = mp.solutions.hands.Hands(max_num_hands=1,
                min_detection_confidence=.5, min_tracking_confidence=.5)
        else:
            from mediapipe.tasks.python import BaseOptions, vision
            if not Path(model_path).is_file():
                raise RuntimeError('Hand model missing: ' + str(model_path))
            self.detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)), running_mode=vision.RunningMode.VIDEO,
                num_hands=1, min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
                min_tracking_confidence=.5))

    def observe(self, frame, mono):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.legacy:
            result = self.detector.process(rgb)
            hands = result.multi_hand_landmarks
            landmarks = hands[0].landmark if hands else None
        else:
            self.last_ms = max(self.last_ms+1, round(mono*1000))
            result = self.detector.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.last_ms)
            landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
        return palm_observation(landmarks, frame.shape[1], frame.shape[0])

    def close(self):
        self.detector.close()


class PalmTraceProcessor:
    def __init__(self, detector, inference_width=640):
        self.detector, self.inference_width = detector, inference_width
        self.motion = PalmPathTracker(flow_hold_s=.35)
        self.size = None
        self.size_epoch = 0

    def process(self, frame, *, view, trial_id='', epoch=0):
        width, height = frame.image.shape[1::-1]
        if self.size is not None and self.size != (width, height):
            self.motion = PalmPathTracker(flow_hold_s=.35)
            self.size_epoch += 1
        self.size = (width, height)
        sample_width = min(width, self.inference_width)
        sample_height = max(1, round(height*sample_width/width))
        sample = cv2.resize(frame.image, (sample_width, sample_height)) if sample_width != width else frame.image
        observation = self.detector.observe(sample, frame.mono)
        self.motion.observe(sample, observation, frame.mono)
        result = self.motion.record()
        point = result['point']
        sx, sy = width/sample_width, height/sample_height
        return dict(view=view, source_frame_id=frame.sequence, capture_mono=frame.mono,
                    processed_mono=time.monotonic(), trial_id=trial_id, trace_epoch=epoch,
                    stroke_id=self.size_epoch*1000000+result['segment'], width=width, height=height,
                    valid=int(point is not None), source=result['source'],
                    raw_palm_x_px=float(observation.palm[0]*sx) if observation else None,
                    raw_palm_y_px=float(observation.palm[1]*sy) if observation else None,
                    display_palm_x_px=float(point[0]*sx) if point else None,
                    display_palm_y_px=float(point[1]*sy) if point else None,
                    landmark_age_ms=result['age_ms'], error='')


class TraceHistory:
    def __init__(self):
        self.points = deque(maxlen=6000)
        self.latest = None

    def add(self, row):
        self.latest = row
        if not row['valid']:
            return
        point = (row['stroke_id'], row['display_palm_x_px']/row['width'],
                 row['display_palm_y_px']/row['height'], row['capture_mono'])
        if self.points:
            previous = self.points[-1]
            distance = np.hypot((point[1]-previous[1])*row['width'], (point[2]-previous[2])*row['height'])
            if point[0] == previous[0] and distance < 2 and point[3]-previous[3] < .2:
                return
        self.points.append(point)

    def snapshot(self, now):
        latest = self.latest
        current = None
        status = 'WAITING'
        if latest:
            age = now-latest['capture_mono']
            status = 'STALE' if age > .5 else latest['source']
            if latest['valid'] and 0 <= age <= .5:
                current = (latest['display_palm_x_px']/latest['width'], latest['display_palm_y_px']/latest['height'])
        return dict(points=list(self.points), current=current, status=status,
                    source_frame_id=latest['source_frame_id'] if latest else None)


class HandTraceService:
    FIELDS = ['view', 'source_frame_id', 'capture_t_s', 'processed_t_s', 'trial_id', 'trace_epoch',
              'stroke_id', 'width', 'height', 'valid', 'source', 'raw_palm_x_px', 'raw_palm_y_px',
              'display_palm_x_px', 'display_palm_y_px', 'landmark_age_ms', 'skipped_input_frames', 'error']

    def __init__(self, session, config, *, detector_factory=HandDetector):
        self.session, self.config, self.detector_factory = session, config, detector_factory
        self.epoch = 0
        self.histories = {view: TraceHistory() for view in ('top', 'side')}
        self._condition = threading.Condition()
        self._pending, self._submitted = {}, {}
        self._results = SimpleQueue()
        self._stop = False
        self.error = ''
        self._file = (session.directory/'hand_trace.csv').open('x', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        session.metadata['hand_trace'] = dict(enabled=True, status='initializing', file='hand_trace.csv',
            coordinates='Unmirrored source-image pixels in each view; not XYZ or object trajectory',
            source='Palm landmarks with <=350 ms validated optical-flow bridging',
            raw='Landmark palm before image-flow correction and display smoothing',
            reset='N and 0 clear the displayed trail; previous CSV rows are retained',
            height='Side image vertical movement is not calibrated world height',
            inference_width=config.hand_inference_width, model=str(config.hand_model),
            grasp_control=False, displayed_history_limit=6000)
        session.metadata['measurements'].update(hand_detection=True, hand_trace_2d=True)
        session.save()
        self._thread = threading.Thread(target=self._run, name='physical-hand-traces', daemon=True)
        self._thread.start()

    def submit(self, frames, trial_id):
        if self.error:
            return
        with self._condition:
            for view, frame in frames.items():
                if frame is not None and self._submitted.get(view) != frame.sequence:
                    self._submitted[view] = frame.sequence
                    self._pending[view] = (frame, trial_id, self.epoch)
            self._condition.notify_all()

    def reset(self):
        self.epoch += 1
        self.histories = {view: TraceHistory() for view in ('top', 'side')}
        self.session.event('hand_trace_cleared', trace_epoch=self.epoch)

    def _run(self):
        detectors, processors, epochs, previous = {}, {}, {}, {}
        turn = 'top'
        try:
            for view in ('top', 'side'):
                detectors[view] = self.detector_factory(self.config.hand_model)
            self._results.put({'ready': True})
            while True:
                with self._condition:
                    while not self._pending and not self._stop:
                        self._condition.wait()
                    if self._stop:
                        break
                    view = turn if turn in self._pending else next(iter(self._pending))
                    frame, trial_id, epoch = self._pending.pop(view)
                    turn = 'side' if view == 'top' else 'top'
                if epochs.get(view) != epoch:
                    processors[view] = PalmTraceProcessor(detectors[view], self.config.hand_inference_width)
                    epochs[view] = epoch
                row = processors[view].process(frame, view=view, trial_id=trial_id, epoch=epoch)
                row['skipped_input_frames'] = max(0, frame.sequence-previous.get(view, frame.sequence)-1)
                previous[view] = frame.sequence
                self._results.put(row)
        except Exception as exc:
            self._results.put({'fatal_error': str(exc)})
        finally:
            for detector in detectors.values():
                detector.close()

    def consume(self):
        while True:
            try:
                row = self._results.get_nowait()
            except Empty:
                break
            if row.get('ready'):
                self.session.metadata['hand_trace']['status'] = 'ready'
                self.session.event('hand_trace_ready', source='vision')
                self.session.save()
                continue
            if 'fatal_error' in row:
                self.error = row['fatal_error']
                self.session.metadata['hand_trace'].update(status='error', error=self.error)
                self.session.event('hand_trace_error', source='vision', reason=self.error)
                self.session.save()
                continue
            record = {key: row.get(key) for key in self.FIELDS}
            record.update(capture_t_s=self.session.clock.seconds(row['capture_mono']),
                          processed_t_s=self.session.clock.seconds(row['processed_mono']))
            self._writer.writerow(record)
            if row['trace_epoch'] == self.epoch:
                self.histories[row['view']].add(row)
        self._file.flush()

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        return {view: history.snapshot(now) for view, history in self.histories.items()}

    def close(self):
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=5)
        self.consume()
        self._file.close()
        self.session.metadata['hand_trace']['worker_closed'] = not self._thread.is_alive()
        self.session.save()

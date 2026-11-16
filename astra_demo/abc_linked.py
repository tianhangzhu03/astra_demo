"""ABC on the established linked RGB views: image pixels, no depth or calibration."""

import time

import cv2
import numpy as np

from .abc_scene import REACH, CARRY, OBSTACLES, BOX_WIDTH, BALL_COLOR, dashed_path
from .abc_task import AbcTask
from .sphere_renderer import draw_sphere


class LinkedAbcScene:
    def __init__(self):
        self.top_size = self.side_size = None
        self.side_origin = self.box_center = None
        self.box_manual = False
        self.revision = 0

    def reference(self, size):
        size = tuple(size)
        if size != self.top_size:
            self.top_size = size
            self.scale = min((size[0]-40)/14.4, (size[1]-60)/8.8)
            self.center = np.array([size[0]/2, size[1]/2+8])
            self.radius = .29*1.25*self.scale
            self.paths = [self.top_pixels(path[:, :2]) for path in (REACH, CARRY)]
            corners = OBSTACLES[:, :2, None] + np.array([-1, 1])*OBSTACLES[:, 2:4, None]/2
            self.obstacle_min = self.top_pixels(corners[:, :, 0])
            self.obstacle_max = self.top_pixels(corners[:, :, 1])
            self.obstacle_min, self.obstacle_max = (np.minimum(self.obstacle_min, self.obstacle_max),
                                                   np.maximum(self.obstacle_min, self.obstacle_max))
            self.revision += 1
        reach, carry = [path.tolist() for path in self.paths]
        return dict(start=reach[0], turn=reach[1], pickup=reach[-1], drop=carry[-1],
                    geometry='abc_linked_pixels', outbound_path=reach, return_path=carry)

    def top_pixels(self, xy):
        return self.center + np.asarray(xy)*[self.scale, -self.scale]

    def place_side(self, origin, size):
        keep_box = self.box_manual and tuple(size) == self.side_size
        self.side_origin, self.side_size = np.asarray(origin, dtype=float), tuple(size)
        self.side_radius = max(8, round(31*size[1]/480))
        self.box_half = np.array([2.1, 1.7])*self.side_radius
        if not keep_box:
            self.place_box(self.side_origin + [size[0]*.28, -3.8*self.side_radius], manual=False)
        else:
            self.revision += 1

    def place_box(self, point, *, manual=True):
        if self.side_origin is None:
            raise ValueError('Left-click the Side pickup ball before positioning the box')
        # Leave image space below the opening for the hand to enter and withdraw.
        point = np.asarray(point, dtype=float)
        low, high = self.box_half+8, np.asarray(self.side_size)-self.box_half-[8, 2*self.side_radius+12]
        if manual and (not np.isfinite(point).all() or np.any(point < low) or np.any(point > high)):
            raise ValueError('Click farther inside Side View; leave room below the box for the hand')
        self.box_center = np.clip(point, low, high)
        self.box_manual = manual
        self.revision += 1

    @property
    def opening_y(self):
        return self.box_center[1]+self.box_half[1]

    def inside_top(self, point):
        return bool(np.all(np.abs(np.asarray(point)-self.paths[1][-1]) <= BOX_WIDTH*self.scale/2-self.radius))

    def inside_side(self, point):
        return bool(np.all(np.abs(np.asarray(point)-self.box_center) <= self.box_half-self.side_radius-5))

    def collision(self, palm, ball, held):
        points = np.array([palm, ball] if held else [palm])
        radii = np.array([.16*self.scale, self.radius] if held else [.16*self.scale])
        nearest = np.clip(points[:, None, :], self.obstacle_min, self.obstacle_max)
        return bool(np.any(np.linalg.norm(points[:, None, :]-nearest, axis=2) < radii[:, None]))

    def snapshot(self):
        return dict(mode='linked_2d', units='displayed image pixels; no metric XYZ',
            top_frame_size=self.top_size, side_frame_size=self.side_size,
            reach_top_px=self.paths[0].tolist() if self.top_size else None,
            carry_top_px=self.paths[1].tolist() if self.top_size else None,
            side_pickup_px=self.side_origin.tolist() if self.side_origin is not None else None,
            side_box_center_px=self.box_center.tolist() if self.box_center is not None else None,
            side_box_half_size_px=self.box_half.tolist() if self.box_center is not None else None,
            side_box_manually_placed=self.box_manual,
            box_opening='downward', side_obstacles='hidden', top_ball_scale=1.25,
            side_ball_opacity=.75, side_box_opacity=.35,
            obstacle_style='rounded sky-blue blocks; conservative rectangular collision footprint')


class LinkedSceneView:
    """Cache fixed image geometry; add only the sphere and a new trace segment each frame."""
    def __init__(self, scene, name):
        self.scene, self.name = scene, name
        self.cache_key = self.size = None
        self.reset_trace()

    def reset_trace(self):
        self.last = self.last_time = None
        self.trace_layer = self.trace_mask = None
        if self.size:
            self.trace_layer = np.zeros((*self.size[::-1], 3), np.uint8)
            self.trace_mask = np.zeros(self.size[::-1], np.uint8)
            if self.name == 'top':
                for path in self.scene.paths:
                    pts = np.rint(path).astype(np.int32)
                    dashed_path(self.trace_layer, pts, (214,184,120))
                    dashed_path(self.trace_mask, pts, 255)

    def _cache(self, frame):
        self.size = frame.shape[1::-1]
        self.paint = np.zeros_like(frame)
        self.mask = np.zeros(frame.shape[:2], np.uint8)
        shadow = np.zeros(frame.shape[:2], np.uint8)
        scene = self.scene
        if self.name == 'top':
            lo, hi = scene.obstacle_min, scene.obstacle_max
        else:
            lo = hi = []
        for a, b in zip(lo, hi):
            x0, y0 = np.rint(a).astype(int)
            x1, y1 = np.rint(b).astype(int)
            width, height = x1-x0+1, y1-y0+1
            radius = max(4., min(width, height)*.08)
            yy, xx = np.mgrid[:height, :width]
            qx = np.abs(xx-(width-1)/2)-((width-1)/2-radius)
            qy = np.abs(yy-(height-1)/2)-((height-1)/2-radius)
            distance = np.hypot(np.maximum(qx,0), np.maximum(qy,0)) + np.minimum(np.maximum(qx,qy),0)-radius
            body = distance <= 0
            # Matte blue, a fine highlight and a narrow lower edge, no picture-frame bevel.
            tone = np.broadcast_to(np.linspace(174.,144.,height)[:,None], (height,width)).copy()
            tone += np.exp(-np.abs(distance)/1.5)*(18.-42.*yy/max(1,height-1))
            colors = np.clip(tone[...,None]+[70,20,-65],0,255).astype(np.uint8)
            self.paint[y0:y1+1,x0:x1+1][body] = colors[body]
            self.mask[y0:y1+1,x0:x1+1][body] = 255
            shadow[y0+5:y1+6,x0+3:x1+4][body] = 255
        self.shadow_alpha = cv2.GaussianBlur(shadow, (13,13), 3).astype(np.float32)[...,None]*(.28/255)
        self.cache_key = (self.size, scene.revision)
        self.reset_trace()

    def draw(self, frame, ball):
        if (frame.shape[1::-1], self.scene.revision) != self.cache_key:
            self._cache(frame)
        if self.name == 'top':
            frame[:] = (frame*(1-self.shadow_alpha)).astype(np.uint8)
        mixed = cv2.addWeighted(frame, .25 if self.name == 'top' else .65, self.paint,
                                .75 if self.name == 'top' else .35, 0)
        cv2.copyTo(mixed, self.mask, frame)
        scene = self.scene
        if self.name == 'top':
            center, half, radius = scene.paths[1][-1], np.full(2, BOX_WIDTH*scene.scale/2), scene.radius
        elif scene.box_center is not None:
            center, half, radius = scene.box_center, scene.box_half, scene.side_radius
        else:
            return
        a, b = np.rint(center-half).astype(int), np.rint(center+half).astype(int)
        if self.name == 'top':
            cv2.rectangle(frame, tuple(a), tuple(b), (193,128,78), 5)
        else:
            # Side section: roof and two walls, deliberately no bottom bar.
            raw = frame.copy()
            cv2.polylines(frame, [np.array([[a[0],b[1]], a, [b[0],a[1]], b])], False, (193,128,78), 8, cv2.LINE_AA)
            cv2.addWeighted(frame, .35, raw, .65, 0, dst=frame)
        if ball is not None:
            if self.name == 'side':
                under_ball = frame.copy()
            draw_sphere(frame, ball, radius, pigment=BALL_COLOR)
            if self.name == 'side':
                cv2.addWeighted(frame, .75, under_ball, .25, 0, dst=frame)

    def trace(self, point, now):
        pass  # Actual trajectory remains in data only.

    def operator_frame(self, frame):
        result = frame.copy()
        if self.trace_layer is not None:
            cv2.copyTo(self.trace_layer, self.trace_mask, result)
        return result


class LinkedAbcTask(AbcTask):
    requires_flip_confirmation = False
    hints = {**AbcTask.hints, 'idle': 'BLE check: tap WASD / I / K. Place Side ball + box, then N at A',
        'flip_cw': 'J: CW / rotate down 90 deg below the box; then I to lift',
        'flip_ccw': 'L: CCW / rotate back 90 deg',
        'complete': 'CCW requested: finish rotating back, then E to end recording'}

    def reset(self):
        super().reset()
        self.valid = self.at_c = False
        self.observed_at = -1e9
        self.side_ball = self.last_side_ball = self.lower_palm_y = None
        self.flip_confirmations = []
        self.pending_result = None

    def start_ready(self, spatial):
        if self.scene.side_origin is None:
            raise ValueError('Click Side View to place the pickup ball first')
        if not self.scene.box_manual:
            raise ValueError('Place the Side box manually: Place box [O], then click its center')

    def start(self, spatial, now):
        super().start(spatial, now)
        self.experiment.active.update(task_tracking='linked_2d', flip_confirmation='none; J/L record operator intent',
            flip_target_degrees=90, flip_angle_measured=False,
            flip_confirmations=[], success_scope='Top/Side pixel task with operator-requested flips; no measured XYZ or rotation angle',
            collision_scope='Top image palm-center and sphere proxies only',
            completion_rule='Capture, lower empty hand, request CCW return, then E ends recording after return motion')
        self.experiment._save_trials()

    def request_torsion(self, settings, *, role=None):
        role = role or ('torsion_ccw' if self.phase == 'flip_ccw' else 'torsion_cw')
        accepted, message = super().request_torsion(settings, role=role)
        expected = {'torsion_cw': 'flip_cw', 'torsion_ccw': 'flip_ccw'}.get(role)
        if not self.experiment.active or self.phase != expected:
            return accepted, message
        now = time.monotonic()
        # Like G, operator intent is independent of BLE delivery; no angle is inferred.
        self.experiment.event('flip_requested_by_operator', mono=now, phase=self.phase,
            target_degrees=90, measured_angle=False, feedback_accepted=accepted)
        trial = self.experiment.active
        trial.setdefault('flip_requests', []).append(dict(phase=self.phase,
            t_s=self.experiment.clock.seconds(now), target_degrees=90,
            measured_angle=False, feedback_accepted=accepted))
        if self.phase == 'flip_cw':
            self.lower_palm_y = float(self.side_palm[1]) if self.valid else None
            self.last_side_ball = self.side_ball.copy() if self.valid else None
            self._change('lift', now)
        else:
            self.pending_result = ('not_started' if trial['start_t_s'] is None else
                'invalid_sequence' if trial['grasp_t_s'] is None or trial['grasp_t_s'] < trial['start_t_s'] else
                'assisted' if trial['manual_interventions'] else 'collision' if self.collision_seen else 'success')
            self._change('complete', now)
            trial['return_requested_t_s'] = self.experiment.clock.seconds(now)
            # Keep the trial and coordinate/video recording alive during the return.
            # E is the existing recording-end action, not an angle confirmation.
        self.experiment._save_trials()
        return True, f'90 deg rotation requested / {message}'

    def update_linked(self, top_obs, side_obs, prop, linked, held, now, *, manual=False, position_valid=True):
        self.valid = top_obs is not None and side_obs is not None and linked.ready and position_valid
        self.observed_at = now
        self.side_palm = side_obs.palm if side_obs is not None else None
        self.side_ball = np.array(linked.prop.pos_xy) if linked.ready else None
        self.at_c = self.valid and np.linalg.norm(top_obs.palm-self.scene.paths[1][-1]) < .75*self.scene.scale
        if not self.experiment.active or self.phase in ('idle', 'failed', 'complete'):
            return
        rising, falling = held and not self.was_held, self.was_held and not held
        self.was_held = held
        inside = self.valid and self.scene.inside_top(prop.pos_xy) and self.scene.inside_side(self.side_ball)
        if falling:
            trial = self.experiment.active
            trial['release_t_s'] = self.experiment.clock.seconds(now)
            trial['placement_error_px'] = float(np.linalg.norm(np.asarray(prop.pos_xy)-self.scene.paths[1][-1])) if self.valid else None
            if manual and not self.manual_grasp:
                trial['manual_interventions'] += 1
            accepted = inside and self.phase == 'release' and self.entered_from_below and (self.drop_cued or manual)
            self.experiment.event('release_detected', mono=now, accepted=bool(accepted),
                reason='manual' if manual else 'open_hand', coordinate_system='linked_2d')
            if accepted:
                self.captured = True
                trial['captured_ball_top_px'], trial['captured_ball_side_px'] = list(prop.pos_xy), self.side_ball.tolist()
                self._change('lower', now)
            else:
                self.result = ('unverified_position' if not self.valid else
                    'missing_drop_cue' if self.phase == 'release' and not self.drop_cued and not manual else 'invalid_release')
                self._change('failed', now)
            self.experiment._save_trials()
            return
        if rising and self.manual_grasp:
            self.waypoint = 1
            self._change('carry', now)
        if not self.valid:
            self.stable_since = self.last_side_ball = None
            return
        if self.lower_palm_y is None and self.phase == 'lift':
            self.lower_palm_y = float(self.side_palm[1])
        collision = self.scene.collision(top_obs.palm, prop.pos_xy, held)
        if collision != self.collision:
            self.experiment.event('obstacle_contact' if collision else 'obstacle_clear', mono=now, model='Top image proxies only')
        self.collision = collision
        self.collision_seen |= collision
        self.experiment.active['collision_seen'] = self.collision_seen
        if self.phase in ('lift', 'release'):
            previous, opening = self.last_side_ball, self.scene.opening_y
            if previous is not None and previous[1] > opening >= self.side_ball[1]:
                t = (opening-previous[1])/(self.side_ball[1]-previous[1])
                x = previous[0]+t*(self.side_ball[0]-previous[0])
                self.entered_from_below = bool(abs(x-self.scene.box_center[0]) <= self.scene.box_half[0]-self.scene.side_radius-5)
                self.experiment.event('box_opening_crossed', mono=now, from_below=self.entered_from_below, units='Side pixels')
            if (self.side_ball[1]-self.scene.side_radius > opening or
                abs(self.side_ball[0]-self.scene.box_center[0]) > self.scene.box_half[0]-self.scene.side_radius-5):
                self.entered_from_below = False
        self.last_side_ball = self.side_ball.copy()
        if self.phase in ('reach', 'carry'):
            path = self.scene.paths[self.phase == 'carry']
            if self._stable(np.linalg.norm(top_obs.palm-path[self.waypoint]) < .65*self.scene.scale, now):
                self.experiment.event('waypoint_reached', mono=now, phase=self.phase, index=self.waypoint)
                self.waypoint += 1
                self.stable_since = None
                if self.waypoint == len(path):
                    self.waypoint = 1
                    self._change('pick' if self.phase == 'reach' else 'flip_cw', now)
        elif self.phase == 'pick' and rising:
            self._change('carry', now)
        elif self.phase == 'lift' and self._stable(self.at_c and inside and self.entered_from_below, now):
            self._change('release', now)
        elif self.phase == 'lower' and self._stable(self.at_c and self.lower_palm_y is not None and self.side_palm[1] >= self.lower_palm_y-self.scene.side_radius*.35, now):
            self._change('flip_ccw', now)

    @property
    def status(self):
        return f'{self.phase.upper()}: {self.hints[self.phase]}'

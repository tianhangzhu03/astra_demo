"""English manual console with an explicit hardware-free preview entry point."""

import argparse
from dataclasses import replace
from functools import lru_cache
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sphere_renderer import draw_sphere
from .manual_control import Actuation, Cue, ManualBleController, REPEATABLE_CUES, REPEAT_MAX_PULSE_S
from .grasp_feedback import GraspFeedback


# OpenCV colors are BGR. Selected from the user's scientific-color reference;
# reserve color for paths and current state, and always retain explicit words.
INK = (74, 45, 11)                 # #0B2D4A
MUTED = (124, 109, 94)
ACCENT = (141, 92, 20)             # #145C8D
BLUE_TINT = (230, 201, 175)        # #AFC9E6
LAVENDER = (217, 162, 199)         # #C7A2D9
PLUM = (134, 90, 139)             # #8B5A86
CREAM = (161, 231, 248)            # #F8E7A1
AMBER_INK = (34, 93, 126)
REFERENCE_COLOR = ACCENT
TRAJECTORY_COLOR = PLUM
LINE = (229, 222, 214)
PAPER = (255, 255, 255)
SURFACE = (250, 248, 246)
ACCENT_FILL = (246, 236, 224)
HELD_COLOR = PLUM
STOP_COLOR = (65, 71, 153)


def enable_dpi_awareness():
    if os.name == 'nt':
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def open_video_window(name, width=960, height=540):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, width, height)


def fit_window_frame(frame, width, height, background=PAPER):
    """Win32 HighGUI stretches WINDOW_NORMAL; explicitly preserve the image aspect ratio."""
    fh, fw = frame.shape[:2]
    scale = min(width / fw, height / fh)
    rw, rh = max(1, round(fw * scale)), max(1, round(fh * scale))
    dx, dy = (width - rw) // 2, (height - rh) // 2
    canvas = np.full((height, width, 3), background, np.uint8)
    canvas[dy:dy+rh, dx:dx+rw] = cv2.resize(frame, (rw, rh))
    return canvas, (dx, dy, rw / fw, rh / fh)


def show_video_frame(name, frame, background=PAPER):
    try:
        _, _, width, height = cv2.getWindowImageRect(name)
    except cv2.error:
        width, height = frame.shape[1], frame.shape[0]
    if width <= 0 or height <= 0:
        width, height = frame.shape[1], frame.shape[0]
    canvas, transform = fit_window_frame(frame, width, height, background)
    cv2.imshow(name, canvas)
    return transform


def draw_view_status(frame, label, held=False, *, left=8, right=None):
    # Secondary state shares the header row; omit it if a narrow frame cannot fit it.
    color = HELD_COLOR if held else INK
    right = frame.shape[1] - 8 if right is None else right
    pixels = 12
    width = _text_mask(label, pixels)[0].shape[1] + 22
    if left + width > right:
        return
    fill = (240, 221, 233) if held else (250, 246, 242)
    rounded_rect(frame, (left, 8, width, 30), fill, PLUM if held else LINE, radius=10)
    text(frame, label, (left + 11, 27), pixels / 34, color)


def haptic_display_lines(snapshot, clock=None):
    """Detailed diagnostic wording; camera overlays use only the compact title."""
    trigger = snapshot.trigger
    if snapshot.state == 'active' and trigger is not None:
        title = 'HAPTIC: ' + trigger.label
        mode = 'SIMULATED' if snapshot.simulated else 'COMMAND SENT'
        detail = f'{mode} | SET {trigger.volts_mv} mV / {trigger.duration_s:.2f} s'
        if clock is not None:
            detail += f' | trigger t={clock.seconds(trigger.sent_at):.3f}s'
        return title, detail
    title = 'HAPTIC: ' + {'idle': 'IDLE', 'pending': 'PENDING',
                         'stopping': 'STOPPING', 'disconnected': 'DISCONNECTED'}[snapshot.state]
    if snapshot.state == 'disconnected':
        return title, 'Output unconfirmed / no BLE connection'
    if snapshot.state == 'pending':
        return title, 'Waiting for BLE write'
    if snapshot.state == 'stopping':
        return title, 'Waiting for output stop to finish'
    detail = 'SIMULATED / no output' if snapshot.simulated else 'No active command'
    if snapshot.last_trigger is not None:
        detail = ('SIMULATED | ' if snapshot.simulated else '') + 'LAST TRIGGER: ' + snapshot.last_trigger.label
        if clock is not None:
            detail += f' | t={clock.seconds(snapshot.last_trigger.sent_at):.3f}s'
    return title, detail


def haptic_colors(snapshot):
    """Fill, border, title and detail colors; state colors never imply a sent cue."""
    if snapshot.state == 'active' and snapshot.trigger is not None:
        cue = snapshot.trigger.label
        if cue == 'NORMAL FORCE':
            return CREAM, AMBER_INK, INK, INK
        if cue == 'GRASP CONFIRMATION':
            return PLUM, PLUM, PAPER, PAPER
        if cue in ('GRASP CUE', 'VIBRATION'):
            return LAVENDER, PLUM, INK, INK
        return BLUE_TINT, ACCENT, INK, INK
    if snapshot.state in ('pending', 'stopping', 'disconnected'):
        return (226, 246, 253), AMBER_INK, INK, MUTED
    return SURFACE, LINE, MUTED, MUTED


def draw_haptic_status(frame, snapshot, clock=None):
    # Draw on the source Display frame: live windows, console and both recordings agree.
    # Mode/settings/timing remain in the console and logs, outside the camera plane.
    title, _ = haptic_display_lines(snapshot)
    fill, border, title_color, _ = haptic_colors(snapshot)
    left, available = 8, min(frame.shape[1] - 40, 280)
    title, title_px = fitted_label(title, available, 14, bold=True)
    width = _text_mask(title, title_px, True)[0].shape[1] + 24
    # Pulses return to neutral IDLE immediately; history never looks like active output.
    rounded_rect(frame, (left, 8, width, 30), fill, border, radius=10)
    text(frame, title, (left + 12, 28), title_px / 34, title_color, bold=True)
    return left + width


@lru_cache(maxsize=96)
def _rounded_masks(width, height, radius):
    scale = 3
    outer = Image.new('L', (width * scale, height * scale))
    inner = outer.copy()
    ImageDraw.Draw(outer).rounded_rectangle((0, 0, width*scale-1, height*scale-1),
                                           radius=radius*scale, fill=255)
    ImageDraw.Draw(inner).rounded_rectangle((scale, scale, (width-1)*scale-1, (height-1)*scale-1),
                                           radius=max(0, radius-1)*scale, fill=255)
    masks = [np.asarray(im.resize((width, height), Image.Resampling.LANCZOS), np.float32)[..., None] / 255
             for im in (outer, inner)]
    return masks[0], np.maximum(0, masks[0] - masks[1])


def rounded_rect(canvas, rect, fill, border=None, *, radius=8):
    """Cached antialiased camera labels; leave the scene visible outside each corner."""
    x, y, width, height = map(int, rect)
    if width < 3 or height < 3:
        return
    outer, edge = _rounded_masks(width, height, min(radius, width//2, height//2))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x+width), min(canvas.shape[0], y+height)
    if x1 <= x0 or y1 <= y0:
        return
    roi = canvas[y0:y1, x0:x1]
    for mask, color in ((outer, fill), (edge, border)):
        if color is not None:
            alpha = mask[y0-y:y1-y, x0-x:x1-x]
            roi[:] = (roi*(1-alpha) + np.asarray(color, np.float32)*alpha).astype(np.uint8)


def route_legend_width(frame_width):
    return 208 if frame_width >= 600 else 160


@lru_cache(maxsize=48)
def _font(pixels, bold=False):
    # Arial / Helvetica-style sans serif, as used for scientific figure lettering.
    windows_name = 'arialbd.ttf' if bold else 'arial.ttf'
    mac_name = 'Arial Bold.ttf' if bold else 'Arial.ttf'
    linux_style = 'Bold' if bold else 'Regular'
    candidates = [
        str(Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / windows_name),
        '/System/Library/Fonts/Supplemental/' + mac_name,
        '/Library/Fonts/' + mac_name,
        f'/usr/share/fonts/truetype/liberation2/LiberationSans-{linux_style}.ttf',
        f'/usr/share/fonts/truetype/liberation/LiberationSans-{linux_style}.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans' + ('-Bold' if bold else '') + '.ttf',
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, pixels)
        except OSError:
            continue
    return ImageFont.load_default()


@lru_cache(maxsize=256)
def _text_mask(label, pixels, bold=False):
    font = _font(pixels, bold)
    left, top, right, bottom = font.getbbox(label, anchor='ls')
    image = Image.new('L', (max(1, right - left), max(1, bottom - top)))
    ImageDraw.Draw(image).text((-left, -top), label, font=font, fill=255, anchor='ls')
    return np.asarray(image, dtype=np.float32)[..., None] / 255, left, top


def fitted_label(label, width, pixels, bold=False):
    label = str(label)
    while pixels > 11 and _text_mask(label, pixels, bold)[0].shape[1] > width:
        pixels -= 1
    if _text_mask(label, pixels, bold)[0].shape[1] > width:
        while label and _text_mask(label + '...', pixels, bold)[0].shape[1] > width:
            label = label[:-1]
        label += '...'
    return label, pixels


def text(canvas, label, xy, size=0.45, color=INK, *, bold=False):
    alpha, dx, dy = _text_mask(str(label), max(10, round(size * 34)), bold)
    x, y = int(xy[0]) + dx, int(xy[1]) + dy
    h, w = alpha.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    a = alpha[y0-y:y1-y, x0-x:x1-x]
    roi = canvas[y0:y1, x0:x1]
    roi[:] = (roi * (1-a) + np.asarray(color, np.float32) * a).astype(np.uint8)


def fit_camera(canvas, frame, rect, empty_label):
    x, y, w, h = rect
    canvas[y:y+h, x:x+w] = SURFACE
    cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), LINE, 1)
    if frame is None:
        text(canvas, empty_label, (x + 22, y + h // 2), 0.45, MUTED)
        return
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    rw, rh = max(1, round(fw * scale)), max(1, round(fh * scale))
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    ox, oy = x + (w - rw) // 2, y + (h - rh) // 2
    canvas[oy:oy+rh, ox:ox+rw] = resized
    return (ox, oy, rw/fw, rh/fh)


def workspace_bounds(w, h):
    # Use the wide phone image rather than constraining every camera to a 3:2 box.
    # Portrait inputs retain a landscape workspace; leave room for the status text.
    width = w * .95
    height = min(h * .90, width / 1.25)
    top = max(min(48, round(h * .15)), round((h - height) / 2))
    bottom = min(h - 12, round(top + height))
    return (round((w - width) / 2), round((w + width) / 2), top, bottom)


def initial_ball_position(w, h):
    x0, x1, y0, y1 = workspace_bounds(w, h)
    return (x0 + (x1 - x0) * .25, y0 + (y1 - y0) * .25)


def draw_workspace_boundary(frame):
    h, w = frame.shape[:2]
    x0, x1, y0, y1 = workspace_bounds(w, h)
    cv2.rectangle(frame, (x0, y0), (x1, y1), PAPER, 3, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (135, 119, 103), 1, cv2.LINE_AA)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    # Reference guides only: these divisions have no hit-test or control semantics.
    dash, gap = max(5, round(w / 64)), max(4, round(w / 80))
    for x in range(x0, x1, dash + gap):
        cv2.line(frame, (x, cy), (min(x + dash, x1), cy), PAPER, 3, cv2.LINE_AA)
        cv2.line(frame, (x, cy), (min(x + dash, x1), cy), (155, 139, 123), 1, cv2.LINE_AA)
    for y in range(y0, y1, dash + gap):
        cv2.line(frame, (cx, y), (cx, min(y + dash, y1)), PAPER, 3, cv2.LINE_AA)
        cv2.line(frame, (cx, y), (cx, min(y + dash, y1)), (155, 139, 123), 1, cv2.LINE_AA)


@lru_cache(maxsize=12)
def _open_hand_sprite(height):
    """Antialiased five-finger outline with a short, opposed thumb; code-native artwork."""
    commands = [
        ('M', -19, 42), ('L', -20, 24), ('C', -25, 20, -28, 15, -32, 11),
        ('L', -46, -6), ('C', -52, -14, -43, -21, -37, -14), ('L', -23, 0),
        ('L', -28, -47), ('C', -29, -58, -15, -60, -14, -48), ('L', -10, -17),
        ('L', -12, -60), ('C', -12, -72, 3, -73, 4, -60), ('L', 5, -18),
        ('L', 8, -53), ('C', 9, -64, 23, -62, 23, -51), ('L', 21, -14),
        ('L', 27, -37), ('C', 30, -47, 43, -43, 40, -32), ('L', 32, 8),
        ('C', 30, 21, 23, 25, 19, 30), ('L', 17, 42), ('L', -19, 42),
    ]
    points, current = [], None
    for command in commands:
        coords = np.asarray(command[1:], float).reshape(-1, 2)
        if command[0] == 'C':
            p1, p2, end = coords
            for t in np.linspace(0, 1, 18)[1:]:
                points.append((1-t)**3 * current + 3*(1-t)**2*t*p1 + 3*(1-t)*t*t*p2 + t**3*end)
            current = end
        else:
            current = coords[0]
            points.append(current)
    scale, supersample = height / 114, 3
    size = (round(120 * scale), round(140 * scale))
    image = Image.new('RGBA', (size[0]*supersample, size[1]*supersample))
    path = [tuple(p) for p in ((np.array(points) + [60, 80]) * scale * supersample)]
    drawing = ImageDraw.Draw(image)
    drawing.polygon(path, fill=(255, 255, 255, 205))
    drawing.line(path, fill=(255, 255, 255, 255), width=max(3, round(5*scale*supersample)), joint='curve')
    drawing.line(path, fill=(*INK[::-1], 255), width=max(2, round(2.2*scale*supersample)), joint='curve')
    rgba = np.asarray(image.resize(size, Image.Resampling.LANCZOS))
    # The marker's anchor is the midpoint between thumb and index tips, matching
    # Experiment.begin's existing start criterion rather than implying palm centering.
    return rgba, np.array([26, 48]) * scale


def draw_start_hand(frame, point, height=96):
    rgba, anchor = _open_hand_sprite(height)
    x, y = np.round(np.asarray(point) - anchor).astype(int)
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(frame.shape[1], x+w), min(frame.shape[0], y+h)
    if x0 < x1 and y0 < y1:
        patch = rgba[y0-y:y1-y, x0-x:x1-x]
        alpha = patch[..., 3:4] / 255.
        frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1-alpha) + patch[..., 2::-1] * alpha
    return (x, y, w, h)


class OperatorPanel:
    window_name = 'Haptic Guidance | Operator'

    def __init__(self, controller=None, settings=None, grab_confirmation=True):
        self.controller = controller
        self.settings = settings or Actuation()
        self.settings.validate()
        vision_only = controller is not None and not getattr(controller, 'enabled', True)
        self.message = ('Vision only / hardware output disabled' if vision_only else
                        'Set voltage, then select a cue' if controller else 'Offline preview only')
        self._held_keys = set()
        self._key_down = None
        self._app_focused = lambda: True
        if os.name == 'nt':
            import ctypes
            self._key_down = lambda vk: bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
            def focused():
                process = ctypes.c_ulong()
                ctypes.windll.user32.GetWindowThreadProcessId(ctypes.windll.user32.GetForegroundWindow(), ctypes.byref(process))
                return process.value == os.getpid()
            self._app_focused = focused
        self._input_tick = time.monotonic()
        self._held_input_vk = None
        self.last_cue = None
        self.cue_time = 0.0
        self.quit_requested = False
        self._display_transform = (0, 0, 1, 1)
        self.manual_request = None
        self.record_requests = []
        self.ui_record_request = False
        self.ui_record_active = False
        self.ui_record_rect = (824, 37, 184, 32)
        self.trial_request = None
        self.restart_pending = False
        self.trial_status = 'N: new trial + video   0: restart   E: end trial'
        self.linked_views = False
        self.side_place_request = None
        self.side_box_request = None
        self.side_placement_mode = 'ball'
        self.side_ball_rect, self.side_box_rect = (944,116,92,30), (1044,116,96,30)
        self._side_camera_transform = None
        self._side_camera_size = None
        self.record_mode = 'Display'
        self.record_active = {'top': False, 'side': False}
        self.record_status = {'top': 'Ready', 'side': 'Ready'}
        self.held = False
        self.grasp_status = 'WAIT CUE [G]'
        self.tracking_status = 'No hand'
        self.xyz_status = ''
        self.advanced_recording = False
        self.dashboard = None
        self.grasp_feedback = GraspFeedback(controller, grab_confirmation)
        self.task_controls = None
        self.record_buttons = {'top': (24, 744, 245, 42), 'side': (281, 744, 245, 42),
                               'both': (538, 744, 254, 42)}
        self.cancel_rect = (24, 798, 132, 35)
        self.release_rect = (164, 798, 112, 35)
        self.record_mode_rect = (284, 798, 172, 35)
        self.trial_rect = (464, 798, 86, 35)
        self.end_trial_rect = (558, 798, 82, 35)
        self.restart_rect = (648, 798, 144, 35)
        self.buttons = {
            Cue.FORWARD: (940, 470, 96, 58),
            Cue.LEFT: (836, 534, 96, 58),
            Cue.NORMAL: (940, 534, 96, 58),
            Cue.RIGHT: (1044, 534, 96, 58),
            Cue.BACK: (940, 598, 96, 58),
            Cue.GRASP: (836, 670, 304, 60),
        }
        self._legacy_buttons = self.buttons.copy()
        self.stop_rect = (836, 742, 304, 50)
        self.exit_rect = (992, 804, 148, 36)
        self.adjust_buttons = []
        for field, y in [('volts_mv', 337), ('duration_s', 377), ('frequency_hz', 417)]:
            self.adjust_buttons.extend([(field, -1, (1056, y, 36, 32)),
                                        (field, 1, (1104, y, 36, 32))])
        self._base = np.full((900, 1180, 3), PAPER, dtype=np.uint8)
        text(self._base, 'Haptic guidance', (24, 56), .88, bold=True)
        cv2.rectangle(self._base, (1024, 37), (1156, 67), SURFACE, -1)
        cv2.rectangle(self._base, (1024, 37), (1156, 67), LINE, 1)
        mode_label = 'VISION ONLY' if vision_only else 'MANUAL BLE' if controller else 'LOCAL PREVIEW'
        text(self._base, mode_label, (1038, 59), 0.35, MUTED)
        cv2.line(self._base, (24, 105), (1156, 105), LINE, 1)
        text(self._base, 'a', (24, 140), .55, bold=True)
        text(self._base, 'Top view', (48, 140), .47, bold=True)
        text(self._base, 'b', (836, 140), .55, bold=True)
        text(self._base, 'Side view', (860, 140), .47, bold=True)
        text(self._base, 'WASD: direction   Space: normal   G: grasp   0: restart   Esc/X: stop   Q: exit', (24, 850), .32, MUTED)
        cv2.line(self._base, (816, 122), (816, 840), LINE, 1)
        cv2.line(self._base, (24, 855), (1156, 855), LINE, 1)

    def configure_task(self, task):
        self.task_controls = task
        self.buttons = ({Cue.FORWARD:(940,470,96,38), Cue.LEFT:(836,514,96,38),
            Cue.RIGHT:(1044,514,96,38), Cue.BACK:(940,558,96,38),
            'normal_up':(836,604,148,42), 'normal_down':(992,604,148,42),
            'torsion_cw':(836,652,148,42), 'torsion_ccw':(992,652,148,42),
            Cue.GRASP:(836,702,304,28)} if task else self._legacy_buttons.copy())
        if task:
            from .operator_dashboard import OperatorDashboard
            self.dashboard = OperatorDashboard(self)

    def update_measurements(self, trial, marker, now):
        if self.dashboard:
            self.dashboard.update(trial, marker, now)

    def queue_side_click(self, point, *, box=False):
        if self.task_controls is not None and (box or self.side_placement_mode == 'box'):
            self.side_box_request = point
        else:
            self.side_place_request = point

    def place_side_mode(self, mode):
        self.side_placement_mode = mode
        self.message = f'Click Side View to place the {"box center" if mode == "box" else "pickup ball"}'

    def hold(self, action, vk):
        if not self.task_controls:
            self.select(action)
            return
        if self.restart_pending or self.grasp_feedback.output_pending:
            self.message = 'Wait for restart / grasp output, or press Stop'
            return
        try:
            cue, role = self.task_controls.force_command(action)
        except ValueError as exc:
            self.message = str(exc)
            return
        if cue not in REPEATABLE_CUES:
            self.select(action)
            return
        guard = lambda: (self._app_focused() and not self.quit_requested
            and not self.restart_pending
            and time.monotonic()-self._input_tick < .75)
        accepted, self.message = self.controller.start_repeat(cue, self.settings, role=role, guard=guard)
        if accepted:
            self._held_input_vk = vk
            self.last_cue, self.cue_time = role if role in ('normal_up', 'normal_down') else cue, time.perf_counter()

    def open(self):
        enable_dpi_awareness()
        open_video_window(self.window_name, 1180, 900)
        cv2.setMouseCallback(self.window_name, self._on_window_mouse)

    def show(self, *args, frame=None, **kwargs):
        self._display_transform = show_video_frame(self.window_name,
            self.render(*args, **kwargs) if frame is None else frame)

    def _on_window_mouse(self, event, x, y, flags, param):
        dx, dy, sx, sy = self._display_transform
        self.on_mouse(event, (x - dx) / sx, (y - dy) / sy, flags, param)

    def set_grab_state(self, held, tracking_status):
        self.held, self.tracking_status = held, tracking_status
        self.grasp_feedback.update(held)

    def release(self):
        self.grasp_feedback.cancel()
        self.manual_request = 'release'

    def cancel_grasp(self):
        self.grasp_feedback.cancel()
        self.manual_request = 'cancel'

    def toggle_record_mode(self):
        if any(self.record_active.values()):
            self.message = 'Stop recordings before changing their content'
        else:
            self.record_mode = 'Raw' if self.record_mode == 'Display' else 'Display'
            self.message = f'Recording content: {self.record_mode}'

    def _first_press(self, vk):
        if self._key_down:
            if vk in self._held_keys:
                return False
            if self._key_down(vk):
                self._held_keys.add(vk)
        return True

    def select(self, cue):
        if self.restart_pending:
            self.message = 'Waiting for restart readiness; Esc/X cancels'
            return
        if self.task_controls is not None:
            self.controller.stop_repeat('Single cue selected')
        if self.task_controls is not None:
            handled = self.task_controls.select(cue, self)
            if handled is not None:
                accepted, self.message = handled
                if accepted:
                    self.last_cue, self.cue_time = cue, time.perf_counter()
                return
        accepted = True
        if cue == Cue.GRASP:
            accepted, self.message = self.grasp_feedback.request_prompt(self.settings)
        elif self.grasp_feedback.output_pending:
            accepted, self.message = False, 'Wait for grasp confirmation, or press Stop to cancel it'
        elif self.controller and self.controller.enabled:
            accepted, self.message = self.controller.submit(cue, self.settings)
        elif self.controller and hasattr(self.controller, 'simulate'):
            accepted, self.message = self.controller.simulate(cue, self.settings)
        else:
            self.message = f'Preview: {cue.value} (no output)'
        if not accepted:
            return
        self.last_cue = cue
        self.cue_time = time.perf_counter()

    def stop_output(self):
        if self.task_controls is not None:
            self.task_controls.cancel_output()
        if self.restart_pending:
            self.trial_status = 'Restart cancelled. N: new trial. 0: restart.'
        self.restart_pending = False
        if self.trial_request == 'restart':
            self.trial_request = None
        self.last_cue = None
        self.grasp_feedback.cancel()
        self.message = self.controller.stop_output() if self.controller else 'Preview: output stopped'

    def restart(self):
        self.stop_output()
        self.manual_request = self.side_place_request = self.side_box_request = None
        self.record_requests.clear()
        self.restart_pending = True
        self.trial_request = 'restart'
        self.message = 'Resetting the scene and starting a new trial'

    def adjust(self, field, direction):
        step, lo, hi = {'volts_mv': (500 if self.task_controls else 25, 0, 5000), 'duration_s': (.1, .1, 5.),
                        'frequency_hz': (10, 10, 200)}[field]
        if self.task_controls:
            self.controller.stop_repeat('Settings changed; press again')
            if field == 'duration_s':
                hi = REPEAT_MAX_PULSE_S
        value = min(hi, max(lo, round(getattr(self.settings, field) + direction * step, 2)))
        self.settings = replace(self.settings, **{field: value})

    def quit(self):
        self.stop_output()
        self.quit_requested = True

    def handle_key(self, key):
        self._input_tick = time.monotonic()
        if self.task_controls and not self._app_focused():
            self.controller.stop_repeat('Application focus lost')
            return
        if self.task_controls and not self.controller.enabled:
            self.controller.poll_repeat()
        if self._key_down:
            self._held_keys = {vk for vk in self._held_keys if self._key_down(vk)}
        keys = {
            ord('w'): Cue.FORWARD, ord('s'): Cue.BACK,
            ord('a'): Cue.LEFT, ord('d'): Cue.RIGHT,
            ord(' '): Cue.NORMAL, ord('g'): Cue.GRASP,
            2490368: Cue.FORWARD, 2621440: Cue.BACK, 2424832: Cue.LEFT, 2555904: Cue.RIGHT,
            65362: Cue.FORWARD, 65364: Cue.BACK, 65361: Cue.LEFT, 65363: Cue.RIGHT,
            63232: Cue.FORWARD, 63233: Cue.BACK, 63234: Cue.LEFT, 63235: Cue.RIGHT,
        }
        if ord('A') <= key <= ord('Z'):
            key += 32
        if self.task_controls is not None and key in map(ord, 'ijklop'):
            vk = ord(chr(key).upper())
            if not self._first_press(vk):
                return
            if key in (ord('o'), ord('p')):
                self.place_side_mode('box' if key == ord('o') else 'ball')
            elif key in (ord('j'), ord('l')):
                self.select('torsion_cw' if key == ord('j') else 'torsion_ccw')
            else:
                self.hold('normal_up' if key == ord('i') else 'normal_down', vk)
            return
        if key == ord('f') and hasattr(self.task_controls, 'confirm_flip') and getattr(self.task_controls, 'requires_flip_confirmation', True):
            if self._first_press(ord('F')):
                self.message = self.task_controls.confirm_flip()
            return
        if key == ord('u'):
            if self._first_press(ord('U')):
                self.ui_record_request = True
            return
        if key == ord('0'):
            vk = 0x60 if self._key_down and self._key_down(0x60) else ord('0')
            if self._first_press(vk):
                self.restart()
            return
        if key in (ord('n'), ord('e')):
            if self._first_press(ord(chr(key).upper())):
                if self.task_controls:
                    self.controller.stop_repeat('Trial control')
                self.trial_request = 'start' if key == ord('n') else 'end'
            return
        if self.task_controls and key in map(ord, 'cr'):
            return  # ABC uses G to toggle grasp and X to stop output.
        if key in map(ord, 'crtybv'):
            if not self._first_press(ord(chr(key).upper())):
                return
            if key == ord('c'):
                self.cancel_grasp()
            elif key == ord('r'):
                self.release()
            elif key == ord('v'):
                self.toggle_record_mode()
            else:
                self.record_requests.append({ord('t'): 'top', ord('y'): 'side', ord('b'): 'both'}[key])
            return
        if key in keys:
            # HighGUI repeats key-down events; on Windows require a real release.
            vk = {Cue.FORWARD: 0x26, Cue.BACK: 0x28, Cue.LEFT: 0x25, Cue.RIGHT: 0x27}.get(keys[key]) if key > 255 else ord(chr(key).upper())
            if self._key_down and vk is not None:
                if vk in self._held_keys:
                    return
                if self._key_down(vk):
                    self._held_keys.add(vk)
            if self.task_controls and keys[key] in REPEATABLE_CUES:
                self.hold(keys[key], vk)
            else:
                self.select(keys[key])
        elif key in (27, ord('x')):
            self.stop_output()
        elif key == ord('q'):
            self.quit()
        else:
            adjustment = {ord('1'): ('volts_mv', -1), ord('2'): ('volts_mv', 1),
                          ord('3'): ('duration_s', -1), ord('4'): ('duration_s', 1),
                          ord('5'): ('frequency_hz', -1), ord('6'): ('frequency_hz', 1)}.get(key)
            if adjustment:
                self.adjust(*adjustment)

    @staticmethod
    def _inside(x, y, rect):
        rx, ry, w, h = rect
        return rx <= x < rx+w and ry <= y < ry+h

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN and self.linked_views and hasattr(self.task_controls, 'confirm_flip'):
            from .linked_views import camera_point
            point = camera_point(x, y, self._side_camera_transform, self._side_camera_size)
            if point is not None:
                self.queue_side_click(point, box=True)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.dashboard and self._inside(x, y, self.dashboard.advanced_rect):
            self.advanced_recording = not self.advanced_recording
            return
        recording_visible = not self.dashboard or self.advanced_recording
        if recording_visible and self._inside(x, y, self.ui_record_rect):
            self.ui_record_request = True
            return
        if self.linked_views and self.task_controls:
            if self._inside(x, y, self.side_ball_rect):
                self.place_side_mode('ball')
                return
            if self._inside(x, y, self.side_box_rect):
                self.place_side_mode('box')
                return
        if self.linked_views:
            from .linked_views import camera_point
            point = camera_point(x, y, self._side_camera_transform, self._side_camera_size)
            if point is not None:
                self.queue_side_click(point)
                return
        for view, rect in self.record_buttons.items():
            if recording_visible and self._inside(x, y, rect):
                self.record_requests.append(view)
                return
        if not self.task_controls and self._inside(x, y, self.cancel_rect):
            self.cancel_grasp()
            return
        if not self.task_controls and self._inside(x, y, self.release_rect):
            self.release()
            return
        if recording_visible and self._inside(x, y, self.record_mode_rect):
            self.toggle_record_mode()
            return
        if self._inside(x, y, self.trial_rect):
            self.trial_request = 'start'
            return
        if self._inside(x, y, self.end_trial_rect):
            self.trial_request = 'end'
            return
        if self._inside(x, y, self.restart_rect):
            self.restart()
            return
        for cue, rect in self.buttons.items():
            if self._inside(x, y, rect):
                if self.task_controls and (cue in REPEATABLE_CUES or cue in ('normal_up', 'normal_down')):
                    self.hold(cue, 0x01)
                else:
                    self.select(cue)
                return
        for field, direction, rect in self.adjust_buttons:
            if self._inside(x, y, rect):
                self.adjust(field, direction)
                return
        if self._inside(x, y, self.stop_rect):
            self.stop_output()
        elif self._inside(x, y, self.exit_rect):
            self.quit()

    def _button(self, canvas, cue, rect, active=False):
        x, y, w, h = rect
        primary = cue == Cue.GRASP
        fill = ACCENT_FILL if active else PAPER
        cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), fill, -1)
        cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), ACCENT if active else LINE, 1)
        if self.task_controls is not None:
            labels = {Cue.FORWARD:'W  Forward', Cue.BACK:'S  Back', Cue.LEFT:'A  Left', Cue.RIGHT:'D  Right',
                'normal_up':'I  Normal Up', 'normal_down':'K  Normal Down',
                'torsion_cw':'J  CW / down 90 deg', 'torsion_ccw':'L  CCW / back 90 deg', Cue.GRASP:'G  Pick / Drop'}
            text(canvas, labels[cue], (x+8, y+17), .34, bold=True)
            if h > 30:
                torsion = cue in ('torsion_cw','torsion_ccw')
                subtitle = ('Verify firmware' if self.controller.enabled and not self.controller.torsion_firmware_ready else 'Single cue') if torsion else 'Single vibration' if cue == Cue.GRASP else 'Tap: 1 pulse / s'
                text(canvas, subtitle, (x+8, y+32), .27, MUTED)
            return
        if primary:
            label = 'Pick / Drop cue' if self.task_controls is not None else 'Grasp cue'
            text(canvas, label, (x+16, y+25), .47, bold=True)
            text(canvas, 'Single vibration pulse', (x+16, y+45), .35, MUTED)
            self._keycap(canvas, 'G', (x+w-42, y+18, 26, 24))
            return
        keys = {Cue.FORWARD:'W', Cue.BACK:'S', Cue.LEFT:'A', Cue.RIGHT:'D', Cue.NORMAL:'Space'}
        mask, _, _ = _text_mask(cue.value, 13, True)
        text(canvas, cue.value, (x+(w-mask.shape[1])//2, y+23), 13/34, bold=True)
        key = keys[cue]
        key_width = 48 if key == 'Space' else 26
        self._keycap(canvas, key, (x+(w-key_width)//2, y+31, key_width, 19))

    @staticmethod
    def _keycap(canvas, label, rect):
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), SURFACE, -1)
        cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), LINE, 1)
        mask, _, _ = _text_mask(label, 11)
        text(canvas, label, (x+(w-mask.shape[1])//2, y+(h+8)//2), 11/34, MUTED)

    def render(self, top=None, side=None, *, fps=0.0,
               ble_status='Disabled', simulated=False):
        if self.dashboard:
            return self.dashboard.render(top, side, fps=fps)
        canvas = self._base.copy()
        if self.task_controls is not None:
            cv2.rectangle(canvas, (24, 835), (810, 854), PAPER, -1)
            confirm_hint = '   F: confirm' if getattr(self.task_controls, 'requires_flip_confirmation', True) else ''
            text(canvas, 'WASD / I,K: 1 Hz   G: pick/drop   J/L: CW/CCW 90 deg' + confirm_hint + '   X: stop', (24, 850), .30, MUTED)
        self._plain_button(canvas, self.ui_record_rect,
                           'Stop UI [U]' if self.ui_record_active else 'Record UI [U]', self.ui_record_active)
        text(canvas, self.trial_status[:110], (25, 85), .40, MUTED)
        fit_camera(canvas, top, (24, 154, 768, 576), 'Waiting for top camera')
        if self.xyz_status:
            cv2.rectangle(canvas, (32, 690), (784, 724), PAPER, -1)
            text(canvas, self.xyz_status[:112], (42, 713), .40, INK)
        self._side_camera_transform = fit_camera(canvas, side, (836, 154, 304, 171), 'Side camera not connected')
        self._side_camera_size = side.shape[1::-1] if side is not None else None
        if self.linked_views:
            if self.task_controls:
                self._plain_button(canvas, self.side_ball_rect, 'Ball [P]', self.side_placement_mode == 'ball')
                self._plain_button(canvas, self.side_box_rect, 'Box [O]', self.side_placement_mode == 'box')
            else:
                text(canvas, 'Linked', (1085, 140), .34, MUTED)
        text(canvas, 'Object held' if self.held else 'Object released', (220, 140), .39,
             HELD_COLOR if self.held else MUTED)
        text(canvas, self.grasp_status, (400, 140), .35, MUTED)
        for view, rect in self.record_buttons.items():
            active = (all(self.record_active.values()) if view == 'both' else self.record_active[view])
            label = {'top': 'top [T]', 'side': 'side [Y]', 'both': 'both [B]'}[view]
            self._plain_button(canvas, rect, ('Stop ' if active else 'Record ') + label, active)
        if not self.task_controls:
            self._plain_button(canvas, self.cancel_rect, 'Cancel cue [C]')
            self._plain_button(canvas, self.release_rect, 'Release [R]')
        self._plain_button(canvas, self.record_mode_rect, f'Video: {self.record_mode} [V]')
        self._plain_button(canvas, self.trial_rect, 'Trial [N]')
        self._plain_button(canvas, self.end_trial_rect, 'End [E]')
        self._plain_button(canvas, self.restart_rect, 'Restart [0]', self.restart_pending)
        text(canvas, self.record_status['top'][:35], (24, 797), .27, MUTED)
        text(canvas, self.record_status['side'][:35], (281, 797), .27, MUTED)
        age = time.perf_counter() - self.cue_time
        repeating = self.task_controls is not None and self.controller.repeat_action is not None
        for cue, rect in self.buttons.items():
            self._button(canvas, cue, rect, cue == self.last_cue and (age < .8 or repeating))
        for label, value, hint, y in [
            ('Voltage', f'{self.settings.volts_mv} mV', '1- / 2+', 337),
            ('Duration', f'{self.settings.duration_s:.1f} s', '3- / 4+', 377),
            ('Vibration', f'{self.settings.frequency_hz} Hz', '5- / 6+', 417),
        ]:
            text(canvas, label, (836, y+15), .39)
            text(canvas, hint, (836, y+30), .28, MUTED)
            text(canvas, value, (946, y+23), .46, bold=True)
        for _, direction, (x, y, w, h) in self.adjust_buttons:
            cv2.rectangle(canvas, (x,y), (x+w,y+h), LINE, 1)
            text(canvas, '+' if direction > 0 else '-', (x+12,y+23), .48)
        x,y,w,h = self.stop_rect
        cv2.rectangle(canvas, (x,y), (x+w,y+h), STOP_COLOR, 1)
        cv2.rectangle(canvas, (x+15,y+20), (x+25,y+30), STOP_COLOR, -1)
        text(canvas, 'Stop output', (x+40,y+31), .45, STOP_COLOR, bold=True)
        text(canvas, 'Esc / X', (x+w-65,y+31), .35, MUTED)
        self._plain_button(canvas, self.exit_rect, 'Exit [Q]')
        status = self.controller.status_msg if self.controller else 'Offline preview / no hardware'
        text(canvas, self.grasp_feedback.message[:115], (24,876), .33)
        if self.task_controls is not None:
            cv2.rectangle(canvas, (24, 858), (1020, 881), PAPER, -1)
            text(canvas, self.task_controls.status[:126], (24,876), .33)
        text(canvas, f'BLE: {status} | {self.message}'[:150], (24,894), .29, MUTED)
        if fps > 0:
            text(canvas, f'{fps:.0f} fps', (1094, 878), .35, MUTED)
        return canvas

    def _plain_button(self, canvas, rect, label, active=False):
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x+w, y+h), ACCENT_FILL if active else PAPER, -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), ACCENT if active else LINE, 1)
        text(canvas, label, (x+12, y+h//2+5), .38, INK, bold=active)


def preview_inputs(held=False, *, width=640, height=480, trajectory=()):
    """Neutral synthetic workspace, explicitly labeled as offline input by the console."""
    from types import SimpleNamespace
    from .experiment import reference_route, draw_trial_overlay
    yy, xx = np.mgrid[:height, :width]
    shade = 246 - 7*((xx-width*.39)/width)**2 - 9*((yy-height*.35)/height)**2
    frame = np.repeat(shade[...,None], 3, axis=2).clip(0,255).astype(np.uint8)
    draw_workspace_boundary(frame)
    route = reference_route(width, height, workspace_bounds(width, height))
    draw_trial_overlay(frame, route, SimpleNamespace(active=None, samples=trajectory))
    draw_sphere(frame, initial_ball_position(width, height), 44, held=held)
    return frame


def main():
    parser = argparse.ArgumentParser(description='Offline UI preview; no cameras, BLE, or session logs.')
    parser.add_argument('--preview', action='store_true', help='Open the offline preview window')
    parser.add_argument('--snapshot', help='Save an offline preview PNG and exit')
    args = parser.parse_args()
    controller = ManualBleController(False, '', '')
    panel = OperatorPanel(controller)
    base = preview_inputs()
    def preview_frame():
        top = base.copy()
        right = draw_haptic_status(top, controller.feedback_snapshot())
        draw_view_status(top, 'WAIT CUE [G]', left=right+8,
                         right=top.shape[1]-route_legend_width(top.shape[1])-16)
        return top
    if args.snapshot:
        if not cv2.imwrite(args.snapshot, panel.render(preview_frame(), simulated=True)):
            raise RuntimeError(f'Could not write snapshot: {args.snapshot}')
        return
    panel.open()
    while not panel.quit_requested:
        ready, cancelled = panel.grasp_feedback.poll_prompt()
        if ready:
            panel.grasp_status = 'PREVIEW / NO GESTURES'
        if cancelled:
            panel.grasp_status = 'WAIT CUE [G]'
        panel.manual_request = None
        if panel.trial_request is not None:
            panel.trial_request = None
            panel.restart_pending = False
            panel.message = 'Trials and restart are available in a camera session'
        if panel.record_requests:
            panel.record_requests.clear()
            panel.message = 'Recording is available in a camera session; preview saves no video'
        if panel.ui_record_request:
            panel.ui_record_request = False
            panel.message = 'UI recording is available in a camera session; preview saves no video'
        panel.show(preview_frame(), simulated=True)
        panel.handle_key(cv2.waitKeyEx(30))
        if cv2.getWindowProperty(panel.window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

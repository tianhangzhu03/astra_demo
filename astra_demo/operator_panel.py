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
from .manual_control import Actuation, Cue


# OpenCV colors are BGR. Labels and outlines duplicate every color cue.
INK = (98, 64, 37)
MUTED = (126, 100, 79)
ACCENT = (148, 91, 43)
LINE = (229, 220, 210)
PAPER = (252, 250, 247)


@lru_cache(maxsize=24)
def _font(pixels):
    candidates = [
        str(Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'bahnschrift.ttf'),
        '/System/Library/Fonts/Avenir Next.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for path in candidates:
        try:
            index = (2 if pixels >= 18 else 7) if path.endswith('Avenir Next.ttc') else 0
            return ImageFont.truetype(path, pixels, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


@lru_cache(maxsize=256)
def _text_mask(label, pixels):
    font = _font(pixels)
    left, top, right, bottom = font.getbbox(label, anchor='ls')
    image = Image.new('L', (max(1, right - left), max(1, bottom - top)))
    ImageDraw.Draw(image).text((-left, -top), label, font=font, fill=255, anchor='ls')
    return np.asarray(image, dtype=np.float32)[..., None] / 255, left, top


def text(canvas, label, xy, size=0.45, color=INK):
    alpha, dx, dy = _text_mask(str(label), max(10, round(size * 34)))
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
    canvas[y:y+h, x:x+w] = (246, 241, 235)
    if frame is None:
        text(canvas, empty_label, (x + 22, y + h // 2), 0.45, MUTED)
        return
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    rw, rh = max(1, round(fw * scale)), max(1, round(fh * scale))
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    ox, oy = x + (w - rw) // 2, y + (h - rh) // 2
    canvas[oy:oy+rh, ox:ox+rw] = resized


def workspace_bounds(w, h):
    # Keep X longer than Y regardless of the incoming camera aspect ratio.
    height = min(h * .84, w * .90 / 1.50)
    half_width = height * 1.50 / 2
    return (round(w / 2 - half_width), round(w / 2 + half_width),
            round((h - height) / 2), round((h + height) / 2))


def initial_ball_position(w, h):
    x0, x1, y0, y1 = workspace_bounds(w, h)
    return (x0 + (x1 - x0) * .25, y0 + (y1 - y0) * .25)


def draw_workspace_boundary(frame):
    h, w = frame.shape[:2]
    x0, x1, y0, y1 = workspace_bounds(w, h)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (165, 165, 165), 1, cv2.LINE_AA)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    # Reference guides only: these divisions have no hit-test or control semantics.
    dash, gap = max(5, round(w / 64)), max(4, round(w / 80))
    for x in range(x0, x1, dash + gap):
        cv2.line(frame, (x, cy), (min(x + dash, x1), cy), (195, 195, 195), 1, cv2.LINE_AA)
    for y in range(y0, y1, dash + gap):
        cv2.line(frame, (cx, y), (cx, min(y + dash, y1)), (195, 195, 195), 1, cv2.LINE_AA)


class OperatorPanel:
    window_name = 'Haptic Guidance | Operator'

    def __init__(self, controller=None, settings=None):
        self.controller = controller
        self.settings = settings or Actuation()
        self.settings.validate()
        self.message = 'Set voltage, then select a cue' if controller else 'Offline preview only'
        self._held_keys = set()
        self._key_down = None
        if os.name == 'nt':
            import ctypes
            self._key_down = lambda vk: bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
        self.last_cue = None
        self.cue_time = 0.0
        self.quit_requested = False
        self.buttons = {
            Cue.FORWARD: (940, 470, 96, 58),
            Cue.LEFT: (836, 534, 96, 58),
            Cue.NORMAL: (940, 534, 96, 58),
            Cue.RIGHT: (1044, 534, 96, 58),
            Cue.BACK: (940, 598, 96, 58),
            Cue.GRASP: (836, 670, 304, 60),
        }
        self.stop_rect = (836, 742, 304, 50)
        self.exit_rect = (992, 804, 148, 36)
        self.adjust_buttons = []
        for field, y in [('volts_mv', 337), ('duration_s', 377), ('frequency_hz', 417)]:
            self.adjust_buttons.extend([(field, -1, (1056, y, 36, 32)),
                                        (field, 1, (1104, y, 36, 32))])
        self._base = np.full((900, 1180, 3), PAPER, dtype=np.uint8)
        text(self._base, 'Haptic guidance', (24, 56), 1.0)
        text(self._base, 'Manual operation', (25, 85), 0.45, MUTED)
        cv2.rectangle(self._base, (1024, 38), (1156, 69), LINE, 1)
        text(self._base, 'MANUAL BLE' if controller else 'LOCAL PREVIEW', (1038, 59), 0.35, MUTED)
        cv2.line(self._base, (24, 105), (1156, 105), LINE, 1)
        text(self._base, 'Workspace', (24, 141), 0.50)
        text(self._base, 'Top camera', (654, 141), 0.39, MUTED)
        text(self._base, 'Side camera', (836, 141), 0.50)
        text(self._base, 'WASD / arrows: direction    Space: normal force    G: vibration', (24, 778), .42)
        text(self._base, 'Esc / X: stop output    Q: stop and exit', (24, 809), .42)
        text(self._base, 'One finite pulse per press. Settings apply to the next cue.', (24, 835), .36, MUTED)
        cv2.line(self._base, (816, 122), (816, 840), LINE, 1)
        cv2.line(self._base, (24, 855), (1156, 855), LINE, 1)

    def open(self):
        # Fixed canvas geometry keeps mouse hit testing aligned on Windows HighGUI.
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

    def select(self, cue):
        accepted = True
        if self.controller:
            accepted, self.message = self.controller.submit(cue, self.settings)
        else:
            self.message = f'Preview: {cue.value} (no output)'
        if not accepted:
            return
        self.last_cue = cue
        self.cue_time = time.perf_counter()

    def stop_output(self):
        self.last_cue = None
        self.message = self.controller.stop_output() if self.controller else 'Preview: output stopped'

    def adjust(self, field, direction):
        step, lo, hi = {'volts_mv': (100, 0, 5000), 'duration_s': (.1, .1, 5.),
                        'frequency_hz': (10, 10, 200)}[field]
        value = min(hi, max(lo, round(getattr(self.settings, field) + direction * step, 2)))
        self.settings = replace(self.settings, **{field: value})

    def quit(self):
        self.stop_output()
        self.quit_requested = True

    def handle_key(self, key):
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
        if key in keys:
            # HighGUI repeats key-down events; on Windows require a real release.
            vk = {Cue.FORWARD: 0x26, Cue.BACK: 0x28, Cue.LEFT: 0x25, Cue.RIGHT: 0x27}.get(keys[key]) if key > 255 else ord(chr(key).upper())
            if self._key_down and vk is not None:
                if vk in self._held_keys:
                    return
                if self._key_down(vk):
                    self._held_keys.add(vk)
            self.select(keys[key])
        elif key in (27, ord('x')):
            self.stop_output()
        elif key == ord('q'):
            self.quit()
        else:
            adjustment = {ord('['): ('volts_mv', -1), ord(']'): ('volts_mv', 1),
                          ord('-'): ('duration_s', -1), ord('='): ('duration_s', 1),
                          ord(','): ('frequency_hz', -1), ord('.'): ('frequency_hz', 1)}.get(key)
            if adjustment:
                self.adjust(*adjustment)

    @staticmethod
    def _inside(x, y, rect):
        rx, ry, w, h = rect
        return rx <= x < rx+w and ry <= y < ry+h

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for cue, rect in self.buttons.items():
            if self._inside(x, y, rect):
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
        fill = ACCENT if active or primary else (255, 255, 255)
        fg = (255, 255, 255) if active or primary else INK
        cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), fill, -1)
        cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), INK if active else (ACCENT if primary else LINE), 3 if active else 1)
        if active:
            cv2.rectangle(canvas, (x+5, y+5), (x+w-6, y+h-6), (255, 255, 255), 1)
        if primary:
            text(canvas, 'Vibration', (x+18, y+27), 0.57, fg)
            text(canvas, 'Grasp cue', (x+18, y+47), 0.35, (255, 255, 255))
            text(canvas, 'G', (x+w-30, y+36), 0.46, fg)
            return
        keys = {Cue.FORWARD:'W', Cue.BACK:'S', Cue.LEFT:'A', Cue.RIGHT:'D', Cue.NORMAL:'Space'}
        text(canvas, keys[cue], (x+7, y+13), 0.28, fg if active else MUTED)
        cx, cy = x+w//2, y+25
        direction = {Cue.FORWARD:(0,-1), Cue.BACK:(0,1), Cue.LEFT:(-1,0), Cue.RIGHT:(1,0)}
        if cue in direction:
            dx, dy = direction[cue]
            cv2.arrowedLine(canvas, (cx-dx*8, cy-dy*8), (cx+dx*8, cy+dy*8), fg, 1, cv2.LINE_AA, tipLength=.4)
        else:
            # Out-of-plane force: a dot in a circle, distinct from output-stop's square.
            cv2.circle(canvas, (cx, cy), 7, fg, 1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 2, fg, -1, cv2.LINE_AA)
        label = cue.value
        mask, _, _ = _text_mask(label, 12)
        text(canvas, label, (x+(w-mask.shape[1])//2, y+49), 12/34, fg)

    def render(self, top=None, side=None, *, fps=0.0,
               ble_status='Disabled', simulated=False):
        canvas = self._base.copy()
        fit_camera(canvas, top, (24, 154, 768, 576), 'Waiting for top camera')
        fit_camera(canvas, side, (836, 154, 304, 171), 'Side camera not connected')
        age = time.perf_counter() - self.cue_time
        for cue, rect in self.buttons.items():
            self._button(canvas, cue, rect, cue == self.last_cue and age < .8)
        for label, value, hint, y in [
            ('Voltage', f'{self.settings.volts_mv} mV', '[ / ]', 337),
            ('Duration', f'{self.settings.duration_s:.1f} s', '- / =', 377),
            ('Vibration', f'{self.settings.frequency_hz} Hz', ', / .', 417),
        ]:
            text(canvas, label, (836, y+15), .39)
            text(canvas, hint, (836, y+30), .28, MUTED)
            text(canvas, value, (946, y+23), .46)
        for _, direction, (x, y, w, h) in self.adjust_buttons:
            cv2.rectangle(canvas, (x,y), (x+w,y+h), LINE, 1)
            text(canvas, '+' if direction > 0 else '-', (x+12,y+23), .48)
        x,y,w,h = self.stop_rect
        cv2.rectangle(canvas, (x,y), (x+w,y+h), INK, 2)
        cv2.rectangle(canvas, (x+15,y+19), (x+27,y+31), INK, -1)
        text(canvas, 'STOP OUTPUT', (x+40,y+31), .49)
        text(canvas, 'Esc / X', (x+w-65,y+31), .35, MUTED)
        text(canvas, 'Exit  Q', (1035,829), .4, MUTED)
        status = self.controller.status_msg if self.controller else 'Offline preview / no hardware'
        text(canvas, self.message[:115], (24,876), .38)
        text(canvas, f'BLE: {status}'[:140], (24,894), .33, MUTED)
        if fps > 0:
            text(canvas, f'{fps:.0f} fps', (1094, 878), .35, MUTED)
        return canvas


def preview_inputs():
    """Neutral synthetic workspace, explicitly labeled as offline input by the console."""
    yy, xx = np.mgrid[:480, :640]
    shade = 246 - 7*((xx-250)/640)**2 - 9*((yy-170)/480)**2
    frame = np.repeat(shade[...,None], 3, axis=2).clip(0,255).astype(np.uint8)
    draw_workspace_boundary(frame)
    draw_sphere(frame, initial_ball_position(640, 480), 44)
    return frame


def main():
    parser = argparse.ArgumentParser(description='Offline UI preview; no cameras, BLE, or session logs.')
    parser.add_argument('--preview', action='store_true', help='Open the offline preview window')
    parser.add_argument('--snapshot', help='Save an offline preview PNG and exit')
    args = parser.parse_args()
    panel = OperatorPanel()
    top = preview_inputs()
    if args.snapshot:
        if not cv2.imwrite(args.snapshot, panel.render(top, simulated=True)):
            raise RuntimeError(f'Could not write snapshot: {args.snapshot}')
        return
    panel.open()
    while not panel.quit_requested:
        cv2.imshow(panel.window_name, panel.render(top, simulated=True))
        panel.handle_key(cv2.waitKeyEx(30))
        if cv2.getWindowProperty(panel.window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

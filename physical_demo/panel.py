"""English operator canvas. It never estimates or draws a virtual object."""

from collections import deque
from functools import lru_cache
import os
from pathlib import Path
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1440, 900
INK = '#17344B'
MUTED = '#637585'
BLUE = '#145C8D'
PLUM = '#8B5A86'
LINE = '#DCE3E8'
BG = '#F5F7F9'


@lru_cache(maxsize=24)
def font(size, bold=False):
    candidates = [Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / ('arialbd.ttf' if bold else 'arial.ttf'),
                  Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def fit_text(draw, value, size, width, *, bold=False):
    value = str(value)
    while size > 11 and draw.textlength(value, font=font(size, bold)) > width:
        size -= 1
    while value and draw.textlength(value, font=font(size, bold)) > width:
        value = value[:-4] + '...' if len(value) > 4 else ''
    return value, font(size, bold)


class Panel:
    window_name = 'Physical Object Pilot | independent experiment'

    def __init__(self, config):
        self.config = config
        self.actions = deque()
        self.buttons = []
        self.transform = (0, 0, 1, 1)
        self._held = set()
        self._key_down = None
        if os.name == 'nt':
            import ctypes
            self._key_down = lambda vk: bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)

    def open(self):
        if os.name == 'nt':
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except (AttributeError, OSError):
                pass
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window_name, 1280, 800)
        cv2.setMouseCallback(self.window_name, self.click)

    def click(self, event, x, y, flags=None, userdata=None):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        dx, dy, sx, sy = self.transform
        x, y = (x-dx)/sx, (y-dy)/sy
        for rect, action in self.buttons:
            x0, y0, x1, y1 = rect
            if x0 <= x < x1 and y0 <= y < y1:
                self.actions.append(action)
                return

    def handle_key(self, key):
        if self._key_down:
            self._held = {vk for vk in self._held if self._key_down(vk)}
        if key < 0:
            return
        arrows = {2490368: ('forward', 0x26), 2621440: ('back', 0x28),
                  2424832: ('left', 0x25), 2555904: ('right', 0x27)}
        if key in arrows:
            action, vk = arrows[key]
        else:
            if 65 <= key <= 90:
                key += 32
            mapping = {'w': 'forward', 's': 'back', 'a': 'left', 'd': 'right', 'g': 'grasp_cue',
                       ' ': 'normal', '\t': 'normal_next', 'x': 'stop', '\x1b': 'stop', 'q': 'quit', 'n': 'new_trial',
                       'f': 'mark_grasp', 'r': 'mark_release', 'z': 'undo', 'e': 'end_trial',
                       '0': 'reset', 'b': 'record', 'l': 'route', '1': 'voltage_down',
                       '2': 'voltage_up', '3': 'duration_down', '4': 'duration_up',
                       '5': 'frequency_down', '6': 'frequency_up'}
            if key > 255 or chr(key) not in mapping:
                return
            action, vk = mapping[chr(key)], ord(chr(key).upper())
            if key == ord('0') and self._key_down and self._key_down(0x60):
                vk = 0x60
        if self._key_down:
            if vk in self._held:
                return
            if self._key_down(vk):
                self._held.add(vk)
        self.actions.append(action)

    def render(self, top, side, state):
        canvas = Image.new('RGB', (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        self.buttons = []

        def label(x, y, value, size=16, color=INK, bold=False, width=None):
            value, face = (fit_text(draw, value, size, width, bold=bold) if width
                           else (str(value), font(size, bold)))
            draw.text((x, y), value, font=face, fill=color)

        def button(rect, text, action, *, active=False, danger=False):
            fill = '#FCEDEC' if danger else '#E8F0F7' if active else '#FFFFFF'
            stroke = '#AC4640' if danger else BLUE if active else LINE
            draw.rounded_rectangle(rect, radius=6, fill=fill, outline=stroke, width=1)
            x0, y0, x1, y1 = rect
            label(x0+12, y0+11, text, 15, '#9E3933' if danger else INK, active, x1-x0-22)
            self.buttons.append((rect, action))

        label(24, 21, 'Physical Object Pilot', 28, bold=True)
        label(24, 58, f'{self.config.object_name}  |  Manual grasp / release annotations', 16, MUTED, width=940)
        recording = state.get('recording', False)
        draw.ellipse((1040, 29, 1050, 39), fill='#B54A44' if recording else '#99A5AF')
        label(1060, 23, 'REC / 3 videos' if recording else 'Recording paused', 17, bold=True)
        label(1040, 53, f'Session {state.get("session_s", 0):.1f} s', 14, MUTED)
        button((1280, 20, 1416, 66), 'Exit [Q]', 'quit')

        now = state.get('mono', 0)

        def camera(frame, name, rect, mirrored=False, view='top'):
            x0, y0, x1, y1 = rect
            label(x0, 87, name, 17, bold=True)
            draw.rounded_rectangle((x0-1, y0-1, x1+1, y1+1), radius=8, fill='#E8EDF1', outline=LINE)
            if frame is None:
                label(x0+24, (y0+y1)//2-10, 'Waiting for camera...', 20, MUTED)
                return
            raw = cv2.flip(frame.image, 1) if mirrored else frame.image
            picture = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            scale = min((x1-x0)/picture.width, (y1-y0)/picture.height)
            width, height = round(picture.width*scale), round(picture.height*scale)
            picture = picture.resize((width, height), Image.Resampling.BILINEAR)
            left, upper = x0+(x1-x0-width)//2, y0+(y1-y0-height)//2
            canvas.paste(picture, (left, upper))
            if name.startswith('TOP') and state.get('route'):
                points = [(left+round(width*.88), upper+round(height*.86)),
                          (left+round(width*.88), upper+round(height*.17)),
                          (left+round(width*.16), upper+round(height*.17))]
                draw.line(points, fill='white', width=7, joint='curve')
                draw.line(points, fill=BLUE, width=3, joint='curve')
                for (x, y), text in zip(points, ('START / DROP', 'TURN', 'PICKUP')):
                    draw.ellipse((x-4, y-4, x+4, y+4), fill=BLUE, outline='white', width=1)
                    tw = draw.textlength(text, font=font(13, True))
                    tx = max(left+4, min(left+width-tw-12, x-tw/2))
                    draw.rounded_rectangle((tx-4, y+10, tx+tw+5, y+31), radius=3, fill='white')
                    label(tx, y+13, text, 13, BLUE, True)
            trace = state.get('traces', {}).get(view)
            if trace:
                def project(point):
                    x, y = point
                    x = 1-x if mirrored else x
                    return (left+round(np.clip(x, 0, 1)*(width-1)), upper+round(np.clip(y, 0, 1)*(height-1)))
                stroke, path = None, []
                def paint_path():
                    if len(path) > 1:
                        draw.line(path, fill='white', width=6, joint='curve')
                        draw.line(path, fill=PLUM, width=3, joint='curve')
                for segment, x, y, captured in trace['points']:
                    if stroke != segment:
                        paint_path()
                        stroke, path = segment, []
                    path.append(project((x, y)))
                paint_path()
                if trace.get('current') is not None:
                    x, y = project(trace['current'])
                    draw.ellipse((x-6, y-6, x+6, y+6), fill=PLUM, outline='white', width=2)
            age = max(0, now-frame.mono)
            stale = age > self.config.stale_after_s
            detail = f'{frame.image.shape[1]} x {frame.image.shape[0]}  |  ' + (f'STALE {age:.1f}s' if stale else 'LIVE')
            if mirrored:
                detail += '  |  UI mirrored'
            if trace:
                detail += '  |  Hand: ' + trace['status']
            label(x0, 604, detail, 13, '#A2443E' if stale else MUTED, width=x1-x0)
            if stale:
                draw.rectangle((x0+10, y0+10, x1-10, y0+44), fill='#FFF0D6')
                label(x0+20, y0+18, 'No new camera frame - showing last image', 14, '#8B581F', width=x1-x0-40)

        camera(top, f'TOP / camera {self.config.top_camera}', (24, 112, 896, 600))
        camera(side, f'SIDE / camera {self.config.side_camera}', (920, 112, 1416, 600), self.config.side_mirror, 'side')
        for rect in ((24, 636, 704, 878), (728, 636, 1416, 878)):
            draw.rounded_rectangle(rect, radius=10, fill='white', outline=LINE)

        feedback = state.get('feedback')
        trigger = feedback.trigger if feedback else None
        cue = trigger.label if trigger else ''
        label(42, 652, 'Haptic controls', 18, bold=True)
        badge = 'SIMULATED / no output' if not self.config.ble_enabled else state.get('ble_status', 'Initializing')
        label(220, 655, badge, 14, MUTED, width=465)
        detail = f'{cue} / {trigger.volts_mv} mV / {trigger.duration_s:.1f} s' if trigger else 'HAPTIC: ' + (feedback.state.upper() if feedback else 'IDLE')
        label(42, 681, detail, 14, PLUM if 'GRASP' in cue else BLUE, width=360)
        normal = state.get('next_normal', 'lower')
        label(410, 681, f'Next: {normal.upper()} / Tab: change', 13, MUTED, width=276)
        for i, (text, action, match) in enumerate((('Forward [W]', 'forward', 'FORWARD'), ('Left [A]', 'left', 'LEFT'),
                                                   ('Right [D]', 'right', 'RIGHT'), ('Back [S]', 'back', 'BACK'))):
            button((42+i*160, 710, 194+i*160, 750), text, action, active=cue == match)
        button((42, 760, 252, 800), 'Grasp cue [G]', 'grasp_cue', active=cue == 'GRASP CUE')
        button((262, 760, 472, 800), normal.title()+' [Space]', 'normal', active=cue in ('NORMAL FORCE', 'LOWER CUE', 'LIFT CUE'))
        button((482, 760, 684, 800), 'Stop [X / Esc]', 'stop', danger=True)
        settings = state['settings']
        for x, caption, value, down, up in ((42, 'Voltage', f'{settings.volts_mv} mV', 'voltage_down', 'voltage_up'),
                                          (258, 'Duration', f'{settings.duration_s:.1f} s', 'duration_down', 'duration_up'),
                                          (474, 'Vibration', f'{settings.frequency_hz} Hz', 'frequency_down', 'frequency_up')):
            pair = {'Voltage': ('1', '2'), 'Duration': ('3', '4'), 'Vibration': ('5', '6')}[caption]
            label(x, 812, f'{caption}: {value}', 14, MUTED)
            button((x, 838, x+94, 869), f'- [{pair[0]}]', down)
            button((x+102, 838, x+196, 869), f'+ [{pair[1]}]', up)

        trial_id = state.get('trial_id') or '-'
        label(746, 652, f'Trial {trial_id}  /  {state.get("object_state", "UNMARKED")} (manual)', 18, bold=True, width=650)
        label(746, 681, state.get('trial_detail', 'Place the case at PICKUP, then start a trial.'), 14, MUTED, width=650)
        for i, (text, action) in enumerate((('New trial [N]', 'new_trial'), ('Mark grasp [F]', 'mark_grasp'),
                                           ('Mark release [R]', 'mark_release'), ('End trial [E]', 'end_trial'))):
            button((746+i*161, 710, 899+i*161, 750), text, action,
                   active=(action == 'mark_grasp' and state.get('object_state') == 'GRASPED') or
                          (action == 'mark_release' and state.get('object_state') == 'RELEASED'))
        for i, (text, action) in enumerate((('Reset [0]', 'reset'), ('Undo mark [Z]', 'undo'),
                                           ('Pause rec [B]' if recording else 'Record [B]', 'record'),
                                           ('Guide ON [L]' if state.get('route') else 'Guide OFF [L]', 'route'))):
            button((746+i*161, 760, 899+i*161, 800), text, action)
        message = state.get('message', 'Ready')
        lines = textwrap.wrap(message, width=83)[:2]
        for i, line in enumerate(lines):
            label(746, 817+i*20, line, 14, MUTED, width=650)
        label(24, 882, 'Purple: 2D hand trail. Blue: route guide. LOWER / LIFT use the same Normal 0 pulse. Height is not measured.', 11, MUTED)
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

    def show(self, frame):
        try:
            _, _, width, height = cv2.getWindowImageRect(self.window_name)
        except cv2.error:
            width, height = WIDTH, HEIGHT
        width, height = max(1, width), max(1, height)
        scale = min(width/WIDTH, height/HEIGHT)
        rw, rh = max(1, round(WIDTH*scale)), max(1, round(HEIGHT*scale))
        dx, dy = (width-rw)//2, (height-rh)//2
        canvas = np.full((height, width, 3), 246, np.uint8)
        canvas[dy:dy+rh, dx:dx+rw] = cv2.resize(frame, (rw, rh))
        self.transform = (dx, dy, rw/WIDTH, rh/HEIGHT)
        cv2.imshow(self.window_name, canvas)

import unittest
import cv2

from astra_demo.manual_control import Cue
from astra_demo.operator_panel import OperatorPanel, preview_inputs, workspace_bounds


class Recorder:
    status_msg = 'Fake connected'

    def __init__(self):
        self.calls = []

    def submit(self, cue, settings):
        self.calls.append((cue, settings))
        return True, 'Requested'

    def stop_output(self):
        self.calls.append('stop')
        return 'Stop requested'


class PanelTests(unittest.TestCase):
    def test_all_buttons_and_keys(self):
        r = Recorder()
        p = OperatorPanel(r)
        for cue, (x,y,w,h) in p.buttons.items():
            p.on_mouse(cv2.EVENT_LBUTTONDOWN, x+w//2, y+h//2, None, None)
            self.assertEqual(r.calls[-1][0], cue)
        for key,cue in [('w',Cue.FORWARD),('s',Cue.BACK),('a',Cue.LEFT),
                        ('d',Cue.RIGHT),(' ',Cue.NORMAL),('g',Cue.GRASP)]:
            p.handle_key(ord(key))
            self.assertEqual(r.calls[-1][0], cue)
        for key in (27,ord('x')):
            p.handle_key(key)
            self.assertEqual(r.calls[-1], 'stop')
        x,y,w,h=p.stop_rect
        p.on_mouse(cv2.EVENT_LBUTTONDOWN,x+2,y+2,None,None)
        self.assertEqual(r.calls[-1], 'stop')
        p.handle_key(ord('q'))
        self.assertTrue(p.quit_requested)
        self.assertEqual(r.calls[-1], 'stop')

    def test_settings_and_limits(self):
        p = OperatorPanel()
        p.handle_key(ord(']'))
        p.handle_key(ord('='))
        self.assertEqual(p.settings.volts_mv,100)
        self.assertEqual(p.settings.duration_s,.3)
        p.handle_key(ord('.'))
        self.assertEqual(p.settings.frequency_hz,110)
        p.handle_key(ord(','))
        self.assertEqual(p.settings.frequency_hz,100)
        for _ in range(100):
            p.adjust('duration_s',1)
            p.adjust('volts_mv',1)
            p.adjust('frequency_hz',1)
        self.assertEqual(p.settings.duration_s,5)
        self.assertEqual(p.settings.volts_mv,5000)
        self.assertEqual(p.settings.frequency_hz,200)
        for _ in range(100):
            p.adjust('duration_s',-1)
            p.adjust('frequency_hz',-1)
        self.assertEqual(p.settings.duration_s,.1)
        self.assertEqual(p.settings.frequency_hz,10)

    def test_windows_key_repeat_requires_release(self):
        r=Recorder()
        p=OperatorPanel(r)
        down={ord('G')}
        p._key_down=lambda key: key in down
        p.handle_key(ord('g'))
        p.handle_key(ord('g'))
        self.assertEqual(len(r.calls),1)
        down.clear()
        p.handle_key(-1)
        down.add(ord('G'))
        p.handle_key(ord('g'))
        self.assertEqual(len(r.calls),2)

    def test_preview_and_landscape_workspace(self):
        p=OperatorPanel()
        p.handle_key(ord('g'))
        self.assertIn('no output',p.message)
        self.assertEqual(p.render(preview_inputs()).shape,(900,1180,3))
        for w,h in [(640,480),(1280,720),(480,640)]:
            x0,x1,y0,y1=workspace_bounds(w,h)
            self.assertAlmostEqual((x1-x0)/(y1-y0),1.5,places=2)

import unittest
import cv2

from astra_demo.manual_control import Cue, ManualBleController, HapticDisplay, HapticTrigger
from astra_demo.operator_panel import OperatorPanel, preview_inputs, workspace_bounds, haptic_display_lines
from astra_demo.experiment import SessionClock


class Recorder:
    status_msg = 'Fake connected'
    enabled = True
    is_connected = True
    connection_id = 1

    def __init__(self):
        self.calls = []

    def submit(self, cue, settings, *, receipt=None):
        self.calls.append((cue, settings))
        if receipt is not None:
            receipt.status = 'completed'
            receipt.sent_at = receipt.finished_at = 0.
            receipt.connection_id = self.connection_id
        return True, 'Requested'

    def cancel(self, receipt):
        if receipt is not None:
            receipt.cancelled = True

    def stop_output(self):
        self.calls.append('stop')
        return 'Stop requested'


class PanelTests(unittest.TestCase):
    def test_trigger_words_distinguish_active_history_and_simulation(self):
        trigger = HapticTrigger('C1', 'GRASP CONFIRMATION', 102.5, 125, .2)
        clock = SessionClock(100., 0.)
        title, detail = haptic_display_lines(HapticDisplay('active', trigger), clock)
        self.assertEqual(title, 'HAPTIC: GRASP CONFIRMATION')
        self.assertIn('COMMAND SENT', detail)
        self.assertIn('SET 125 mV / 0.20 s', detail)
        self.assertIn('trigger t=2.500s', detail)
        title, detail = haptic_display_lines(HapticDisplay(last_trigger=trigger), clock)
        self.assertEqual(title, 'HAPTIC: IDLE')
        self.assertIn('LAST TRIGGER: GRASP CONFIRMATION', detail)
        self.assertNotIn('COMMAND SENT', detail)
        _, detail = haptic_display_lines(HapticDisplay('active', trigger, simulated=True), clock)
        self.assertIn('SIMULATED', detail)
        self.assertNotIn('COMMAND SENT', detail)

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
        p.handle_key(ord('2'))
        p.handle_key(ord('4'))
        self.assertEqual(p.settings.volts_mv,25)
        p.handle_key(ord('1'))
        self.assertEqual(p.settings.volts_mv,0)
        p.handle_key(ord('2'))
        self.assertEqual(p.settings.duration_s,.3)
        p.handle_key(ord('3'))
        self.assertEqual(p.settings.duration_s,.2)
        p.handle_key(ord('6'))
        self.assertEqual(p.settings.frequency_hz,110)
        p.handle_key(ord('5'))
        self.assertEqual(p.settings.frequency_hz,100)
        for _ in range(200):
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
            self.assertGreaterEqual((x1-x0)/w, .94)
            self.assertGreater((x1-x0), (y1-y0))
            self.assertTrue(0 <= x0 < x1 < w and 0 <= y0 < y1 < h)
            old_height = min(h*.84, w*.90/1.50)
            self.assertGreater((x1-x0)*(y1-y0), old_height**2*1.50)

    def test_grasp_cue_in_vision_only_does_not_start_ble(self):
        controller = ManualBleController(False, 'test', 'test')
        panel = OperatorPanel(controller)
        panel.select(Cue.GRASP)
        self.assertEqual(panel.grasp_feedback.poll_prompt(), (True, False))
        panel.set_grab_state(True, 'Hand tracked')
        self.assertIsNone(controller._pending)
        self.assertIsNone(controller._thread)
        self.assertIn('disabled', panel.grasp_feedback.message)

    def test_normal_keeps_grasp_and_manual_release_cancels_cue(self):
        panel = OperatorPanel()
        panel.set_grab_state(True, 'Hand tracked')
        panel.handle_key(ord(' '))
        self.assertTrue(panel.held)
        self.assertIsNone(panel.manual_request)
        panel.handle_key(ord('r'))
        self.assertEqual(panel.manual_request, 'release')
        self.assertEqual(panel.grasp_feedback.poll_prompt(), (False, True))

    def test_no_force_hold_bypass_and_independent_recording_controls(self):
        panel = OperatorPanel()
        panel._key_down = None
        panel.handle_key(ord('h'))
        self.assertIsNone(panel.manual_request)
        panel.handle_key(ord('c'))
        self.assertEqual(panel.manual_request, 'cancel')
        panel.handle_key(ord('r'))
        self.assertEqual(panel.manual_request, 'release')
        for key in 'tyb':
            panel.handle_key(ord(key))
        self.assertEqual(panel.record_requests, ['top', 'side', 'both'])
        panel.handle_key(ord('v'))
        self.assertEqual(panel.record_mode, 'Raw')
        panel.record_active['side'] = True
        panel.handle_key(ord('v'))
        self.assertEqual(panel.record_mode, 'Raw')

    def test_maximized_button_coordinates_map_to_original_layout(self):
        panel = OperatorPanel()
        panel._display_transform = (200, 0, 1.5, 1.5)
        x, y, w, h = panel.release_rect
        panel._on_window_mouse(cv2.EVENT_LBUTTONDOWN, 200 + (x+w/2)*1.5,
                               (y+h/2)*1.5, None, None)
        self.assertEqual(panel.manual_request, 'release')

    def test_trial_keys_and_buttons_do_not_send_haptic_commands(self):
        controller = Recorder()
        panel = OperatorPanel(controller)
        panel._key_down = None
        panel.handle_key(ord('n'))
        self.assertEqual(panel.trial_request, 'start')
        panel.handle_key(ord('e'))
        self.assertEqual(panel.trial_request, 'end')
        x,y,w,h = panel.trial_rect
        panel.on_mouse(cv2.EVENT_LBUTTONDOWN, x+w/2, y+h/2, None, None)
        self.assertEqual(panel.trial_request, 'start')
        self.assertEqual(controller.calls, [])

    def test_restart_cancels_feedback_and_waiting_commands_without_sending_a_cue(self):
        controller = Recorder()
        panel = OperatorPanel(controller)
        panel._key_down = None
        panel.select(Cue.GRASP)
        prompt = panel.grasp_feedback.prompt
        panel.grasp_feedback.poll_prompt()
        panel.set_grab_state(True, 'Hand tracked')
        confirmation = panel.grasp_feedback.confirmation
        before_restart = len(controller.calls)
        panel.record_requests = ['both']
        panel.manual_request = 'release'
        panel.handle_key(ord('0'))
        self.assertEqual(panel.trial_request, 'restart')
        self.assertTrue(panel.restart_pending)
        self.assertTrue(prompt.cancelled)
        if confirmation is not None:
            self.assertTrue(confirmation.cancelled)
        self.assertFalse(panel.grasp_feedback.output_pending)
        self.assertEqual(panel.record_requests, [])
        self.assertIsNone(panel.manual_request)
        before = list(controller.calls)
        panel.select(Cue.GRASP)
        panel.select(Cue.FORWARD)
        self.assertEqual(controller.calls, before)
        panel.handle_key(27)
        self.assertFalse(panel.restart_pending)
        self.assertIsNone(panel.trial_request)
        x, y, w, h = panel.restart_rect
        panel.on_mouse(cv2.EVENT_LBUTTONDOWN, x+w/2, y+h/2, None, None)
        self.assertEqual(panel.trial_request, 'restart')
        self.assertEqual(controller.calls[before_restart:], ['stop', 'stop', 'stop'])

    def test_held_zero_key_does_not_restart_repeatedly(self):
        for vk in (ord('0'), 0x60):
            with self.subTest(vk=vk):
                controller = Recorder()
                panel = OperatorPanel(controller)
                down = {vk}
                panel._key_down = lambda key: key in down
                panel.handle_key(ord('0'))
                panel.handle_key(ord('0'))
                self.assertEqual(controller.calls, ['stop'])
                down.clear()
                panel.handle_key(-1)
                down.add(vk)
                panel.handle_key(ord('0'))
                self.assertEqual(controller.calls, ['stop', 'stop'])

    def test_ui_recording_controls_are_independent_of_trial_and_camera_recording(self):
        controller=Recorder()
        panel=OperatorPanel(controller)
        panel._key_down=None
        panel.handle_key(ord('u'))
        self.assertTrue(panel.ui_record_request)
        self.assertEqual(panel.record_requests,[])
        self.assertIsNone(panel.trial_request)
        self.assertEqual(controller.calls,[])
        panel.ui_record_request=False
        x,y,w,h=panel.ui_record_rect
        panel.on_mouse(cv2.EVENT_LBUTTONDOWN,x+w/2,y+h/2,None,None)
        self.assertTrue(panel.ui_record_request)
        panel.ui_record_active=True
        panel.handle_key(ord('0'))
        self.assertTrue(panel.ui_record_active)

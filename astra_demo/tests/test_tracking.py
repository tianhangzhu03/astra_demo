import unittest
from dataclasses import replace
from types import SimpleNamespace
import cv2
import numpy as np

from astra_demo.tracking import (HandObservation, OneEuroFilter, PalmTracker, TopHandTracker, observe_hand,
                                 PositionSmoother, DisplayPalmTrace)
from astra_demo.virtual_prop import VirtualProp, VirtualPropState


def observation():
    return HandObservation(np.array([150., 160.]), np.eye(2), 80.,
                           np.array([155., 110.]), .2, .04,
                           np.array([[130., 200.], [100., 130.], [150., 120.],
                                     [180., 130.], [190., 170.]]))


class TrackingTests(unittest.TestCase):
    def test_manual_abc_grab_centers_both_views_and_retains_on_loss(self):
        from astra_demo.linked_views import LinkedSideView
        frame = np.zeros((480, 640, 3), np.uint8)
        obs = observation()
        top = TopHandTracker()
        side = LinkedSideView(palm_inset=1.)
        side.set_frame_size((640, 480))
        side.place([450., 400.])
        top.observe(frame, obs, 1.)
        np.testing.assert_allclose(top.follow_ball([450.,400.], True, palm_centered=True), obs.palm)
        side.update(frame, obs, True, 1.)
        np.testing.assert_allclose(side.prop.pos_xy, obs.palm)
        top.observe(frame, None, 2.)
        self.assertIsNone(top.follow_ball([450.,400.], True, palm_centered=True))
        side.update(frame, None, True, 2.)
        np.testing.assert_allclose(side.prop.pos_xy, obs.palm)
        self.assertEqual(side.prop.state, VirtualPropState.HELD)
        # Releasing and grabbing elsewhere must not revive the old placement offset.
        top.follow_ball(obs.palm, False, palm_centered=True)
        side.update(frame, None, False, 2.1)
        moved = replace(obs, palm=obs.palm+[50,30], pinch_xy=obs.pinch_xy+[50,30],
                        palm_points=obs.palm_points+[50,30])
        top.observe(frame, moved, 3.)
        np.testing.assert_allclose(top.follow_ball(obs.palm, True, palm_centered=True), moved.palm)
        side.update(frame, moved, True, 3.)
        np.testing.assert_allclose(side.prop.pos_xy, moved.palm)

    def test_top_flow_tracks_current_palm_before_grasp_and_expires(self):
        tracker = TopHandTracker()
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
        obs = observation()
        tracker.observe(frame, obs, 1.)
        shifted = cv2.warpAffine(frame, np.float32([[1, 0, 6], [0, 1, 4]]), (300, 300))
        tracker.observe(shifted, None, 1.1)
        self.assertTrue(tracker.valid)
        self.assertEqual(tracker.status, 'Motion tracking')
        np.testing.assert_allclose(tracker.palm_points, obs.palm_points+[6,4], atol=.3)
        self.assertEqual(tracker.record()['age_ms'], 100)
        tracker.observe(shifted, None, 1.6)
        self.assertFalse(tracker.valid)
        self.assertIsNone(tracker.follow_ball([100.,100.], True))

    def test_top_grasp_offset_does_not_scale_rotate_or_snap_ball(self):
        tracker = TopHandTracker()
        frame = np.zeros((300, 300, 3), np.uint8)
        obs = observation()
        tracker.observe(frame, obs, 1.)
        np.testing.assert_allclose(tracker.follow_ball([110.,100.], True), [110.,100.])
        for i in range(1, 20):
            changed = replace(obs, basis=np.array([[0., -1.],[1., 0.]]), scale=160.,
                              pinch_xy=obs.pinch_xy+[100.,100.])
            tracker.observe(frame, changed, 1.+i/15)
            np.testing.assert_allclose(tracker.follow_ball([110.,100.], True), [110.,100.])

    def test_top_image_flow_damps_landmark_noise_without_using_wrist_proxy(self):
        tracker = TopHandTracker()
        rng = np.random.default_rng(13)
        frame = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
        obs = observation()
        points = []
        for i in range(60):
            delta = np.array([4.*(-1)**i, 5.*(-1)**i])
            shaky = replace(obs, palm=obs.palm+delta, palm_points=obs.palm_points+delta)
            tracker.observe(frame, shaky, i/15)
            points.append(tracker.point)
        self.assertLess(np.array(points[20:]).std(axis=0).max(), 1.5)
        # Wrist moves alone: it is not substituted for the entire palm center.
        shifted = obs.palm_points.copy()
        shifted[0] += [0,40]
        tracker.observe(frame, replace(obs, palm_points=shifted), 4.)
        self.assertLess(np.linalg.norm(np.array(tracker.point)-obs.palm), 5.)

    def test_top_ball_and_trail_share_translation_after_reacquisition(self):
        tracker = TopHandTracker()
        frame = np.zeros((400, 400, 3), np.uint8)
        obs = observation()
        tracker.observe(frame, obs, 1.)
        tracker.follow_ball([100.,100.], True)
        offset = np.array([100.,100.])-tracker.point
        tracker.observe(frame, None, 2.)
        self.assertFalse(tracker.valid)
        tracker.observe(frame, replace(obs, palm=obs.palm+[80,60], palm_points=obs.palm_points+[80,60]), 2.1)
        np.testing.assert_allclose(tracker.follow_ball([100.,100.], True), np.array(tracker.point)+offset)
        self.assertGreater(tracker.segment_id, 1)

    def test_display_trace_suppresses_jitter_at_live_fps_and_breaks_on_loss(self):
        for fps in (10, 15, 30):
            trace = DisplayPalmTrace()
            raw = np.array([[100.+3*(-1)**i, 100.+4*(-1)**i] for i in range(60)])
            result = np.array([trace.update(p, i/fps) for i,p in enumerate(raw)])
            self.assertLess(result[20:].std(axis=0).max(), raw[20:].std(axis=0).max()*.4)
            self.assertIsNone(trace.update(None, 61/fps))
            self.assertIsNone(trace.update([200., 200.], 62/fps))
            np.testing.assert_allclose(trace.update([200., 200.], 63/fps), [200., 200.])
            np.testing.assert_allclose(trace.update([300., 300.], 64/fps+1), [300., 300.])

    def test_wrist_display_rejects_pitch_motion_but_retains_translation(self):
        trace = DisplayPalmTrace()
        for i in range(60):
            point = trace.update([100., 150.+35*np.sin(i*.4)], i/15, wrist=[100., 210.], scale=140.)
            np.testing.assert_allclose(point, [100., 150.])
        for i in range(1, 31):
            point = trace.update([100.+4*i, 150.+35*np.sin(i*.4)], (59+i)/15,
                                 wrist=[100.+4*i, 210.], scale=140.)
        self.assertLess(220-point[0], 20.)
        self.assertGreater(point[0], 200.)

    def test_isolated_jump_is_not_drawn_and_brief_gaps_keep_filter(self):
        trace = DisplayPalmTrace()
        trace.update([100., 100.], 0.)
        self.assertIsNone(trace.update([220., 170.], .1))
        self.assertEqual(trace.status, 'jump_rejected')
        np.testing.assert_allclose(trace.update([100., 100.], .2), [100., 100.])
        trace.update(None, .3)
        point = trace.update([104., 104.], .4)
        self.assertLess(point[0], 104.)
        segment = trace.segment_id
        trace.update([220., 170.], .5)
        point = trace.update([221., 170.], .6)
        np.testing.assert_allclose(point, [221., 170.])
        self.assertGreater(trace.segment_id, segment)

    def test_horizontal_motion_does_not_unlock_vertical_jitter(self):
        smoother = PositionSmoother()
        result = np.array([smoother.update([100.+10*i, 100.+4*(-1)**i], i/15) for i in range(60)])
        self.assertLess(result[20:,1].std(), 1.)
        self.assertLess(690-result[-1,0], 20.)
        slow = PositionSmoother()
        for i in range(61): point = slow.update([100.+i*.2, 100.], i/15)
        self.assertGreater(point[0], 109.)

    def test_side_on_palm_is_trackable_without_a_gesture_decision(self):
        pts = np.column_stack((np.full(21,150.), np.linspace(200.,100.,21)))
        pts[0] = [150.,220.]
        pts[5],pts[9],pts[13],pts[17] = [150.,150.],[150.,140.],[151.,145.],[152.,150.]
        lms = SimpleNamespace(landmark=[SimpleNamespace(x=x/640,y=y/480) for x,y in pts])
        self.assertIsNone(observe_hand(lms,640,480))
        edge = observe_hand(lms,640,480,allow_edge_on=True)
        self.assertIsNotNone(edge)
        self.assertIsNone(edge.pinch_ratio)
        self.assertTrue(np.isfinite(edge.basis).all())

    def test_side_flow_preserves_fixed_grasp_offset(self):
        tracker = PalmTracker(translation_only=True)
        rng = np.random.default_rng(42)
        frame = rng.integers(0,256,(300,300,3),dtype=np.uint8)
        obs = observation()
        tracker.update(frame,obs,True,1.)
        transform = cv2.getRotationMatrix2D(tuple(obs.palm),8,1.04)
        shifted = cv2.warpAffine(frame,transform,(300,300))
        self.assertIsNotNone(tracker.update(shifted,None,True,1.1))
        np.testing.assert_allclose(tracker.anchor-tracker.palm,obs.pinch_xy-obs.palm,atol=.01)

    def test_finger_jitter_does_not_move_held_ball(self):
        tracker = PalmTracker()
        frame = np.zeros((300, 300, 3), np.uint8)
        obs = observation()
        initial = tracker.update(frame, obs, True, 1.)
        for i in range(1, 20):
            shaky = replace(obs, pinch_xy=obs.pinch_xy + [30 * (-1)**i, 20])
            np.testing.assert_allclose(tracker.update(frame, shaky, True, 1 + i / 30), initial)
        shifted = replace(obs, palm=obs.palm + [20, 0], palm_points=obs.palm_points + [20, 0])
        point = tracker.update(frame, shifted, True, 2.)
        self.assertGreater(point[0], initial[0])

    def test_optical_flow_bridges_translation_but_expires(self):
        tracker = PalmTracker()
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
        obs = observation()
        tracker.update(frame, obs, True, 1.)
        shifted = cv2.warpAffine(frame, np.float32([[1, 0, 6], [0, 1, 4]]), (300, 300))
        point = tracker.update(shifted, None, True, 1.1)
        self.assertIsNotNone(point)
        self.assertEqual(tracker.status, 'Motion tracking')
        np.testing.assert_allclose(tracker.anchor, obs.pinch_xy + [6, 4], atol=.3)
        self.assertIsNone(tracker.update(shifted, None, True, 2.))
        self.assertIn('hold retained', tracker.status)

    def test_one_frame_palm_jump_is_rejected_then_reacquired(self):
        tracker = PalmTracker()
        frame = np.zeros((500, 500, 3), np.uint8)
        obs = observation()
        tracker.update(frame, obs, True, 1.)
        distant = replace(obs, palm=obs.palm + [180, 0], palm_points=obs.palm_points + [180, 0])
        self.assertIsNone(tracker.update(frame, distant, True, 1.1))
        self.assertIsNotNone(tracker.update(frame, distant, True, 1.2))

    def test_filter_reduces_stationary_jitter_without_long_motion_lag(self):
        filt = OneEuroFilter()
        rng = np.random.default_rng(1)
        raw = rng.normal(100, 3, (120, 2))
        smoothed = np.array([filt.update(p, i / 30) for i, p in enumerate(raw)])
        self.assertLess(smoothed[30:].std(), raw[30:].std() * .65)
        for i in range(1, 31):
            desired = np.array([100 + i * 5., 100.])
            actual = filt.update(desired, 4 + i / 30)
        self.assertLess(np.linalg.norm(desired - actual), 15)

    def test_ratio_is_scale_invariant_and_bad_tips_are_unknown(self):
        pts = np.array([[150, 220], [130, 200], [110, 180], [100, 155], [110, 135],
                        [120, 155], [120, 125], [122, 105], [125, 85],
                        [150, 145], [150, 115], [150, 90], [150, 70],
                        [175, 150], [175, 120], [175, 100], [175, 80],
                        [195, 165], [200, 140], [200, 120], [200, 100]], dtype=float)
        def hand(points):
            return SimpleNamespace(landmark=[SimpleNamespace(x=x/640, y=y/480) for x, y in points])
        original = observe_hand(hand(pts), 640, 480)
        scaled = observe_hand(hand(pts * 1.3), 640, 480)
        self.assertIsNotNone(original.pinch_ratio)
        self.assertAlmostEqual(original.pinch_ratio, scaled.pinch_ratio)
        pts[8] = [-200, 85]
        uncertain = observe_hand(hand(pts), 640, 480)
        self.assertIsNotNone(uncertain)
        self.assertIsNone(uncertain.pinch_ratio)

    def test_prop_keeps_position_and_color_when_hand_missing(self):
        prop = VirtualProp(size_follow=20, size_held=20)
        prop.update((150, 150), True, 0)
        prop.update(None, True, 10000)
        self.assertEqual(prop.state, VirtualPropState.HELD)
        self.assertEqual(prop.pos_xy, (150, 150))
        held = np.full((300, 300, 3), 240, np.uint8)
        prop.draw(held)
        prop.update(None, False, 11000)
        released = np.full_like(held, 240)
        prop.draw(released)
        self.assertEqual(prop.pos_xy, (150, 150))
        self.assertGreater(held[150, 150, 1], held[150, 150, 0])
        self.assertGreater(released[150, 150, 0], released[150, 150, 1])

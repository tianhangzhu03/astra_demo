import unittest
from types import SimpleNamespace

import numpy as np

from astra_demo.grab_logic import GrabState
from astra_demo.wearable_grip import WearableGrip, four_finger_pose, near_palm, wearable_observation


def openness(image, world):
    return four_finger_pose(image, packed(world), 640, 480)[0]


def landmarks(closed=False):
    world = np.zeros((21, 3))
    for base, x, y in [(5, -.03, .07), (9, -.01, .08), (13, .01, .07), (17, .03, .06)]:
        world[base] = [x, y, 0]
        direction = world[base] / np.linalg.norm(world[base])
        perpendicular = np.array([0., 0., 1.])
        angle1, angle2 = np.radians([80, 155] if closed else [0, 0])
        world[base+1] = world[base] + .03*(direction*np.cos(angle1) + perpendicular*np.sin(angle1))
        world[base+2] = world[base+1] + .02*(direction*np.cos(angle2) + perpendicular*np.sin(angle2))
    image = world[:, :2]*1800 + [250, 100]
    return image, world


def packed(world):
    return SimpleNamespace(landmark=[SimpleNamespace(x=x, y=y, z=z) for x, y, z in world])


def observation(score):
    return SimpleNamespace(grip_openness=score) if score is not None else None


class WearableGripTests(unittest.TestCase):
    def test_masked_fingertips_do_not_change_joint_pose(self):
        for closed in (False, True):
            image, world = landmarks(closed)
            score = openness(image, world)
            self.assertIsNotNone(score)
            self.assertLessEqual(score, .42) if closed else self.assertGreaterEqual(score, .85)
            # Including ring/little tips: no tip participates in this mode.
            for joint in [4, 8, 12, 16, 20]:
                image[joint] = [-999, 999]
                world[joint] = [np.nan, np.nan, np.nan]
            self.assertEqual(score, openness(image, world))

    def test_world_rotation_does_not_turn_open_hand_into_closed(self):
        image, world = landmarks()
        for angle in [0, .5, 1.0, 1.5]:
            rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                                 [-np.sin(angle), 0, np.cos(angle)]])
            self.assertGreater(openness(image, world@rotation.T), .85)

    def test_four_finger_consensus_tolerates_one_bad_or_disagreeing_chain(self):
        image, world = landmarks()
        _, curled = landmarks(True)
        world[17:20] = curled[17:20]
        self.assertGreater(openness(image, world), .85)  # Three open vs one closed.
        world[13:16] = curled[13:16]
        self.assertEqual(openness(image, world), .5)  # Two vs two is ambiguous.
        world[13:20] = np.nan
        self.assertGreater(openness(image, world), .85)  # Both remaining chains agree.
        world[5:8] = np.nan
        self.assertIsNone(openness(image, world))  # A single chain cannot decide.
        self.assertIsNone(four_finger_pose(image, None, 640, 480)[0])

    def run_step(self, grip, top, side, now, **kwargs):
        gates = dict(hand_near_ball=True, top_hand_present=True, top_anchor_valid=True,
                     setup_ready=True, now_ms=now)
        gates.update(kwargs)
        return grip.update(observation(top), observation(side), **gates)

    def grab(self, grip, source='side'):
        result = None
        for i, score in enumerate([.95, .95, .2, .2, .2]):
            values = (score, None) if source == 'top' else (None, score)
            result = self.run_step(grip, *values, i*100, prompt_sent=i == 0)
        self.assertTrue(result.trigger_on)
        self.assertEqual(grip.source, source)

    def test_side_grasp_needs_no_top_finger_pose_and_loss_keeps_hold(self):
        grip = WearableGrip()
        self.grab(grip)
        for t in range(500, 5000, 100):
            self.assertTrue(self.run_step(grip, None, None, t).trigger_on)

    def test_no_cue_no_attach_and_no_open_close_splicing_between_views(self):
        grip = WearableGrip()
        for i, value in enumerate([.95]*3+[.2]*5):
            self.assertFalse(self.run_step(grip, value, value, i*100).trigger_on)
        for i, (top, side) in enumerate([(None, .95)]*3+[(.2, None)]*8):
            out = self.run_step(grip, top, side, 1000+i*100, prompt_sent=i == 0)
            self.assertFalse(out.trigger_on)

    def test_wrong_top_pose_cannot_veto_side_or_release_on_its_own(self):
        grip = WearableGrip()
        for i, value in enumerate([.95]*3 + [.2]*3):
            out = self.run_step(grip, .2 if value > .6 else .95, value, i*100, prompt_sent=i == 0)
        self.assertTrue(out.trigger_on)
        for t in range(500, 2000, 100):
            self.assertTrue(self.run_step(grip, .95, .2, t).trigger_on)

    def test_release_needs_continuous_open_in_one_view(self):
        grip = WearableGrip()
        self.grab(grip)
        for t, top, side in [(500, .95, None), (600, .95, None), (700, .95, None),
                             (800, None, .95), (900, None, .95), (1000, None, None)]:
            self.assertTrue(self.run_step(grip, top, side, t).trigger_on)
        for t in (1100, 1200, 1300, 1400):
            self.assertTrue(self.run_step(grip, None, .95, t).trigger_on)
        self.assertFalse(self.run_step(grip, None, .95, 1500).trigger_on)
        for t in range(1600, 2300, 100):
            self.assertFalse(self.run_step(grip, .2, .2, t).trigger_on)

    def test_closing_outside_target_requires_reopening(self):
        grip = WearableGrip(median_window=1)
        self.run_step(grip, None, .95, 0, prompt_sent=True)
        self.run_step(grip, None, .95, 100)
        self.run_step(grip, None, .2, 200, hand_near_ball=False)
        for t in (300, 400, 500):
            self.assertFalse(self.run_step(grip, None, .2, t).trigger_on)

    def test_current_side_closure_cancels_release_before_median_catches_up(self):
        grip = WearableGrip()
        self.grab(grip)
        for t in (500, 600, 700, 800, 900):
            self.assertTrue(self.run_step(grip, .95, .95, t).trigger_on)
        self.assertTrue(self.run_step(grip, .95, .2, 1000).trigger_on)

    def test_brief_occlusion_and_manual_release(self):
        grip = WearableGrip(median_window=1)
        for i, value in enumerate([.95, .95, None, None, .2, .2]):
            out = self.run_step(grip, None, value, i*100, prompt_sent=i == 0)
        self.assertTrue(out.trigger_on)
        self.assertFalse(self.run_step(grip, None, None, 600, manual_release=True).trigger_on)
        self.assertEqual(grip.context.state, GrabState.IDLE)

    def test_top_proximity_uses_current_palm_region_not_finger_pose(self):
        from astra_demo.tests.test_tracking import observation
        obs = observation()
        obs.grip_xy = None
        obs.grip_openness = None
        obs.pinch_xy = np.array([900., 900.])
        self.assertTrue(near_palm(obs, (120., 140.), 48))
        self.assertFalse(near_palm(obs, (450., 400.), 48))
        self.assertFalse(near_palm(None, (120., 140.), 48))
        converted = wearable_observation(obs, top=True)
        np.testing.assert_allclose(converted.pinch_xy, obs.palm_points[1:].mean(axis=0))

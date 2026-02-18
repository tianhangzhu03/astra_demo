import unittest

import numpy as np

from astra_demo.grab_logic import GrabContext, GrabState, sample_depth_5x5, update_grab_state


class GrabLogicTests(unittest.TestCase):
    def test_depth_sample_median_filters_invalid(self):
        depth = np.zeros((10, 10), dtype=np.uint16)
        depth[3:8, 3:8] = np.array(
            [
                [500, 510, 0, 520, 530],
                [500, 510, 9999, 520, 530],
                [500, 510, 520, 520, 530],
                [500, 510, 520, 520, 530],
                [500, 510, 520, 520, 530],
            ],
            dtype=np.uint16,
        )
        self.assertEqual(sample_depth_5x5(depth, (5, 5)), 520)

    def test_state_machine_reaches_grab_and_releases(self):
        ctx = GrabContext()

        # Enter ARMED with stable pinch + hover.
        out = update_grab_state(
            ctx,
            pinch_dist=0.05,
            depth_at_mid_mm=700,
            hover_key=4,
            pinch_enter=0.075,
            pinch_exit=0.095,
            depth_enter_mm=520,
            depth_exit_mm=560,
            enter_frames=2,
            exit_frames=2,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.IDLE)

        out = update_grab_state(
            ctx,
            pinch_dist=0.05,
            depth_at_mid_mm=700,
            hover_key=4,
            pinch_enter=0.075,
            pinch_exit=0.095,
            depth_enter_mm=520,
            depth_exit_mm=560,
            enter_frames=2,
            exit_frames=2,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.ARMED)

        # Enter GRAB with stable near depth.
        out = update_grab_state(
            ctx,
            pinch_dist=0.05,
            depth_at_mid_mm=510,
            hover_key=4,
            pinch_enter=0.075,
            pinch_exit=0.095,
            depth_enter_mm=520,
            depth_exit_mm=560,
            enter_frames=2,
            exit_frames=2,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.ARMED)

        out = update_grab_state(
            ctx,
            pinch_dist=0.05,
            depth_at_mid_mm=510,
            hover_key=4,
            pinch_enter=0.075,
            pinch_exit=0.095,
            depth_enter_mm=520,
            depth_exit_mm=560,
            enter_frames=2,
            exit_frames=2,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.GRAB)
        self.assertTrue(out.trigger_on)
        self.assertEqual(out.target_key, 4)

        # Release with depth gate off for 2 frames.
        out = update_grab_state(
            ctx,
            pinch_dist=0.05,
            depth_at_mid_mm=900,
            hover_key=4,
            pinch_enter=0.075,
            pinch_exit=0.095,
            depth_enter_mm=520,
            depth_exit_mm=560,
            enter_frames=2,
            exit_frames=2,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.GRAB)

        out = update_grab_state(
            ctx,
            pinch_dist=0.05,
            depth_at_mid_mm=900,
            hover_key=4,
            pinch_enter=0.075,
            pinch_exit=0.095,
            depth_enter_mm=520,
            depth_exit_mm=560,
            enter_frames=2,
            exit_frames=2,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.IDLE)
        self.assertFalse(out.trigger_on)


if __name__ == "__main__":
    unittest.main()

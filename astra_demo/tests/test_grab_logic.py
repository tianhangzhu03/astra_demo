import unittest

from astra_demo.grab_logic import GrabContext, GrabState, update_grab_state


class GrabLogicTests(unittest.TestCase):
    def test_state_machine_reaches_grab_and_releases(self):
        ctx = GrabContext()

        # Enter ARMED with stable pinch + top-view presence.
        out = update_grab_state(
            ctx,
            pinch_dist=0.05,
            hover_key=4,
            pinch_enter=0.075,
            pinch_exit=0.094,
            enter_frames=2,
            exit_frames=1,
            top_hand_present=True,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.IDLE)

        out = update_grab_state(
            ctx,
            pinch_dist=0.05,
            hover_key=4,
            pinch_enter=0.075,
            pinch_exit=0.094,
            enter_frames=2,
            exit_frames=1,
            top_hand_present=True,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.ARMED)

        # Enter GRAB after another stable interval in ARMED.
        out = update_grab_state(
            ctx,
            pinch_dist=0.05,
            hover_key=4,
            pinch_enter=0.075,
            pinch_exit=0.094,
            enter_frames=2,
            exit_frames=1,
            top_hand_present=True,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.ARMED)

        out = update_grab_state(
            ctx,
            pinch_dist=0.05,
            hover_key=4,
            pinch_enter=0.075,
            pinch_exit=0.094,
            enter_frames=2,
            exit_frames=1,
            top_hand_present=True,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.GRAB)
        self.assertTrue(out.trigger_on)
        self.assertEqual(out.target_key, 4)

        # Release immediately when the pinch opens.
        out = update_grab_state(
            ctx,
            pinch_dist=0.12,
            hover_key=4,
            pinch_enter=0.075,
            pinch_exit=0.094,
            enter_frames=2,
            exit_frames=1,
            top_hand_present=True,
        )
        self.assertEqual(out.context.state, GrabState.IDLE)
        self.assertFalse(out.trigger_on)

    def test_short_missing_pinch_does_not_release_immediately(self):
        ctx = GrabContext(state=GrabState.GRAB, top_pinch_state=True, grab_key=2, armed_key=2)

        out = update_grab_state(
            ctx,
            pinch_dist=None,
            hover_key=2,
            pinch_enter=0.075,
            pinch_exit=0.094,
            enter_frames=2,
            exit_frames=2,
            pinch_missing_hold_frames=2,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.GRAB)
        self.assertTrue(out.trigger_on)

        out = update_grab_state(
            ctx,
            pinch_dist=None,
            hover_key=2,
            pinch_enter=0.075,
            pinch_exit=0.094,
            enter_frames=2,
            exit_frames=2,
            pinch_missing_hold_frames=2,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.GRAB)
        self.assertTrue(out.trigger_on)

        out = update_grab_state(
            ctx,
            pinch_dist=None,
            hover_key=2,
            pinch_enter=0.075,
            pinch_exit=0.094,
            enter_frames=2,
            exit_frames=2,
            pinch_missing_hold_frames=2,
        )
        ctx = out.context
        self.assertEqual(ctx.state, GrabState.GRAB)
        self.assertTrue(out.trigger_on)

        out = update_grab_state(
            ctx,
            pinch_dist=None,
            hover_key=2,
            pinch_enter=0.075,
            pinch_exit=0.094,
            enter_frames=2,
            exit_frames=2,
            pinch_missing_hold_frames=2,
        )
        self.assertEqual(out.context.state, GrabState.IDLE)
        self.assertFalse(out.trigger_on)

    def test_open_pinch_releases_immediately_when_exit_frames_is_one(self):
        ctx = GrabContext(state=GrabState.GRAB, top_pinch_state=True, grab_key=3, armed_key=3)

        out = update_grab_state(
            ctx,
            pinch_dist=0.120,
            hover_key=3,
            pinch_enter=0.075,
            pinch_exit=0.094,
            enter_frames=2,
            exit_frames=1,
            pinch_missing_hold_frames=4,
        )
        self.assertEqual(out.context.state, GrabState.IDLE)
        self.assertFalse(out.trigger_on)


if __name__ == "__main__":
    unittest.main()

import unittest
from astra_demo.grab_logic import GrabContext, GrabState, near_ball, update_grab_state


def step(ctx, ratio, now, **kwargs):
    options = dict(hand_near_ball=True, top_hand_present=True, now_ms=now)
    options.update(kwargs)
    return update_grab_state(ctx, ratio, **options).context


class GrabLogicTests(unittest.TestCase):
    def held(self):
        ctx = step(GrabContext(), .65, 0, prompt_sent=True)
        ctx = step(ctx, .65, 33)
        for t in (66, 99):
            ctx = step(ctx, .2, t)
        self.assertEqual(ctx.state, GrabState.GRAB)
        return ctx

    def test_unprompted_gestures_cannot_attach(self):
        ctx = GrabContext()
        for i, ratio in enumerate([.7] * 10 + [.2] * 20):
            ctx = step(ctx, ratio, i * 33)
            self.assertEqual(ctx.state, GrabState.IDLE)

    def test_all_three_conditions_are_required(self):
        for missing in ({'hand_near_ball': False}, {'top_hand_present': False}):
            ctx = step(GrabContext(), .7, 0, prompt_sent=True, **missing)
            for i, ratio in enumerate([.7] * 4 + [.2] * 10):
                ctx = step(ctx, ratio, 33 + i * 33, **missing)
                self.assertNotEqual(ctx.state, GrabState.GRAB)
        self.held()  # A partly open .65 pose, not full release .85, is sufficient.

    def test_closed_before_cue_must_open_again(self):
        ctx = step(GrabContext(), .2, 0, prompt_sent=True)
        for i in range(20):
            ctx = step(ctx, .2, (i + 1) * 33)
            self.assertNotEqual(ctx.state, GrabState.GRAB)
        for i, ratio in enumerate([.65, .65, .2, .2]):
            ctx = step(ctx, ratio, 1000 + i * 33)
        self.assertEqual(ctx.state, GrabState.GRAB)

    def test_closing_far_away_then_entering_target_cannot_attach(self):
        ctx = step(GrabContext(), .7, 0, prompt_sent=True)
        ctx = step(ctx, .7, 33)
        ctx = step(ctx, .2, 66, hand_near_ball=False)
        for i in range(10):
            ctx = step(ctx, .2, 100 + i * 33)
            self.assertNotEqual(ctx.state, GrabState.GRAB)

    def test_single_open_spike_cannot_arm_closure(self):
        ctx = step(GrabContext(), .7, 0, prompt_sent=True)
        for i in range(10):
            ctx = step(ctx, .2, 33 + i * 33)
            self.assertNotEqual(ctx.state, GrabState.GRAB)
        ctx = step(ctx, .7, 500)
        ctx = step(ctx, None, 533)
        for i in range(10):
            ctx = step(ctx, .2, 600 + i * 33)
            self.assertNotEqual(ctx.state, GrabState.GRAB)

    def test_brief_side_occlusion_retains_confirmed_opening(self):
        ctx = step(GrabContext(), .7, 0, prompt_sent=True)
        ctx = step(ctx, .7, 90)
        for t in (180, 270, 360, 450):
            ctx = step(ctx, None, t)
            self.assertTrue(ctx.open_observed)
            self.assertEqual(ctx.state, GrabState.ARMED)
        ctx = step(ctx, .2, 540)
        self.assertEqual(ctx.state, GrabState.ARMED)
        self.assertEqual(step(ctx, .2, 630).state, GrabState.GRAB)

    def test_loss_cannot_count_toward_consecutive_closure(self):
        ctx = step(GrabContext(), .7, 0, prompt_sent=True)
        ctx = step(ctx, .7, 90)
        for t, ratio in [(180, .2), (270, None), (360, .2)]:
            ctx = step(ctx, ratio, t)
            self.assertEqual(ctx.state, GrabState.ARMED)
        self.assertEqual(step(ctx, .2, 450).state, GrabState.GRAB)

    def test_long_loss_or_paused_frames_require_new_opening(self):
        for missing in (False, True):
            ctx = step(GrabContext(), .7, 0, prompt_sent=True)
            ctx = step(ctx, .7, 90)
            if missing:
                ctx = step(ctx, None, 180)
            for t in (600, 690, 780):
                ctx = step(ctx, .2, t)
                self.assertFalse(ctx.open_observed)
                self.assertEqual(ctx.state, GrabState.ARMED)

    def test_top_loss_or_leaving_target_invalidates_retained_opening(self):
        for missing in ({'top_hand_present': False}, {'top_anchor_valid': False},
                        {'hand_near_ball': False}, {'setup_ready': False}):
            ctx = step(GrabContext(), .7, 0, prompt_sent=True)
            ctx = step(ctx, .7, 90)
            ctx = step(ctx, None, 180, **missing)
            for t in (270, 360, 450):
                ctx = step(ctx, .2, t)
                self.assertFalse(ctx.open_observed)
                self.assertEqual(ctx.state, GrabState.ARMED)

    def test_status_identifies_actual_blocker(self):
        for options, status in [({'setup_ready': False}, 'PLACE SIDE BALL'),
                                ({'top_hand_present': False}, 'SHOW TOP HAND'),
                                ({'top_anchor_valid': False}, 'SHOW TOP FINGERS'),
                                ({'hand_near_ball': False}, 'MOVE TO BALL')]:
            kwargs = dict(hand_near_ball=True, top_hand_present=True, prompt_sent=True)
            kwargs.update(options)
            self.assertEqual(update_grab_state(GrabContext(), .7, **kwargs).status, status)
        self.assertEqual(update_grab_state(GrabContext(), None, prompt_sent=True,
                         hand_near_ball=True, top_hand_present=True).status, 'SHOW SIDE FINGERS')

    def test_cancel_and_each_release_need_a_new_prompt(self):
        ctx = step(GrabContext(), .7, 0, prompt_sent=True)
        ctx = step(ctx, .7, 33, cancel_prompt=True)
        for i, ratio in enumerate([.7] * 3 + [.2] * 3):
            ctx = step(ctx, ratio, 100 + i * 33)
            self.assertEqual(ctx.state, GrabState.IDLE)
        for manual in (True, False):
            ctx = self.held()
            if manual:
                ctx = step(ctx, None, 100, manual_release=True)
            else:
                for t in (100, 300, 450):
                    ctx = step(ctx, 1.1, t)
            self.assertEqual(ctx.state, GrabState.IDLE)
            for i, ratio in enumerate([.7] * 3 + [.2] * 3):
                ctx = step(ctx, ratio, 1000 + i * 33)
            self.assertEqual(ctx.state, GrabState.IDLE)

    def test_occlusion_proximity_and_cancel_do_not_drop_held_ball(self):
        ctx = self.held()
        for i in range(200):
            ctx = step(ctx, None, 200 + i * 100, top_hand_present=False,
                       hand_near_ball=False, cancel_prompt=True)
            self.assertEqual(ctx.state, GrabState.GRAB)

    def test_release_requires_sustained_clear_open(self):
        ctx = self.held()
        for t in (100, 200, 300, 449):
            ctx = step(ctx, 1.1, t)
            self.assertEqual(ctx.state, GrabState.GRAB)
        self.assertEqual(step(ctx, 1.1, 450).state, GrabState.IDLE)

    def test_uncertain_or_closed_interrupts_release_timer(self):
        for interruption in (None, .6, .2, float('nan')):
            ctx = self.held()
            for ratio, t in [(1.1, 100), (1.1, 300), (interruption, 400), (1.1, 500), (1.1, 700)]:
                ctx = step(ctx, ratio, t)
            self.assertEqual(ctx.state, GrabState.GRAB)
            self.assertEqual(step(ctx, 1.1, 850).state, GrabState.IDLE)

    def test_low_fps_single_open_sample_cannot_release(self):
        ctx = step(self.held(), 1.1, 100)
        ctx = step(ctx, 1.1, 1000)
        self.assertEqual(ctx.state, GrabState.GRAB)
        self.assertEqual(step(ctx, 1.1, 1100).state, GrabState.IDLE)

    def test_proximity_uses_current_ball_and_finite_coordinates(self):
        self.assertTrue(near_ball((164, 100), (100, 100), 44, 20))
        self.assertFalse(near_ball((165, 100), (100, 100), 44, 20))
        self.assertFalse(near_ball((100, 100), (400, 300), 44, 20))
        self.assertTrue(near_ball((400, 300), (400, 300), 44, 20))
        self.assertFalse(near_ball(None, (100, 100), 44))
        self.assertFalse(near_ball((float('nan'), 100), (100, 100), 44))

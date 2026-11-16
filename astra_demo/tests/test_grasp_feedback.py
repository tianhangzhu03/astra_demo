import unittest

from astra_demo.grasp_feedback import GraspFeedback
from astra_demo.manual_control import Actuation, Cue


class Controller:
    enabled = True
    is_connected = True
    connection_id = 1

    def __init__(self):
        self.calls = []
        self.busy = False
        self.error = None

    def submit(self, cue, settings, *, receipt=None):
        if self.error:
            return False, self.error
        if self.busy:
            return False, 'Busy; wait'
        self.busy = True
        receipt.connection_id = self.connection_id
        self.calls.append((cue, settings, receipt))
        return True, 'Requested'

    def cancel(self, receipt):
        if receipt is not None:
            receipt.status = 'cancelled'
            receipt.cancelled = True


class FeedbackTests(unittest.TestCase):
    def prepare(self):
        controller = Controller()
        feedback = GraspFeedback(controller)
        self.assertTrue(feedback.request_prompt(Actuation(100, .2, 100))[0])
        return controller, feedback

    def sent(self, feedback):
        feedback.prompt.status = 'sent'
        feedback.prompt.sent_at = 1.
        self.assertEqual(feedback.poll_prompt(observed_at=1.01), (True, False))

    def complete_prompt(self, controller, feedback):
        feedback.prompt.status = 'completed'
        feedback.prompt.finished_at = 1.3
        controller.busy = False

    def test_permission_requires_successful_write_and_a_fresh_frame(self):
        _, feedback = self.prepare()
        self.assertEqual(feedback.poll_prompt(), (False, False))
        feedback.prompt.status = 'sent'
        feedback.prompt.sent_at = 1.
        self.assertEqual(feedback.poll_prompt(observed_at=.99), (False, False))
        self.assertEqual(feedback.poll_prompt(observed_at=1.01), (True, False))
        self.assertEqual(feedback.poll_prompt(observed_at=1.02), (False, False))

    def test_failed_expired_cancelled_or_rejected_prompt_never_arms(self):
        for state in ('failed', 'expired', 'cancelled'):
            _, feedback = self.prepare()
            feedback.prompt.status = state
            self.assertEqual(feedback.poll_prompt(), (False, True))
            self.assertEqual(feedback.poll_prompt(), (False, False))
        for reason in ('Not connected', 'Busy', 'Set voltage above 0 mV first'):
            controller = Controller()
            controller.error = reason
            feedback = GraspFeedback(controller)
            self.assertFalse(feedback.request_prompt(Actuation(100))[0])
            self.assertEqual(feedback.poll_prompt(), (False, False))

    def test_two_identical_single_pulses_in_order_and_only_once(self):
        controller, feedback = self.prepare()
        self.sent(feedback)
        feedback.update(True, 1.05)  # Fast grasp while the prompt pulse is still running.
        feedback.update(True, 1.2)
        self.assertEqual(len(controller.calls), 1)
        self.complete_prompt(controller, feedback)
        feedback.update(True, 1.4)  # Still in the gap after prompt output stopped.
        self.assertEqual(len(controller.calls), 1)
        feedback.update(True, 1.46)
        self.assertEqual(len(controller.calls), 2)
        self.assertEqual(controller.calls[0][:2], controller.calls[1][:2])
        self.assertEqual(controller.calls[1][0], Cue.GRASP)
        feedback.confirmation.status = 'sent'
        for i in range(100):
            feedback.update(True, 1.5 + i * .01)
        self.assertEqual(len(controller.calls), 2)
        self.assertFalse(feedback.request_prompt(Actuation(200))[0])

    def test_release_cancel_or_disconnect_discards_waiting_confirmation(self):
        for event in ('release', 'cancel', 'disconnect', 'reconnect'):
            controller, feedback = self.prepare()
            self.sent(feedback)
            feedback.update(True, 1.05)
            self.complete_prompt(controller, feedback)
            if event == 'release':
                feedback.update(False, 1.1)
            elif event == 'cancel':
                feedback.cancel()
            elif event == 'disconnect':
                controller.is_connected = False
            else:
                controller.connection_id += 1
            feedback.update(event != 'release', 1.8)
            controller.is_connected = True
            feedback.update(event != 'release', 2.)
            self.assertEqual(len(controller.calls), 1, event)

    def test_release_cancels_submitted_but_unsent_confirmation(self):
        controller, feedback = self.prepare()
        self.sent(feedback)
        self.complete_prompt(controller, feedback)
        feedback.update(True, 1.5)
        receipt = feedback.confirmation
        self.assertEqual(receipt.status, 'pending')
        feedback.update(False, 1.6)
        self.assertTrue(receipt.cancelled)
        self.assertFalse(feedback.output_pending)

    def test_busy_confirmation_retries_boundedly_without_replaying_errors(self):
        controller, feedback = self.prepare()
        self.sent(feedback)
        self.complete_prompt(controller, feedback)
        controller.busy = True
        feedback.update(True, 1.5)
        feedback.update(True, 1.6)
        controller.busy = False
        feedback.update(True, 1.7)
        self.assertEqual(len(controller.calls), 2)

        controller, feedback = self.prepare()
        self.sent(feedback)
        feedback.update(True, 1.05)
        self.complete_prompt(controller, feedback)
        controller.busy = True
        feedback.update(True, 3.)  # Beyond the deadline; never replay on a future trial.
        controller.busy = False
        feedback.update(True, 3.1)
        self.assertEqual(len(controller.calls), 1)

    def test_vision_only_simulates_permission_without_any_submit(self):
        controller = Controller()
        controller.enabled = False
        feedback = GraspFeedback(controller)
        self.assertTrue(feedback.request_prompt(Actuation())[0])
        self.assertEqual(feedback.poll_prompt(), (True, False))
        feedback.update(True)
        self.assertEqual(controller.calls, [])

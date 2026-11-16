"""One operator prompt and one success pulse, with no replay across trials or connections."""

import time

from .manual_control import CommandReceipt, Cue


class GraspFeedback:
    def __init__(self, controller=None, confirmation_enabled=True):
        self.controller = controller
        self.confirmation_enabled = confirmation_enabled
        self.held = False
        self.prompt = self.confirmation = None
        self.settings = None
        self._announced = self._cancel_event = False
        self._waiting = False
        self._deadline = 0.0
        self.message = 'Press G at the ball, then open and pinch'

    @property
    def hardware_enabled(self):
        return self.controller is not None and self.controller.enabled

    @property
    def output_pending(self):
        return self._waiting or (self.confirmation is not None
                                 and self.confirmation.status in ('pending', 'sent'))

    def request_prompt(self, settings):
        if self.held:
            return False, 'Already holding; release before a new grasp cue'
        receipt = CommandReceipt(role='grab_prompt')
        if self.hardware_enabled:
            accepted, reason = self.controller.submit(Cue.GRASP, settings, receipt=receipt)
            if not accepted:
                return False, reason
        else:
            if self.controller is not None and hasattr(self.controller, 'simulate'):
                _, reason = self.controller.simulate(Cue.GRASP, settings, receipt=receipt)
            else:
                receipt.status = 'completed'
                receipt.sent_at = receipt.finished_at = time.monotonic()
                reason = 'Simulated grasp cue (no output)'
        # A new accepted prompt supersedes an old completed permission.
        if self.prompt is not None and self.hardware_enabled:
            self.controller.cancel(self.prompt)
        self.prompt, self.settings = receipt, settings
        self._announced = self._cancel_event = False
        self.message = reason
        return True, reason

    def cancel(self):
        if self.controller is not None and hasattr(self.controller, 'cancel'):
            self.controller.cancel(self.prompt)
            self.controller.cancel(self.confirmation)
        self.prompt = self.confirmation = None
        self._announced = self._waiting = False
        self._cancel_event = True
        self.message = 'Grasp cue cancelled'

    def poll_prompt(self, observed_at=None):
        """Return one-shot (prompt_sent, cancel_permission) events for the visual state machine."""
        if self._cancel_event:
            self._cancel_event = False
            return False, True
        if self.prompt is None:
            return False, False
        invalid_link = self.hardware_enabled and (
            not self.controller.is_connected or self.prompt.connection_id != self.controller.connection_id)
        if invalid_link or self.prompt.cancelled or self.prompt.status in ('failed', 'expired', 'cancelled'):
            self.cancel()
            self._cancel_event = False
            self.message = 'Cue unavailable / cancelled; press G again when ready'
            return False, True
        if not self._announced and self.prompt.status in ('sent', 'completed'):
            if observed_at is not None and (self.prompt.sent_at is None or observed_at < self.prompt.sent_at):
                return False, False
            self._announced = True
            self.message = 'Cue sent / open and pinch near the ball'
            return True, False
        return False, False

    def update(self, held, now=None):
        now = time.monotonic() if now is None else now
        rising, falling = held and not self.held, self.held and not held
        self.held = held
        if falling:
            self.cancel()
            return
        if rising:
            if not self.hardware_enabled or not self.confirmation_enabled:
                self.message = 'Grasp confirmed / vibration disabled'
            elif self.prompt is not None and self._announced:
                self._waiting = True
                self._deadline = now + self.settings.duration_s + 1.0
                self.message = 'Grasp confirmed / waiting for prompt pulse to end'
        if not held or not self.hardware_enabled:
            return
        # Reconnect cannot revive a queued confirmation, even if the UI missed the disconnect.
        if self.prompt is not None and (not self.controller.is_connected
                or self.prompt.connection_id != self.controller.connection_id
                or self.prompt.cancelled or self.prompt.status in ('failed', 'expired', 'cancelled')):
            self.cancel()
            self.message = 'Confirmation cancelled / connection or cue changed'
            return
        if self._waiting:
            if now >= self._deadline:
                self._waiting = False
                self.message = 'Confirmation expired; no delayed replay'
            elif (self.prompt is not None and self.prompt.status == 'completed'
                  and self.prompt.finished_at is not None and now >= self.prompt.finished_at + .15):
                # Wait for the first pulse's zero-output/stop sequence, then leave a short gap.
                # Both signals remain ONE pulse with identical captured settings.
                receipt = CommandReceipt(role='grasp_confirmation')
                accepted, reason = self.controller.submit(Cue.GRASP, self.settings, receipt=receipt)
                if accepted:
                    self.confirmation, self._waiting = receipt, False
                    self.message = 'Confirmation requested / single pulse'
                elif not reason.startswith('Busy'):
                    self._waiting = False
                    self.message = 'Confirmation skipped: ' + reason
        if self.confirmation is not None:
            if self.confirmation.status in ('sent', 'completed'):
                self.message = 'Confirmation sent / single pulse'
            elif self.confirmation.status in ('failed', 'cancelled', 'expired'):
                self.message = 'Confirmation not sent / ' + self.confirmation.status

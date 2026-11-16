import asyncio
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from astra_demo.manual_control import (
    Actuation, CommandReceipt, Cue, DIRECTIONS, ManualBleController, STOP_PACKET, command_packet,
)
from astra_demo.grasp_feedback import GraspFeedback


class FakeClient:
    def __init__(self, properties=None):
        self.is_connected = True
        self.writes = []
        self.char = SimpleNamespace(uuid='test', properties=properties or ['write'])
        self.services = [SimpleNamespace(characteristics=[self.char])]

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False

    async def write_gatt_char(self, uuid, packet, response):
        self.writes.append((packet, response))


async def wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(.005)
    await asyncio.wait_for(poll(), timeout=2)


class ProtocolTests(unittest.TestCase):
    def test_simulated_indicator_expires_and_never_consumes_logged_events(self):
        controller = ManualBleController(False, 'test', 'test')
        receipt = CommandReceipt(role='grab_prompt')
        with patch('time.monotonic', return_value=100.0):
            controller.simulate(Cue.GRASP, Actuation(125, .2), receipt=receipt)
        display = controller.feedback_snapshot(100.1)
        self.assertTrue(display.simulated)
        self.assertEqual(display.trigger.label, 'GRASP CUE')
        self.assertEqual((display.trigger.volts_mv, display.trigger.duration_s), (125, .2))
        ended = controller.feedback_snapshot(100.21)
        self.assertEqual(ended.state, 'idle')
        self.assertIsNone(ended.trigger)
        self.assertEqual(ended.last_trigger.label, 'GRASP CUE')
        self.assertIsNone(controller.feedback_snapshot(102.1).last_trigger)
        self.assertEqual([e['event'] for e in controller.drain_events()],
                         ['command_requested', 'command_simulated'])
        controller.stop_output()
        self.assertEqual(controller.feedback_snapshot(100.15).state, 'idle')
        self.assertIsNone(controller._thread)

    def test_cancel_simulated_prompt_hides_active_indicator(self):
        controller = ManualBleController(False, 'test', 'test')
        feedback = GraspFeedback(controller)
        feedback.request_prompt(Actuation(125, 5))
        self.assertEqual(controller.feedback_snapshot().state, 'active')
        feedback.cancel()
        self.assertEqual(controller.feedback_snapshot().state, 'idle')

    def test_all_cues_and_endianness(self):
        for cue, direction in DIRECTIONS.items():
            packet = command_packet(cue, Actuation(2500, .2, 100))
            self.assertEqual(packet, bytes([255, 16, int(cue == Cue.GRASP), direction,
                                           196, 9, 100 if cue == Cue.GRASP else 0, 0, 1]))

    def test_invalid_settings(self):
        for settings in [Actuation(-1), Actuation(5001), Actuation(duration_s=0),
                         Actuation(duration_s=6), Actuation(duration_s=float('nan')),
                         Actuation(frequency_hz=201), Actuation(frequency_hz=9)]:
            with self.assertRaises(ValueError):
                command_packet(Cue.NORMAL, settings)

    def test_disconnected_and_zero_voltage_discarded(self):
        controller = ManualBleController(True, 'test', 'test')
        self.assertFalse(controller.submit(Cue.GRASP, Actuation(100))[0])
        controller.running = controller.is_connected = True
        self.assertFalse(controller.submit(Cue.GRASP, Actuation())[0])
        self.assertIsNone(controller._pending)

    def test_vision_only_never_starts_worker_or_accepts_feedback(self):
        controller = ManualBleController(False, 'test', 'test')
        controller.start()
        self.assertIsNone(controller._thread)
        self.assertFalse(controller.submit(Cue.GRASP, Actuation(100))[0])
        self.assertIsNone(controller._pending)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.c = ManualBleController(True, 'test', 'test')
        self.c.running = True
        self.client = FakeClient()
        self.task = asyncio.create_task(self.c._serve(self.client, self.client.char))
        await wait_until(lambda: self.c.is_connected)
        self.client.writes.clear()

    async def asyncTearDown(self):
        self.c.running = False
        await self.task

    async def test_indicator_follows_successful_write_and_commanded_duration(self):
        receipt = CommandReceipt(role='grasp_confirmation')
        self.c.submit(Cue.GRASP, Actuation(125, .1), receipt=receipt)
        self.assertEqual(self.c.feedback_snapshot().state, 'pending')
        self.assertIsNone(self.c.feedback_snapshot().trigger)
        await wait_until(lambda: receipt.status == 'sent')
        display = self.c.feedback_snapshot()
        self.assertEqual(display.state, 'active')
        self.assertEqual(display.trigger.label, 'GRASP CONFIRMATION')
        # A rejected button press must not replace the command currently on the wire.
        self.assertFalse(self.c.submit(Cue.LEFT, Actuation(125, .1))[0])
        self.assertEqual(self.c.feedback_snapshot().trigger, display.trigger)
        self.assertEqual(self.c.feedback_snapshot(receipt.sent_at + .11).state, 'stopping')
        await wait_until(lambda: receipt.status == 'completed')
        self.assertEqual(self.c.feedback_snapshot().state, 'idle')
        self.assertIsNone(self.c.feedback_snapshot().trigger)

    async def test_indicator_clears_on_stop_and_disconnection(self):
        self.c.submit(Cue.NORMAL, Actuation(125, 5))
        await wait_until(lambda: self.c.feedback_snapshot().state == 'active')
        self.assertEqual(self.c.feedback_snapshot().trigger.label, 'NORMAL FORCE')
        self.c.stop_output()
        self.assertEqual(self.c.feedback_snapshot().state, 'stopping')
        self.assertIsNone(self.c.feedback_snapshot().trigger)
        await wait_until(lambda: not self.c.busy)
        self.assertEqual(self.c.feedback_snapshot().state, 'idle')
        self.c.submit(Cue.FORWARD, Actuation(125, 5))
        await wait_until(lambda: self.c.feedback_snapshot().state == 'active')
        self.c.is_connected = False
        display = self.c.feedback_snapshot()
        self.assertEqual(display.state, 'disconnected')
        self.assertIsNone(display.trigger)
        self.assertIsNone(display.last_trigger)

    async def test_finite_pulse_and_no_queue(self):
        self.assertTrue(self.c.submit(Cue.LEFT, Actuation(100, .1))[0])
        self.assertFalse(self.c.submit(Cue.RIGHT, Actuation(100, .1))[0])
        await wait_until(lambda: not self.c._busy)
        self.assertEqual([p for p, _ in self.client.writes], [
            command_packet(Cue.LEFT, Actuation(100, .1)),
            bytes([255, 16, 0, 0, 0, 0, 0, 0, 1]), STOP_PACKET])

    async def test_stop_cancels_unsent_command(self):
        self.c.submit(Cue.NORMAL, Actuation(100, 5))
        self.c.stop_output()
        await wait_until(lambda: not self.c._busy)
        self.assertEqual(len(self.client.writes), 2)
        self.assertEqual(self.client.writes[-1][0], STOP_PACKET)

    async def test_stop_interrupts_active_output_and_no_old_timer(self):
        self.c.submit(Cue.GRASP, Actuation(100, 5))
        await wait_until(lambda: len(self.client.writes) == 1)
        self.c.stop_output()
        await wait_until(lambda: not self.c._busy)
        self.assertEqual(self.client.writes[-1][0], STOP_PACKET)
        self.assertTrue(self.c.submit(Cue.NORMAL, Actuation(100, .1))[0])
        await wait_until(lambda: not self.c._busy)
        self.assertEqual(len(self.client.writes), 6)

    async def test_write_without_response(self):
        self.client.char.properties = ['write-without-response']
        await self.c._write(self.client, self.client.char, STOP_PACKET)
        self.assertFalse(self.client.writes[-1][1])

    async def test_tracked_command_reports_sent_then_completed(self):
        receipt = CommandReceipt()
        self.c.submit(Cue.GRASP, Actuation(100, .1), receipt=receipt)
        self.assertEqual(receipt.status, 'pending')
        await wait_until(lambda: receipt.status == 'sent')
        self.assertIsNotNone(receipt.sent_at)
        await wait_until(lambda: receipt.status == 'completed')
        self.assertGreater(receipt.finished_at, receipt.sent_at)
        self.assertEqual(receipt.connection_id, self.c.connection_id)

    async def test_cancelling_receipt_before_send_drops_only_that_command(self):
        receipt = CommandReceipt()
        self.c.submit(Cue.GRASP, Actuation(100, .1), receipt=receipt)
        self.c.cancel(receipt)
        await asyncio.sleep(.03)
        self.assertEqual(self.client.writes, [])
        self.assertEqual(receipt.status, 'cancelled')
        self.assertTrue(self.c.submit(Cue.LEFT, Actuation(100, .1))[0])
        self.c.cancel(receipt)
        await wait_until(lambda: not self.c._busy)
        self.assertEqual(self.client.writes[0][0], command_packet(Cue.LEFT, Actuation(100, .1)))

    async def test_real_worker_sends_two_single_pulses_for_fast_grasp(self):
        feedback = GraspFeedback(self.c)
        settings = Actuation(100, .1, 100)
        self.assertTrue(feedback.request_prompt(settings)[0])
        await wait_until(lambda: feedback.prompt.status == 'sent')
        self.assertEqual(feedback.poll_prompt(), (True, False))
        feedback.update(True)
        async def finish():
            while feedback.confirmation is None or feedback.confirmation.status != 'completed':
                feedback.update(True)
                await asyncio.sleep(.01)
        await asyncio.wait_for(finish(), 2)
        packets = [p for p, _ in self.client.writes]
        vibration = command_packet(Cue.GRASP, settings)
        zero = bytes([255, 16, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(packets, [vibration, zero, STOP_PACKET, vibration, zero, STOP_PACKET])
        sent = [e for e in self.c.drain_events() if e['event'] == 'command_sent']
        self.assertEqual([e['role'] for e in sent], ['grab_prompt', 'grasp_confirmation'])
        self.assertTrue(all(e['commanded_voltage_mv'] == 100 and e['commanded_duration_s'] == .1
                            and e['frequency_hz'] == 100 for e in sent))
        self.assertLess(sent[0]['mono'], sent[1]['mono'])

    async def test_hard_stop_attempted_if_soft_stop_fails(self):
        original = self.client.write_gatt_char
        async def failing(uuid, packet, response):
            if packet != STOP_PACKET:
                raise RuntimeError('write failed')
            await original(uuid, packet, response)
        with patch.object(self.client, 'write_gatt_char', failing):
            with self.assertRaises(RuntimeError):
                await self.c._stop_sequence(self.client, self.client.char)
        self.assertEqual(self.client.writes[-1][0], STOP_PACKET)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_only_characteristic_never_written(self):
        client = FakeClient(['read'])
        async def scan(*args, **kwargs):
            return object()
        fake_bleak = SimpleNamespace(BleakScanner=SimpleNamespace(find_device_by_address=scan),
                                     BleakClient=lambda *a, **k: client)
        c = ManualBleController(True, 'test', 'test')
        c.running = True
        with patch.dict(sys.modules, {'bleak': fake_bleak}):
            task = asyncio.create_task(c._run())
            await wait_until(lambda: 'not writable' in c.status_msg)
            c.running = False
            await asyncio.wait_for(task, 2)
        self.assertEqual(client.writes, [])
        self.assertFalse(c.is_connected)

    async def test_disconnect_discards_pending_command(self):
        client = FakeClient()
        async def scan(*args, **kwargs):
            return object()
        fake_bleak = SimpleNamespace(BleakScanner=SimpleNamespace(find_device_by_address=scan),
                                     BleakClient=lambda *a, **k: client)
        c = ManualBleController(True, 'test', 'test')
        c.running = True
        with patch.dict(sys.modules, {'bleak': fake_bleak}):
            task = asyncio.create_task(c._run())
            await wait_until(lambda: c.is_connected)
            c.submit(Cue.LEFT, Actuation(100, 1))
            client.is_connected = False
            await wait_until(lambda: not c.is_connected)
            c.running = False
            await asyncio.wait_for(task, 2)
        self.assertIsNone(c._pending)
        self.assertEqual(len(client.writes), 2)  # Only the initial stop handshake.
        self.assertIn('unknown', c.status_msg)

    async def test_shutdown_stops_before_disconnect(self):
        client = FakeClient()
        async def scan(*args, **kwargs):
            return object()
        fake_bleak = SimpleNamespace(BleakScanner=SimpleNamespace(find_device_by_address=scan),
                                     BleakClient=lambda *a, **k: client)
        c = ManualBleController(True, 'test', 'test')
        c.running = True
        with patch.dict(sys.modules, {'bleak': fake_bleak}):
            task = asyncio.create_task(c._run())
            await wait_until(lambda: c.is_connected)
            c.submit(Cue.FORWARD, Actuation(100, 5))
            await wait_until(lambda: len(client.writes) == 3)
            c.stop_output()
            c.running = False
            await asyncio.wait_for(task, 2)
        self.assertEqual(client.writes[-1][0], STOP_PACKET)
        self.assertFalse(client.is_connected)
        self.assertFalse(c.is_connected)

    async def test_reconnect_does_not_replay_pending_command(self):
        client = FakeClient()
        c = ManualBleController(True, 'test', 'test')
        c.running = c.is_connected = True
        c.submit(Cue.RIGHT, Actuation(100, .1))
        task = asyncio.create_task(c._serve(client, client.char))
        await wait_until(lambda: c.status_msg == 'Ready / output stopped')
        c.running = False
        await task
        self.assertEqual(len(client.writes), 2)
        self.assertIsNone(c._pending)


if __name__ == '__main__':
    unittest.main()

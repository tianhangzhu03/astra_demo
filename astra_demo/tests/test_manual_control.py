import asyncio
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from astra_demo.manual_control import (
    Actuation, Cue, DIRECTIONS, ManualBleController, STOP_PACKET, command_packet,
)


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
    def test_all_six_cues_and_endianness(self):
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

    def test_no_visual_grab_dispatch(self):
        source = Path(__file__).parents[1] / 'main.py'
        tree = ast.parse(source.read_text())
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == 'ble']
        self.assertEqual(sorted(calls), ['start', 'stop'])


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

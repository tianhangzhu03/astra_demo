"""Single-writer, finite-duration BLE control using the CEL_GUI wire protocol."""

import asyncio
from dataclasses import dataclass
from enum import Enum
import math
import threading
import time


class Cue(str, Enum):
    FORWARD = "Forward"
    BACK = "Back"
    LEFT = "Left"
    RIGHT = "Right"
    NORMAL = "Normal force"
    GRASP = "Vibration"


DIRECTIONS = {Cue.FORWARD: 1, Cue.BACK: 2, Cue.LEFT: 3,
              Cue.RIGHT: 4, Cue.NORMAL: 0, Cue.GRASP: 0}
STOP_PACKET = bytes([0xFF, 0x06, 0, 0, 0, 0, 0, 0, 0])


@dataclass(frozen=True)
class Actuation:
    volts_mv: int = 0
    duration_s: float = 0.2
    frequency_hz: int = 100

    def validate(self):
        if not isinstance(self.volts_mv, int) or not 0 <= self.volts_mv <= 5000:
            raise ValueError("Voltage must be 0..5000 mV")
        if not math.isfinite(self.duration_s) or not 0.1 <= self.duration_s <= 5.0:
            raise ValueError("Duration must be 0.1..5.0 s (no infinite output)")
        if not isinstance(self.frequency_hz, int) or not 10 <= self.frequency_hz <= 200:
            raise ValueError("Vibration frequency must be 10..200 Hz")


def command_packet(cue, settings):
    settings.validate()
    direction = DIRECTIONS[cue]
    vibration = cue == Cue.GRASP
    frequency = settings.frequency_hz if vibration else 0
    v = settings.volts_mv
    return bytes([0xFF, 0x10, int(vibration), direction,
                  v & 255, v >> 8, frequency & 255, frequency >> 8, 1])


class ManualBleController:
    def __init__(self, enabled, address, uuid):
        self.enabled, self.address, self.uuid = enabled, address, uuid
        self.status_msg = "Initializing" if enabled else "BLE disabled"
        self.is_connected = False
        self.running = False
        self._lock = threading.Lock()
        self._pending = None
        self._stop_requested = False
        self._busy = False
        self._thread = None

    def start(self):
        if self.enabled and not self.running:
            self.running = True
            self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)
            self._thread.start()

    def submit(self, cue, settings):
        try:
            packet = command_packet(cue, settings)
        except (ValueError, KeyError) as exc:
            return False, str(exc)
        with self._lock:
            if not self.running or not self.is_connected:
                return False, "Not connected; command discarded"
            if self._busy or self._stop_requested:
                return False, "Busy; wait or stop output first"
            if settings.volts_mv == 0:
                return False, "Set voltage above 0 mV first"
            self._busy = True
            self._pending = (cue, packet, settings.duration_s, time.monotonic())
        return True, f"Requested: {cue.value}"

    def stop_output(self):
        with self._lock:
            self._pending = None
            self._stop_requested = True
        return "Stop requested" if self.is_connected else "Disconnected: stop unconfirmed"

    def stop(self):
        self.stop_output()
        self.running = False
        if self._thread:
            self._thread.join(timeout=4.0)
            if self._thread.is_alive():
                print("[BLE] Shutdown pending; output stop is not confirmed.")

    async def _write(self, client, char, packet):
        await asyncio.wait_for(client.write_gatt_char(
            char.uuid, packet, response="write" in char.properties), timeout=1.0)

    async def _stop_sequence(self, client, char):
        # Match CEL_GUI: zero static output, then the dedicated stop opcode.
        # Always attempt the hard stop even when the zero-output write fails.
        try:
            await self._write(client, char, bytes([255, 16, 0, 0, 0, 0, 0, 0, 1]))
        finally:
            await asyncio.sleep(0.1)
            await self._write(client, char, STOP_PACKET)

    async def _serve(self, client, char):
        await self._stop_sequence(client, char)
        with self._lock:
            self._pending = None
            self._busy = False
            self._stop_requested = False
            self.is_connected = True
        self.status_msg = "Ready / output stopped"
        deadline = None
        while self.running and client.is_connected:
            with self._lock:
                stop = self._stop_requested
                self._stop_requested = False
                command, self._pending = self._pending, None
            if stop or (deadline is not None and time.monotonic() >= deadline):
                await self._stop_sequence(client, char)
                deadline = None
                with self._lock:
                    self._busy = False
                self.status_msg = "Output stopped (commands sent)"
            elif command:
                cue, packet, duration, submitted = command
                if time.monotonic() - submitted > 0.5:
                    with self._lock:
                        self._busy = False
                    self.status_msg = "Expired command discarded"
                else:
                    await self._write(client, char, packet)
                    deadline = time.monotonic() + duration
                    self.status_msg = f"Sent: {cue.value} / {duration:.1f} s"
            await asyncio.sleep(0.01)

    async def _run(self):
        # Lazy import keeps offline previews and protocol tests independent of BLE.
        from bleak import BleakClient, BleakScanner

        while self.running:
            client, char = None, None
            try:
                self.status_msg = "Scanning"
                device = await BleakScanner.find_device_by_address(self.address, timeout=3.0)
                if not device:
                    raise RuntimeError("Device not found")
                if not self.running:
                    break
                self.status_msg = "Connecting"
                client = BleakClient(device, timeout=5.0)
                await asyncio.wait_for(client.connect(), timeout=6.0)
                candidate = next((c for s in client.services for c in s.characteristics
                                  if c.uuid.lower() == self.uuid.lower()), None)
                if candidate is None or not ({"write", "write-without-response"} & set(candidate.properties)):
                    raise RuntimeError("Configured UUID missing or not writable")
                char = candidate
                await self._serve(client, char)
                self.status_msg = "Disconnected / output state unknown"
            except Exception as exc:
                self.status_msg = f"BLE error: {exc} / output unconfirmed"
            finally:
                with self._lock:
                    self.is_connected = False
                    self._pending = None
                    self._busy = False
                if client and client.is_connected:
                    try:
                        if char:
                            await self._stop_sequence(client, char)
                    except Exception:
                        self.status_msg = "Stop failed / output state unknown"
                    finally:
                        try:
                            await asyncio.wait_for(client.disconnect(), timeout=1.0)
                        except Exception:
                            pass
            # Short sleeps allow shutdown while waiting to reconnect.
            for _ in range(50):
                if not self.running:
                    break
                await asyncio.sleep(0.02)

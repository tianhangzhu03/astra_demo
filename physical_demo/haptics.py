# Frozen CEL_GUI transport snapshot from the working demo, 2026-11-13.
# Intentionally independent: this package never imports or changes astra_demo.
# Manual object annotations are handled elsewhere and never send confirmation pulses.

"""Single-writer, finite-duration BLE control using the CEL_GUI wire protocol."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import math
import threading
import time
from itertools import count
from queue import SimpleQueue, Empty


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


@dataclass
class CommandReceipt:
    # 'sent' confirms the BLE write, not the actuator's physical response.
    status: str = 'pending'
    sent_at: float | None = None
    finished_at: float | None = None
    connection_id: int = 0
    cancelled: bool = False
    role: str = 'operator'
    log_data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class HapticTrigger:
    command_id: str
    label: str
    sent_at: float
    volts_mv: int
    duration_s: float
    simulated: bool = False


@dataclass(frozen=True)
class HapticDisplay:
    state: str = 'idle'
    trigger: HapticTrigger | None = None
    last_trigger: HapticTrigger | None = None
    simulated: bool = False


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
        self.connection_id = 0
        self._active_receipt = None
        self.trial_id = ''
        self.events = SimpleQueue()
        self._command_ids = count(1)
        self._last_trigger = None
        self._simulated_until = 0.0

    def _remember_trigger(self, receipt, *, simulated=False):
        # Caller holds _lock. Snapshot settings only after a successful write (or simulation).
        data = receipt.log_data
        label = data['cue'].upper()
        if data['cue'] == Cue.GRASP.value:
            label = {'grab_prompt': 'GRASP CUE',
                     'grasp_confirmation': 'GRASP CONFIRMATION'}.get(receipt.role, 'VIBRATION')
        elif data['cue'] == Cue.NORMAL.value:
            label = {'normal_lower': 'LOWER CUE', 'normal_lift': 'LIFT CUE'}.get(receipt.role, 'NORMAL FORCE')
        self._last_trigger = HapticTrigger(data['command_id'], label, receipt.sent_at,
            data['commanded_voltage_mv'], data['commanded_duration_s'], simulated)

    def feedback_snapshot(self, now=None):
        """Read-only UI state; does not consume the experiment's command event queue.

        Active means a BLE write succeeded within its commanded duration, never a
        measurement of the actuator. A disconnected link has unknown output state.
        """
        with self._lock:
            now = time.monotonic() if now is None else now
            last = self._last_trigger
            recent = last if last and 0 <= now - last.sent_at < 2.0 else None
            if not self.enabled:
                active = last if last and last.sent_at <= now < self._simulated_until else None
                return HapticDisplay('active' if active else 'idle', active, recent, True)
            if not self.running or not self.is_connected:
                return HapticDisplay('disconnected')
            receipt = self._active_receipt
            if self._stop_requested or (receipt and receipt.cancelled):
                return HapticDisplay('stopping', last_trigger=recent)
            if receipt and receipt.status == 'sent' and last and last.command_id == receipt.log_data['command_id']:
                if now < last.sent_at + last.duration_s:
                    return HapticDisplay('active', last, recent)
                return HapticDisplay('stopping', last_trigger=recent)
            if self._pending is not None or (receipt and receipt.status == 'pending'):
                return HapticDisplay('pending', last_trigger=recent)
            return HapticDisplay(last_trigger=recent)

    def _emit(self, event, receipt=None, **data):
        # No filesystem I/O on the BLE writer thread.
        self.events.put({'event': event, 'mono': time.monotonic(),
                         'trial_id': self.trial_id, **(receipt.log_data if receipt else {}), **data})

    def drain_events(self):
        while True:
            try:
                yield self.events.get_nowait()
            except Empty:
                return

    def _request_record(self, cue, settings, receipt):
        receipt.log_data = dict(command_id=f'C{next(self._command_ids):05d}', trial_id=self.trial_id,
            cue=cue.value, role=receipt.role, commanded_voltage_mv=settings.volts_mv,
            commanded_duration_s=settings.duration_s,
            frequency_hz=settings.frequency_hz if cue == Cue.GRASP else 0,
            connection_id=self.connection_id)
        self._emit('command_requested', receipt)

    def simulate(self, cue, settings, *, receipt=None):
        if self.enabled:
            raise ValueError('Simulation is only available with BLE disabled')
        settings.validate()
        receipt = receipt or CommandReceipt()
        self._request_record(cue, settings, receipt)
        with self._lock:
            receipt.status = 'completed'
            receipt.sent_at = receipt.finished_at = time.monotonic()
            self._remember_trigger(receipt, simulated=True)
            self._simulated_until = receipt.sent_at + settings.duration_s
            self._emit('command_simulated', receipt, reason='Vision only; no physical output')
        return True, f'Simulated {cue.value} (no output)'

    def start(self):
        if self.enabled and not self.running:
            self.running = True
            self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)
            self._thread.start()

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def submit(self, cue, settings, *, receipt=None):
        receipt = receipt or CommandReceipt()
        self._request_record(cue, settings, receipt)
        def reject(reason):
            receipt.status = 'rejected'
            self._emit('command_rejected', receipt, reason=reason)
            return False, reason
        try:
            packet = command_packet(cue, settings)
        except (ValueError, KeyError) as exc:
            return reject(str(exc))
        with self._lock:
            if not self.enabled or not self.running or not self.is_connected:
                return reject("Not connected; command discarded")
            if self._busy or self._stop_requested:
                return reject("Busy; wait or stop output first")
            if settings.volts_mv == 0:
                return reject("Set voltage above 0 mV first")
            self._busy = True
            if receipt is not None:
                receipt.connection_id = self.connection_id
            self._pending = (cue, packet, settings.duration_s, time.monotonic(), receipt)
        return True, f"Requested: {cue.value}"

    def cancel(self, receipt):
        """Cancel this receipt without stopping an unrelated operator command."""
        if receipt is None:
            return
        with self._lock:
            self._emit('command_cancel_requested', receipt)
            if self._last_trigger and self._last_trigger.command_id == receipt.log_data.get('command_id'):
                self._simulated_until = 0.0
            receipt.cancelled = True
            receipt.status = 'cancelled'
            if self._pending is not None and self._pending[-1] is receipt:
                self._emit('command_cancelled', receipt)
                self._pending = None
                self._busy = False
            if self._active_receipt is receipt:
                self._stop_requested = True

    def _discard_pending(self):
        # Caller holds _lock.
        if self._pending is not None and self._pending[-1] is not None:
            self._emit('command_cancelled', self._pending[-1])
            self._pending[-1].status = 'cancelled'
            self._pending[-1].cancelled = True
        self._pending = None

    def stop_output(self):
        self._emit('output_stop_requested')
        with self._lock:
            self._simulated_until = 0.0
            self._discard_pending()
            if self._active_receipt is not None:
                self._active_receipt.cancelled = True
                self._active_receipt.status = 'cancelled'
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
            self._emit('zero_voltage_sent')
        finally:
            await asyncio.sleep(0.1)
            await self._write(client, char, STOP_PACKET)
            self._emit('stop_packet_sent')

    async def _serve(self, client, char):
        await self._stop_sequence(client, char)
        with self._lock:
            self._discard_pending()
            self._active_receipt = None
            self._last_trigger = None
            self._busy = False
            self._stop_requested = False
            self.connection_id += 1
            self.is_connected = True
        self.status_msg = "Ready / output stopped"
        self._emit('ble_connected', connection_id=self.connection_id)
        deadline = None
        while self.running and client.is_connected:
            with self._lock:
                stop = self._stop_requested
                self._stop_requested = False
                command, self._pending = self._pending, None
                if command:
                    self._active_receipt = command[-1]
            if stop or (deadline is not None and time.monotonic() >= deadline):
                await self._stop_sequence(client, char)
                deadline = None
                with self._lock:
                    if self._active_receipt is not None:
                        receipt = self._active_receipt
                        receipt.finished_at = time.monotonic()
                        receipt.status = 'cancelled' if receipt.cancelled or stop else 'completed'
                        self._emit('command_' + receipt.status, receipt)
                    self._active_receipt = None
                    self._busy = False
                self.status_msg = "Output stopped (commands sent)"
            elif command:
                cue, packet, duration, submitted, receipt = command
                if (receipt is not None and receipt.cancelled) or time.monotonic() - submitted > 0.5:
                    with self._lock:
                        if receipt is not None:
                            receipt.status = 'cancelled' if receipt.cancelled else 'expired'
                            self._emit('command_' + receipt.status, receipt)
                        self._active_receipt = None
                        self._busy = False
                    self.status_msg = "Expired command discarded"
                else:
                    await self._write(client, char, packet)
                    with self._lock:
                        receipt.sent_at = time.monotonic()
                        if not receipt.cancelled:
                            receipt.status = 'sent'
                        self._remember_trigger(receipt)
                        self._emit('command_sent', receipt)
                        deadline = receipt.sent_at + duration
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
                self._emit('ble_error', reason=str(exc))
            finally:
                with self._lock:
                    self.is_connected = False
                    self._discard_pending()
                    if self._active_receipt is not None:
                        self._active_receipt.status = 'failed'
                        self._emit('command_failed', self._active_receipt, reason=self.status_msg)
                        self._active_receipt = None
                    self._busy = False
                self._emit('ble_disconnected', reason=self.status_msg)
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

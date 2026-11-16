"""Single-writer, finite-duration BLE control using the CEL_GUI wire protocol."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json
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
    TORSION = "Torsion"


TORSION_DIRECTION = 5
TORSION_CCW_DIRECTION = 6
VERSION_UUID = '030062c4-b99e-4141-9439-c4f9db977899'
EXPECTED_FIRMWARE = 'v0.2.1'
DIRECTIONS = {Cue.FORWARD: 1, Cue.BACK: 2, Cue.LEFT: 3,
              Cue.RIGHT: 4, Cue.NORMAL: 0, Cue.GRASP: 0, Cue.TORSION: TORSION_DIRECTION}
STOP_PACKET = bytes([0xFF, 0x06, 0, 0, 0, 0, 0, 0, 0])
V21_STOP_PACKET = bytes([0xFF, 0x06])
# Colleague's table, ordered ch0..ch3. Direction is an enum, not a bitmask:
# e.g. Normal uses direction=0 although its channel pattern is [1,1,1,1].
CHANNEL_STATES = {
    Cue.FORWARD: (1, 1, 0, 0), Cue.BACK: (0, 0, 1, 1),
    Cue.LEFT: (1, 0, 1, 0), Cue.RIGHT: (0, 1, 0, 1),
    Cue.NORMAL: (1, 1, 1, 1), Cue.TORSION: (1, 0, 0, 1),
}
TORSION_CHANNELS = CHANNEL_STATES[Cue.TORSION]
TORSION_FIRMWARE_ERROR = 'Torsion direction=5 needs updated device firmware. Enable --torsion-firmware-ready after installing it; no output sent.'
REPEATABLE_CUES = frozenset((Cue.FORWARD, Cue.BACK, Cue.LEFT, Cue.RIGHT, Cue.NORMAL))
REPEAT_INTERVAL_S = 1.0
REPEAT_MAX_PULSE_S = .8  # Leave time for the existing 100 ms zero/stop sequence.
BLE_SCAN_TIMEOUT_S = 30.0
BLE_CONNECT_TIMEOUT_S = 60.0


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


@dataclass
class HeldCue:
    cue: Cue
    settings: Actuation
    role: str
    guard: object
    repeat_id: str
    connection_id: int
    trial_id: str
    receipt: CommandReceipt | None = None
    next_at: float | None = None
    index: int = 0


def direction_code(cue, role='operator'):
    return TORSION_CCW_DIRECTION if cue == Cue.TORSION and role == 'torsion_ccw' else DIRECTIONS[cue]


def command_packet(cue, settings, role='operator'):
    settings.validate()
    direction = direction_code(cue, role)
    vibration = cue == Cue.GRASP
    frequency = settings.frequency_hz if vibration else 0
    v = settings.volts_mv
    return bytes([0xFF, 0x10, int(vibration), direction,
                  v & 255, v >> 8, frequency & 255, frequency >> 8, 1])


class ManualBleController:
    def __init__(self, enabled, address, uuid, *, torsion_firmware_ready=False, protocol='legacy'):
        self.enabled, self.address, self.uuid = enabled, address, uuid
        if protocol not in ('legacy', EXPECTED_FIRMWARE):
            raise ValueError('Unknown CEL protocol')
        self.protocol = protocol
        self.firmware_version = None
        self.torsion_firmware_ready = bool(torsion_firmware_ready)
        self.status_msg = "Initializing" if enabled else "BLE disabled"
        self.is_connected = False
        self.running = False
        self._lock = threading.Lock()
        self._pending = None
        self._stop_requested = False
        self._busy = False
        self._thread = None
        self._loop = None
        self._worker_task = None
        self.connection_id = 0
        self._active_receipt = None
        self.trial_id = ''
        self.events = SimpleQueue()
        self._command_ids = count(1)
        self._last_trigger = None
        self._simulated_until = 0.0
        self._repeat = None
        self._repeat_ids = count(1)
        self._last_repeat_sent_at = None

    def _remember_trigger(self, receipt, *, simulated=False):
        # Caller holds _lock. Snapshot settings only after a successful write (or simulation).
        data = receipt.log_data
        if data.get('repeat_id'):
            self._last_repeat_sent_at = receipt.sent_at
        label = data['cue'].upper()
        if data['cue'] == Cue.GRASP.value:
            label = {'grab_prompt': 'GRASP CUE',
                     'release_prompt': 'DROP CUE',
                     'grasp_confirmation': 'GRASP CONFIRMATION'}.get(receipt.role, 'VIBRATION')
        elif receipt.role in ('normal_up', 'normal_down'):
            label = 'NORMAL +Z' if receipt.role == 'normal_up' else 'NORMAL -Z'
        elif data['cue'] == Cue.TORSION.value:
            label = {'torsion_cw': 'CW FLIP DOWN CUE', 'torsion_ccw': 'CCW FLIP BACK CUE'}.get(receipt.role, 'TORSION CUE')
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
        receipt.log_data = dict(receipt.log_data, command_id=f'C{next(self._command_ids):05d}', trial_id=self.trial_id,
            cue=cue.value, role=receipt.role, commanded_voltage_mv=settings.volts_mv,
            commanded_duration_s=settings.duration_s,
            frequency_hz=settings.frequency_hz if cue == Cue.GRASP else 0,
            connection_id=self.connection_id, direction_code=direction_code(cue, receipt.role))
        if cue in CHANNEL_STATES and receipt.role != 'torsion_ccw':
            receipt.log_data['commanded_channels_json'] = json.dumps(CHANNEL_STATES[cue])
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
            self._thread = threading.Thread(target=lambda: asyncio.run(self._worker_main()), daemon=True)
            self._thread.start()

    async def _worker_main(self):
        self._loop = asyncio.get_running_loop()
        self._worker_task = asyncio.current_task()
        try:
            await self._run()
        except asyncio.CancelledError:
            if self.running:
                raise
        finally:
            self._worker_task = self._loop = None

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def submit(self, cue, settings, *, receipt=None, replace_output=False):
        receipt = receipt or CommandReceipt()
        self._request_record(cue, settings, receipt)
        def reject(reason):
            receipt.status = 'rejected'
            self._emit('command_rejected', receipt, reason=reason)
            return False, reason
        try:
            packet = command_packet(cue, settings, receipt.role)
            if self.protocol == EXPECTED_FIRMWARE and settings.duration_s > 1.0:
                raise ValueError('v0.2.1 supports at most approximately 1 s per run')
        except (ValueError, KeyError) as exc:
            return reject(str(exc))
        if cue == Cue.TORSION and not self.torsion_firmware_ready:
            return reject(TORSION_FIRMWARE_ERROR)
        with self._lock:
            if not self.enabled or not self.running or not self.is_connected:
                return reject("Not connected; command discarded")
            if (self._busy or self._stop_requested) and not replace_output:
                return reject("Busy; wait or stop output first")
            if settings.volts_mv == 0:
                return reject("Set voltage above 0 mV first")
            if replace_output:
                self._discard_pending()
                if self._active_receipt is not None:
                    self._active_receipt.cancelled = True
                    self._stop_requested = True
            self._busy = True
            if receipt is not None:
                receipt.connection_id = self.connection_id
            self._pending = (cue, packet, settings.duration_s, time.monotonic(), receipt)
        return True, f"Requested: {cue.value}"

    @property
    def repeat_action(self):
        with self._lock:
            repeat = self._repeat
            return (repeat.cue, repeat.role) if repeat else None

    def start_repeat(self, cue, settings, *, role='operator', guard):
        """One selected direction, finite pulses scheduled by the writer, not OS key repeat."""
        def reject(reason):
            receipt = CommandReceipt(role=role)
            self._request_record(cue, settings, receipt)
            receipt.status = 'rejected'
            self._emit('command_rejected', receipt, reason=reason)
            return False, reason
        try:
            settings.validate()
        except ValueError as exc:
            return reject(str(exc))
        if cue not in REPEATABLE_CUES or settings.duration_s > REPEAT_MAX_PULSE_S:
            return reject('Repeat supports Shear/Normal pulses of 0.1..0.8 s')
        with self._lock:
            if self.enabled and (not self.running or not self.is_connected):
                return reject('Not connected; hold discarded')
            if self.enabled and settings.volts_mv == 0:
                return reject('Set voltage above 0 mV first')
            if self.enabled and self._repeat is None and (self._busy or self._stop_requested):
                return reject('Busy; wait for output to stop')
        if self.repeat_action == (cue, role):
            return True, 'Direction already selected; X stops'
        self.stop_repeat('Direction changed')
        repeat = HeldCue(cue, settings, role, guard, f'H{next(self._repeat_ids):05d}', self.connection_id, self.trial_id)
        with self._lock:
            if self._last_repeat_sent_at is not None:
                repeat.next_at = max(time.monotonic(), self._last_repeat_sent_at+REPEAT_INTERVAL_S)
            self._repeat = repeat
        self._emit('repeat_started', cue=cue.value, role=role, repeat_id=repeat.repeat_id,
                   repeat_interval_s=REPEAT_INTERVAL_S, reason='Selected direction; Stop/focus loss cancels')
        if not self.enabled:
            self.poll_repeat()
        return True, f'{cue.value}: one pulse per second; select a direction or X to stop'

    def stop_repeat(self, reason='Input released'):
        with self._lock:
            repeat, self._repeat = self._repeat, None
            if repeat and not self.enabled:
                self._simulated_until = 0.
        if repeat is not None:
            if repeat.receipt and repeat.receipt.status in ('pending', 'sent'):
                self.cancel(repeat.receipt)
            self._emit('repeat_stopped', cue=repeat.cue.value, role=repeat.role,
                       repeat_id=repeat.repeat_id, reason=reason)

    def poll_repeat(self, now=None):
        """Called on the sole BLE writer; no-BLE previews call it on the UI tick."""
        with self._lock:
            repeat = self._repeat
        if repeat is None:
            return
        now = time.monotonic() if now is None else now
        try:
            keep = repeat.guard()
        except Exception:
            keep = False
        if (not keep or repeat.trial_id != self.trial_id or
            (self.enabled and (not self.is_connected or repeat.connection_id != self.connection_id))):
            self.stop_repeat('Focus lost, stale UI, trial changed or connection lost')
            return
        receipt = repeat.receipt
        if receipt and receipt.status in ('rejected', 'failed', 'cancelled', 'expired'):
            self.stop_repeat('Pulse did not complete')
            return
        if receipt and repeat.next_at is None:
            if receipt.sent_at is None:
                return
            repeat.next_at = receipt.sent_at + REPEAT_INTERVAL_S
        if repeat.next_at is not None and now < repeat.next_at:
            return
        # Switching directions waits for the previous zero/stop sequence, without a queue.
        with self._lock:
            busy = self.enabled and (self._busy or self._stop_requested)
        if busy and repeat.next_at is None:
            return
        if repeat.next_at is not None:
            late = now-repeat.next_at
            if busy or late > .1:
                missed = int(late//REPEAT_INTERVAL_S)+1
                repeat.next_at += missed*REPEAT_INTERVAL_S
                self._emit('repeat_slot_skipped', repeat_id=repeat.repeat_id,
                           reason='Busy or late; no catch-up burst', skipped=missed)
                return
            repeat.next_at += REPEAT_INTERVAL_S
        with self._lock:
            if self._repeat is not repeat:
                return
        receipt = CommandReceipt(role=repeat.role, log_data=dict(repeat_id=repeat.repeat_id,
            repeat_index=repeat.index, repeat_interval_s=REPEAT_INTERVAL_S))
        repeat.receipt = receipt
        repeat.index += 1
        send = self.submit if self.enabled else self.simulate
        accepted, _ = send(repeat.cue, repeat.settings, receipt=receipt)
        # A UI stop can race the BLE thread between its guard and submission.
        with self._lock:
            current = self._repeat is repeat
        if not current and receipt.status in ('pending', 'sent'):
            self.cancel(receipt)
        elif not accepted:
            self.stop_repeat('Pulse rejected')

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
        self.stop_repeat('Stop requested')
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
        # Interrupt scan/connect waits so _run's finally releases WinRT before exit.
        loop, task = self._loop, self._worker_task
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # The worker finished and closed its loop during shutdown.
        if self._thread:
            self._thread.join(timeout=4.0)
            if self._thread.is_alive():
                print("[BLE] Shutdown pending; output stop is not confirmed.")

    async def _write(self, client, char, packet):
        try:
            await asyncio.wait_for(client.write_gatt_char(
                char.uuid, packet, response="write" in char.properties), timeout=1.0)
        except TimeoutError as exc:
            action = 'Stop' if packet in (STOP_PACKET, V21_STOP_PACKET) else 'Zero voltage' if packet[4:6] == b'\x00\x00' else 'Actuation'
            raise TimeoutError(f'{action} BLE write: no acknowledgement within the 1 s limit') from exc

    async def _stop_sequence(self, client, char):
        if self.protocol == EXPECTED_FIRMWARE:
            await self._write(client, char, V21_STOP_PACKET)
            self._emit('stop_packet_sent', packet_hex=V21_STOP_PACKET.hex())
            await asyncio.sleep(.03)
            return
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
            self.poll_repeat()
            with self._lock:
                stop = self._stop_requested
                self._stop_requested = False
                command = None
                if not stop and deadline is None:
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
                    self._busy = self._pending is not None
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
                        self._emit('command_sent', receipt, packet_hex=packet.hex())
                        deadline = receipt.sent_at + duration
                    self.status_msg = f"Sent: {cue.value} / {duration:.1f} s"
            await asyncio.sleep(0.01)

    async def _run(self):
        # Lazy import keeps offline previews and protocol tests independent of BLE.
        from bleak import BleakClient, BleakScanner

        while self.running:
            client, char = None, None
            try:
                self.status_msg = f'Scanning {self.address} ({BLE_SCAN_TIMEOUT_S:g} s)'
                device = await BleakScanner.find_device_by_address(self.address.strip(), timeout=BLE_SCAN_TIMEOUT_S)
                if not device:
                    raise RuntimeError(f'No advertisement from {self.address} in {BLE_SCAN_TIMEOUT_S:g} s')
                if not self.running:
                    break
                device_name = getattr(device, 'name', None)
                self._emit('ble_device_found', address=self.address, name=device_name)
                self.status_msg = f'Connecting {device_name or self.address} ({BLE_CONNECT_TIMEOUT_S:g} s)'
                client = BleakClient(device, timeout=BLE_CONNECT_TIMEOUT_S, winrt={'use_cached_services': False})
                try:
                    await asyncio.wait_for(client.connect(), timeout=BLE_CONNECT_TIMEOUT_S + 5.0)
                except TimeoutError as exc:
                    raise TimeoutError(f'GATT connection to {self.address} timed out ({BLE_CONNECT_TIMEOUT_S:g} s)') from exc
                candidate = next((c for s in client.services for c in s.characteristics
                                  if c.uuid.lower() == self.uuid.lower()), None)
                if candidate is None or not ({"write", "write-without-response"} & set(candidate.properties)):
                    raise RuntimeError("Configured UUID missing or not writable")
                if self.protocol == EXPECTED_FIRMWARE:
                    if 'write' not in candidate.properties:
                        raise RuntimeError('v0.2.1 requires acknowledged BLE writes')
                    raw = await asyncio.wait_for(client.read_gatt_char(VERSION_UUID, use_cached=False), 3.)
                    version = bytes(raw).decode('utf-8', errors='replace').strip('\x00').strip()
                    if not (device_name or '').upper().startswith('CEL') or version != EXPECTED_FIRMWARE:
                        raise RuntimeError(f'Expected CEL {EXPECTED_FIRMWARE}; received {device_name} / {version}. No output sent.')
                    self.firmware_version = version
                    self.torsion_firmware_ready = True
                    self._emit('firmware_verified', version=version, name=device_name)
                char = candidate
                await self._serve(client, char)
                self.status_msg = "Disconnected / output state unknown"
            except (Exception, asyncio.CancelledError) as exc:
                # WinRT can cancel an individual GATT operation without cancelling
                # this worker. Recover that failure, but preserve explicit task cancellation.
                if isinstance(exc, asyncio.CancelledError) and asyncio.current_task().cancelling():
                    raise
                reason = f'{type(exc).__name__}: {exc}'.rstrip(': ')
                self.status_msg = f"BLE error: {reason} / output unconfirmed"
                self._emit('ble_error', reason=reason)
            finally:
                self.stop_repeat('BLE disconnected')
                if self.protocol == EXPECTED_FIRMWARE:
                    self.torsion_firmware_ready = False
                with self._lock:
                    self.is_connected = False
                    self._discard_pending()
                    if self._active_receipt is not None:
                        self._active_receipt.status = 'failed'
                        self._emit('command_failed', self._active_receipt, reason=self.status_msg)
                        self._active_receipt = None
                    self._busy = False
                self._emit('ble_disconnected', reason=self.status_msg)
                if client:
                    try:
                        if char and client.is_connected:
                            await self._stop_sequence(client, char)
                    except (Exception, asyncio.CancelledError) as exc:
                        if isinstance(exc, asyncio.CancelledError) and asyncio.current_task().cancelling():
                            raise
                        self.status_msg = "Stop failed / output state unknown"
                        self._emit('ble_stop_failed', reason=f'{type(exc).__name__}: {exc}'.rstrip(': '))
                    finally:
                        try:
                            await asyncio.wait_for(client.disconnect(), timeout=1.0)
                        except (Exception, asyncio.CancelledError) as exc:
                            if isinstance(exc, asyncio.CancelledError) and asyncio.current_task().cancelling():
                                raise
                            self._emit('ble_disconnect_failed', reason=f'{type(exc).__name__}: {exc}'.rstrip(': '))
            # Short sleeps allow shutdown while waiting to reconnect.
            for _ in range(50):
                if not self.running:
                    break
                await asyncio.sleep(0.02)

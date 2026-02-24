from __future__ import annotations

import asyncio
import platform
import threading
from dataclasses import dataclass

from bleak import BleakClient, BleakScanner


@dataclass
class BleConfig:
    enabled: bool
    mac_address: str
    uuid: str
    fixed_volts: int = 2000
    fixed_freq: int = 100
    pulse_ms: int = 2000


def create_packet(volts: int, freq: int, start: bool = True) -> bytes:
    header, opcode = 0xFF, 0x10
    mode_val = 1
    dir_val = 0
    volts = int(max(0, min(5000, volts)))
    freq = int(max(0, min(100, freq)))
    v_low, v_high = volts & 0xFF, (volts >> 8) & 0xFF
    f_low, f_high = freq & 0xFF, (freq >> 8) & 0xFF
    start_val = 1 if start else 0
    return bytes([header, opcode, mode_val, dir_val, v_low, v_high, f_low, f_high, start_val])


class BleController:
    def __init__(self, config: BleConfig):
        self._cfg = config

        self.running = False
        self.status_msg = "BLE disabled" if not config.enabled else "Initializing..."
        self.is_connected = False

        self.trigger_on = False
        self.target_key = 0
        self.pulse_freq = config.fixed_freq

        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._cfg.enabled:
            return
        self.running = True
        self._thread = threading.Thread(target=self._thread_entry, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def set_target(self, trigger_on: bool, key: int, freq_hz: int | None = None) -> None:
        self.trigger_on = trigger_on
        self.target_key = key
        if freq_hz is not None:
            self.pulse_freq = int(freq_hz)

    def _thread_entry(self) -> None:
        if platform.system() == "Windows":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(self._ble_task())

    async def _ble_task(self) -> None:
        while self.running:
            client = None
            try:
                self.status_msg = "Scanning..."
                self.is_connected = False

                device = await BleakScanner.find_device_by_address(self._cfg.mac_address, timeout=5.0)
                if not device:
                    self.status_msg = "Device Not Found"
                    await asyncio.sleep(2.0)
                    continue

                self.status_msg = "Connecting..."
                client = BleakClient(device)
                await client.connect()

                write_char = None
                for service in client.services:
                    for char in service.characteristics:
                        if char.uuid.lower() == self._cfg.uuid.lower():
                            write_char = char
                            break
                    if write_char:
                        break

                if not write_char:
                    self.status_msg = "UUID Error"
                    await client.disconnect()
                    await asyncio.sleep(2.0)
                    continue

                self.status_msg = "Connected"
                self.is_connected = True
                last_pressed = False
                pulse_active = False
                pulse_end_t = 0.0
                loop_time = asyncio.get_running_loop().time

                while self.running and client.is_connected:
                    cur_trigger = self.trigger_on
                    cur_key = self.target_key
                    cur_freq = self.pulse_freq
                    now_t = loop_time()

                    if cur_trigger and not last_pressed:
                        if cur_key > 0:
                            packet = create_packet(self._cfg.fixed_volts, cur_freq, start=True)
                            await client.write_gatt_char(write_char.uuid, packet, response=True)
                            pulse_active = True
                            pulse_end_t = now_t + max(0.0, self._cfg.pulse_ms / 1000.0)
                        last_pressed = True
                    elif not cur_trigger and last_pressed:
                        last_pressed = False

                    if pulse_active and now_t >= pulse_end_t:
                        packet = create_packet(0, 0, start=True)
                        await client.write_gatt_char(write_char.uuid, packet, response=True)
                        pulse_active = False

                    await asyncio.sleep(0.01)

                self.status_msg = "Connection Lost"
            except Exception as exc:
                self.status_msg = f"BLE Error: {exc.__class__.__name__}"
            finally:
                if client:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
                await asyncio.sleep(1.0)

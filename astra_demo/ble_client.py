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

        self._thread: threading.Thread | None = None

    @staticmethod
    def _char_properties(char) -> set[str]:
        props = getattr(char, "properties", []) or []
        return {str(p).lower() for p in props}

    def _find_write_characteristic(self, client: BleakClient):
        exact = None
        writable_fallback = None
        desired_uuid = self._cfg.uuid.lower().strip()

        for service in client.services:
            for char in service.characteristics:
                props = self._char_properties(char)
                is_writable = ("write" in props) or ("write-without-response" in props)
                if char.uuid.lower() == desired_uuid:
                    exact = char
                    if is_writable:
                        return char, "exact"
                if is_writable and writable_fallback is None:
                    writable_fallback = char

        if writable_fallback is not None:
            return writable_fallback, "fallback"
        if exact is not None:
            return exact, "uuid-not-writable"
        return None, "not-found"

    async def _write_packet(self, client: BleakClient, char, packet: bytes) -> None:
        props = self._char_properties(char)
        supports_write_rsp = "write" in props
        supports_write_no_rsp = "write-without-response" in props

        if supports_write_rsp:
            try:
                await client.write_gatt_char(char.uuid, packet, response=True)
                return
            except Exception:
                if not supports_write_no_rsp:
                    raise

        await client.write_gatt_char(char.uuid, packet, response=False)

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

    def set_target(self, trigger_on: bool, key: int) -> None:
        self.trigger_on = trigger_on
        self.target_key = key

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
                try:
                    await client.get_services()
                except Exception:
                    # Some Bleak versions populate services on connect; ignore if explicit fetch is unsupported.
                    pass

                write_char, select_mode = self._find_write_characteristic(client)
                if not write_char:
                    self.status_msg = "UUID/WRITE Char Error"
                    await client.disconnect()
                    await asyncio.sleep(2.0)
                    continue
                if select_mode == "uuid-not-writable":
                    self.status_msg = "UUID Not Writable"
                    await client.disconnect()
                    await asyncio.sleep(2.0)
                    continue

                if select_mode == "fallback":
                    self.status_msg = f"Connected(Fallback UUID:{write_char.uuid[:8]})"
                else:
                    self.status_msg = "Connected"
                self.is_connected = True
                last_pressed = False

                while self.running and client.is_connected:
                    cur_trigger = self.trigger_on
                    cur_key = self.target_key

                    if cur_trigger and not last_pressed:
                        if cur_key > 0:
                            packet = create_packet(self._cfg.fixed_volts, self._cfg.fixed_freq, start=True)
                            await self._write_packet(client, write_char, packet)
                        last_pressed = True
                    elif not cur_trigger and last_pressed:
                        packet = create_packet(0, 0, start=True)
                        await self._write_packet(client, write_char, packet)
                        last_pressed = False

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

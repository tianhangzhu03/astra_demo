from __future__ import annotations

import argparse
import asyncio
import platform
from dataclasses import dataclass

from bleak import BleakClient, BleakScanner

from .ble_client import create_packet


DEFAULT_MAC = "FE:56:9B:F0:CF:0E"
DEFAULT_SERVICE_UUID = "000062c4-b99e-4141-9439-c4f9db977899"
DEFAULT_VOLTS = 2500
DEFAULT_FREQ = 100
DEFAULT_PULSE_MS = 1000


@dataclass
class ProbeConfig:
    mac: str
    service_uuid: str
    volts: int
    freq: int
    pulse_ms: int
    write_gap_ms: int
    range_start: int
    range_end: int
    list_only: bool


def _norm_uuid(u: str) -> str:
    return u.strip().lower()


def _props_set(char) -> set[str]:
    return {str(p).lower() for p in (getattr(char, "properties", []) or [])}


def _is_writable(char) -> bool:
    props = _props_set(char)
    return ("write" in props) or ("write-without-response" in props)


def _format_props(char) -> str:
    props = sorted(_props_set(char))
    return ",".join(props) if props else "-"


def _candidate_uuids_from_service(service_uuid: str, start_idx: int, end_idx: int) -> list[str]:
    tail = service_uuid[2:]
    return [f"{i:02d}{tail}" for i in range(start_idx, end_idx + 1)]


async def _write_packet(client: BleakClient, char, packet: bytes) -> str:
    props = _props_set(char)
    if "write" in props:
        try:
            await client.write_gatt_char(char.uuid, packet, response=True)
            return "write"
        except Exception:
            if "write-without-response" not in props:
                raise
    await client.write_gatt_char(char.uuid, packet, response=False)
    return "write-without-response"


def _iter_chars(client: BleakClient):
    for service in client.services:
        for char in service.characteristics:
            yield service, char


def _print_gatt_table(client: BleakClient) -> None:
    print("\n[GATT] Services / Characteristics")
    for service in client.services:
        print(f"  [SVC] {service.uuid}")
        for char in service.characteristics:
            print(f"    [CHR] {char.uuid}  props={_format_props(char)}")


def _select_probe_targets(client: BleakClient, cfg: ProbeConfig):
    svc_uuid = _norm_uuid(cfg.service_uuid)
    candidates = [_norm_uuid(u) for u in _candidate_uuids_from_service(cfg.service_uuid, cfg.range_start, cfg.range_end)]

    chars_by_uuid = {}
    writable_in_service = []
    for service, char in _iter_chars(client):
        char_uuid = _norm_uuid(char.uuid)
        chars_by_uuid[char_uuid] = (service, char)
        if _norm_uuid(service.uuid) == svc_uuid and _is_writable(char):
            writable_in_service.append((service, char))

    ordered = []
    seen = set()

    # Priority 1: writable chars that match 00~12 pattern.
    for u in candidates:
        found = chars_by_uuid.get(u)
        if not found:
            continue
        service, char = found
        if _norm_uuid(service.uuid) != svc_uuid:
            continue
        if not _is_writable(char):
            continue
        if u in seen:
            continue
        ordered.append((service, char, "pattern"))
        seen.add(u)

    # Priority 2: other writable chars under the target service.
    for service, char in writable_in_service:
        u = _norm_uuid(char.uuid)
        if u in seen:
            continue
        ordered.append((service, char, "writable"))
        seen.add(u)

    return ordered, candidates


async def _probe_one(client: BleakClient, char, cfg: ProbeConfig) -> str:
    on_packet = create_packet(cfg.volts, cfg.freq, start=True)
    off_packet = create_packet(0, 0, start=True)
    mode_used = await _write_packet(client, char, on_packet)
    await asyncio.sleep(max(0.0, cfg.pulse_ms / 1000.0))
    await _write_packet(client, char, off_packet)
    await asyncio.sleep(max(0.0, cfg.write_gap_ms / 1000.0))
    return mode_used


async def _run_probe(cfg: ProbeConfig) -> None:
    print(f"[BLE] Scanning by MAC: {cfg.mac}")
    device = await BleakScanner.find_device_by_address(cfg.mac, timeout=8.0)
    if not device:
        print("[ERR] Device not found by MAC.")
        return

    print("[BLE] Connecting...")
    async with BleakClient(device) as client:
        try:
            await client.get_services()
        except Exception:
            pass

        _print_gatt_table(client)

        targets, candidates = _select_probe_targets(client, cfg)
        print("\n[PROBE] Candidate UUIDs (00~12 pattern)")
        for u in candidates:
            print(f"  {u}")

        if cfg.list_only:
            print("\n[PROBE] list-only mode, no packets sent.")
            return

        if not targets:
            print("\n[ERR] No writable characteristic found under target service.")
            return

        print("\n[PROBE] Starting sequential test writes")
        print(f"  service={cfg.service_uuid}")
        print(f"  pulse={cfg.volts}mV? / {cfg.freq}Hz for {cfg.pulse_ms}ms")
        print("  After each test, confirm whether the device vibrated.")

        for idx, (service, char, source) in enumerate(targets, start=1):
            print("\n" + "=" * 60)
            print(f"[{idx}/{len(targets)}] Testing char: {char.uuid}")
            print(f"  service: {service.uuid}")
            print(f"  props:   {_format_props(char)}")
            print(f"  source:  {source}")
            try:
                mode_used = await _probe_one(client, char, cfg)
                print(f"  write:   {mode_used}")
            except Exception as exc:
                print(f"  [ERR] Write failed: {exc.__class__.__name__}: {exc}")
                continue

            ans = input("  Did it vibrate? [y/N/q]: ").strip().lower()
            if ans == "y":
                print(f"\n[RESULT] Likely control characteristic UUID: {char.uuid}")
                return
            if ans == "q":
                print("\n[ABORT] User stopped probing.")
                return

        print("\n[RESULT] No tested writable characteristic triggered vibration.")


def parse_args() -> ProbeConfig:
    p = argparse.ArgumentParser(
        description="Probe writable BLE characteristics and send low-risk test packets to locate vibration channel."
    )
    p.add_argument("--mac", default=DEFAULT_MAC, help="Target BLE device MAC address")
    p.add_argument("--service-uuid", default=DEFAULT_SERVICE_UUID, help="Target service UUID (default 0000...62c4...)")
    p.add_argument("--volts", type=int, default=DEFAULT_VOLTS, help="Test packet voltage field (default: 1000)")
    p.add_argument("--freq", type=int, default=DEFAULT_FREQ, help="Test packet frequency field (default: 50)")
    p.add_argument("--pulse-ms", type=int, default=DEFAULT_PULSE_MS, help="Vibration pulse duration in ms (default: 1000)")
    p.add_argument("--write-gap-ms", type=int, default=400, help="Gap after OFF packet before next test (default: 400)")
    p.add_argument("--range-start", type=int, default=0, help="Start index for 00~12 UUID probing pattern")
    p.add_argument("--range-end", type=int, default=12, help="End index for 00~12 UUID probing pattern")
    p.add_argument("--list-only", action="store_true", help="Only list services/characteristics and probe targets")
    args = p.parse_args()

    return ProbeConfig(
        mac=args.mac,
        service_uuid=args.service_uuid,
        volts=args.volts,
        freq=args.freq,
        pulse_ms=args.pulse_ms,
        write_gap_ms=args.write_gap_ms,
        range_start=min(args.range_start, args.range_end),
        range_end=max(args.range_start, args.range_end),
        list_only=args.list_only,
    )


def main() -> None:
    cfg = parse_args()
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run_probe(cfg))


if __name__ == "__main__":
    main()

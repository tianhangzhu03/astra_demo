> **2026-11-15 当前协议：** 用户已提供刷入完成的 CEL v0.2.1 和新版 `CEL_GUI (1).py`。ABC 现在自动核对版本，CW=5、CCW=6，STOP 为两字节 FF06。下面的 direction=5 接入方案及 `cel_direction_map.h` 是此前的历史提案，不是当前固件源码或烧录依据；CCW 通道组合未知，不自行补造。当前操作见 `astra_demo/ABC.md`。

# CEL Torsion protocol extension

The project assigns **direction 5** to the colleague-confirmed channel pattern
**[1, 0, 0, 1]** (channels 0 through 3). This extends the existing enum;
directions 0 through 4 retain their original meaning. It is not a bitmask.

`cel_direction_map.h` is the complete mapping helper for the device firmware.
The nRF52 application, GPIO assignments, actuator driver, and build/flash project
have not been supplied, so this directory is an integration component, not a
flashable firmware image. No device has been reflashed by this change.

## Wire format

The existing nine-byte command remains:

```
FF 10 00 05 Vlo Vhi 00 00 01
      |  |  +------ voltage setpoint in mV, unsigned little-endian
      |  +--------- Torsion direction ID
      +------------ Static mode
```

For example, encoding a 2500 mV setpoint yields `FF 10 00 05 C4 09 00 00 01`.
This is a protocol example, not a recommended operating voltage.
Duration remains host-controlled; it is not a field in this packet.
The host reuses the existing stop sequence: zero-voltage Normal command
`FF 10 00 00 00 00 00 00 01`, then `FF 06 00 00 00 00 00 00 00`.

## Device integration

In the existing handler for opcode `0x10`, retain packet-length, mode, voltage,
and start/stop validation. Map `packet[3]` through
`qsib_cel_direction_channels(packet[3], states)` and pass the resulting four
values to the existing channel driver, in its established ch0..ch3 order.
Apply the pattern as one actuator configuration, not four timed feedback cues.

The header intentionally does not assume that a channel value of zero means
"power off" or invent pin polarities. Continue using the board's existing driver
interpretation. Zero voltage and opcode `0x06` must also stop direction 5.
If the mapping helper returns false, use the existing reject/stop path.

After integrating, building and installing that firmware, launch the desktop
demo with `Start_AR_Demo.cmd --ble --torsion-firmware-ready`. This declares firmware
support; the current BLE protocol has no automatic capability handshake.
Without this flag the host rejects hardware Torsion requests while preserving
the other cues. `--no-ble` can simulate the new cue without the flag.

Both flip stages use direction 5. Roles `torsion_cw` and `torsion_ccw` describe
the intended hand action; neither this table nor this change establishes two
opposite physical torsion polarities.

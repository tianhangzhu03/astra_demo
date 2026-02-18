from dataclasses import dataclass


@dataclass
class AstraDemoConfig:
    top_camera_id: int = 0

    camera_fps: int = 30
    depth_unit: str = "mm"

    pinch_enter: float = 0.075
    pinch_exit: float = 0.095

    # Depth is considered "near enough" when z_mm <= threshold.
    depth_enter_mm: int = 520
    depth_exit_mm: int = 560

    enter_frames: int = 2
    exit_frames: int = 2
    smooth_alpha: float = 0.35

    frame_width: int = 640
    frame_height: int = 480

    ble_enabled: bool = True
    ble_mac_address: str = "FE:56:9B:F0:CF:0E"
    ble_uuid: str = "120062c4-b99e-4141-9439-c4f9db977899"
    ble_fixed_volts: int = 2000
    ble_fixed_freq: int = 100

    release_return_ms: int = 200

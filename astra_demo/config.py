from dataclasses import dataclass


@dataclass
class AstraDemoConfig:
    top_camera_id: int = 2
    side_color_camera_id: int = 1

    camera_fps: int = 30
    depth_unit: str = "mm"

    # Demo preset: easier pinch detection.
    pinch_enter: float = 0.085
    pinch_exit: float = 0.110

    # Demo preset: depth is only a soft gate, so thresholds are relaxed.
    depth_enter_mm: int = 760
    depth_exit_mm: int = 860

    enter_frames: int = 2
    exit_frames: int = 2
    smooth_alpha: float = 0.35

    frame_width: int = 640
    frame_height: int = 480

    # Top 3x3 panel layout
    grid_w_ratio: float = 0.84
    grid_h_ratio: float = 0.78
    grid_y_ratio: float = 0.10

    # Side depth visualization range
    depth_vis_min_mm: int = 220
    depth_vis_max_mm: int = 1350
    pointcloud_stride: int = 8

    # Virtual prop visuals
    prop_size_follow: int = 42
    prop_size_held: int = 56
    prop_follow_alpha: float = 0.30
    prop_toggle_cooldown_ms: int = 220

    ble_enabled: bool = True
    ble_mac_address: str = "FE:56:9B:F0:CF:0E"
    ble_uuid: str = "120062c4-b99e-4141-9439-c4f9db977899"
    ble_fixed_volts: int = 2000
    ble_fixed_freq: int = 100

    release_return_ms: int = 200

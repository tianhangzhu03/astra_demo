"""Local configuration for the physical pilot only."""

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    top_camera: int = 2
    side_camera: int = 3
    camera_fps: int = 30
    recording_fps: float = 30.0
    ui_fps: int = 20
    stale_after_s: float = 1.0
    side_mirror: bool = True
    hand_trace: bool = True
    hand_model: Path = PACKAGE_DIR / 'models' / 'hand_landmarker.task'
    hand_inference_width: int = 640
    ble_enabled: bool = True
    ble_address: str = 'FE:56:9B:F0:CF:0E'
    ble_uuid: str = '120062c4-b99e-4141-9439-c4f9db977899'
    subject: str = 'physical_pilot'
    object_name: str = 'AirPods case'
    object_notes: str = ''
    log_root: Path = PACKAGE_DIR / 'runs'

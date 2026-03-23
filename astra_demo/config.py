from dataclasses import dataclass


@dataclass
class AstraDemoConfig:
    # Top camera ID (used for grid hit testing).
    top_camera_id: int = 2
    # Side color camera ID (used for hand detection and rendering).
    side_color_camera_id: int = 1

    # Target capture FPS; higher values reduce perceived latency but increase CPU usage.
    camera_fps: int = 30
    depth_unit: str = "mm"

    # Thumb-index pinch enter threshold (normalized distance); larger is easier to trigger.
    pinch_enter: float = 0.085
    # Pinch exit threshold (should be > enter to form hysteresis); larger is harder to release.
    pinch_exit: float = 0.130

    # Side-view hand detection minimum confidence; higher reduces false positives but may miss detections.
    side_min_detection_confidence: float = 0.5
    # Side-view hand tracking minimum confidence; higher reduces jitter but may lose tracking.
    side_min_tracking_confidence: float = 0.65
    # Whether to apply CLAHE enhancement to side-view frames (recommended in low light / low contrast).
    side_use_clahe: bool = True
    # Hand detection input scale (<1 can reduce latency); smaller is faster but slightly less accurate.
    hand_process_scale: float = 0.75
    # Top-view hand detection scale (<1 reduces latency for the phone top camera path).
    top_hand_process_scale: float = 0.60
    # Top-view MediaPipe confidence thresholds.
    top_min_detection_confidence: float = 0.55
    top_min_tracking_confidence: float = 0.55
    # Top-view cursor smoothing / prediction (higher alpha and small prediction reduce visual lag).
    top_smooth_alpha: float = 0.82
    top_predict_beta: float = 0.20
    # Use palm center (more stable in top view) instead of thumb-index midpoint for top XY.
    top_use_palm_center: bool = True

    # Depth enter threshold (mm); smaller means the hand must be closer. Currently used as a soft gate.
    depth_enter_mm: int = 760
    # Depth exit threshold (mm, should be > enter); larger makes release less sensitive.
    depth_exit_mm: int = 860
    # Whether depth gating participates in grab decision; when False, depth is visualization only.
    use_depth_gate: bool = False

    # Consecutive frames required to enter a state; larger is more stable but adds latency.
    enter_frames: int = 2
    # Consecutive frames required to exit a state; larger is more stable but releases slower.
    exit_frames: int = 3
    # Midpoint smoothing factor (0~1); larger is more responsive, smaller is smoother.
    smooth_alpha: float = 0.65

    frame_width: int = 640
    frame_height: int = 480

    # Physical width of the calibrated interaction plane (cm); used to convert normalized pinch to centimeters.
    pinch_plane_width_cm: float = 62.0

    # Top grid width ratio; larger makes the grid bigger.
    grid_w_ratio: float = 0.84
    # Top grid height ratio; larger makes the grid bigger.
    grid_h_ratio: float = 0.78
    # Top grid Y-offset ratio; larger moves the grid downward.
    grid_y_ratio: float = 0.10

    # Minimum depth visualization distance (mm); Astra hand demos typically use 180~1200.
    depth_vis_min_mm: int = 180
    # Maximum depth visualization distance (mm); larger shows farther background.
    depth_vis_max_mm: int = 1200

    # Depth standalone window size; smaller reduces rendering cost.
    depth_view_width: int = 520
    depth_view_height: int = 390

    # Virtual prop idle/follow radius in pixels.
    prop_size_follow: int = 34
    # Virtual prop held radius in pixels; larger gives a minimal visual confirmation for vision-only trials.
    prop_size_held: int = 44
    # Whether the prop remains visible after release (True=keep, False=hide).
    prop_idle_visible: bool = True
    # Virtual prop follow smoothing factor; larger is more responsive, smaller is smoother.
    prop_follow_alpha: float = 0.68

    ble_enabled: bool = True
    ble_mac_address: str = "FE:56:9B:F0:CF:0E"
    ble_uuid: str = "120062c4-b99e-4141-9439-c4f9db977899"
    # Single demo haptic voltage shared by every successful grab.
    ble_fixed_volts: int = 2500
    # Single demo haptic frequency (Hz) shared by every successful grab.
    ble_fixed_freq: int = 100
    # Single one-shot vibration pulse duration (ms) shared by every successful grab.
    ble_pulse_ms: int = 500

    # Default participant label used when --subject is not provided.
    default_subject_id: str = "subject01"
    # Per-run CSV output directory (created under the repo root).
    session_log_dir: str = "session_logs"

    # Return animation duration after release (ms).
    release_return_ms: int = 200

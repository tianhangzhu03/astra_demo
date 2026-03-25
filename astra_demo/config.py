from dataclasses import dataclass


@dataclass
class AstraDemoConfig:
    # Top camera ID (used for grid hit testing).
    top_camera_id: int = 1
    # Side color camera ID (used for hand detection and rendering).
    side_color_camera_id: int = 2
    # Use the configured camera IDs directly instead of silently falling back to other camera indices.
    strict_camera_ids: bool = True
    # Mirror the top camera horizontally; rear-phone top view usually wants False.
    top_mirror_horizontal: bool = False
    # Mirror the side camera so operator motion feels natural.
    side_mirror_horizontal: bool = True

    # Target capture FPS; higher values reduce perceived latency but increase CPU usage.
    camera_fps: int = 30

    # Thumb-index pinch enter threshold (normalized distance); larger is easier to trigger.
    pinch_enter: float = 0.085
    # Pinch exit threshold (should be > enter to form hysteresis); smaller releases earlier after opening fingers.
    pinch_exit: float = 0.094
    # Keep pinch "closed" for a few frames when side tracking briefly drops during fast motion.
    pinch_missing_hold_frames: int = 4
    # Median window for side-view pinch distance; a short window suppresses single-frame spikes during fast motion.
    pinch_median_window: int = 3

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
    # Frames to keep the last top-view hand position when tracking drops briefly.
    top_visual_hold_frames: int = 12
    # Extra hold during active grab; larger helps avoid false visual drop while moving across zones.
    top_grab_hold_frames: int = 8

    # Consecutive frames required to enter a state; larger is more stable but adds latency.
    enter_frames: int = 2
    # Consecutive frames required to exit a state; 1 makes deliberate finger opening feel more immediate.
    exit_frames: int = 1
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
    # Briefly hold the last valid zone when the top-view cursor flickers on a boundary.
    hover_key_hold_frames: int = 4
    # Show the side RGB window used by the operator to monitor pinch behavior.
    show_side_color_window: bool = True

    # Virtual prop idle/follow radius in pixels.
    prop_size_follow: int = 44
    # Virtual prop held radius in pixels; keep it equal to idle/follow size so grabbing does not change scale.
    prop_size_held: int = 44
    # Whether the prop remains visible after release (True=keep, False=hide).
    prop_idle_visible: bool = True
    # Virtual prop follow smoothing factor; larger is more responsive, smaller is smoother.
    prop_follow_alpha: float = 0.68

    ble_enabled: bool = True
    ble_mac_address: str = "FE:56:9B:F0:CF:0E"
    ble_uuid: str = "120062c4-b99e-4141-9439-c4f9db977899"
    # Single demo haptic voltage shared by every successful grab.
    ble_fixed_volts: int = 2000
    # Single demo haptic frequency (Hz) shared by every successful grab.
    ble_fixed_freq: int = 100
    # Single one-shot vibration pulse duration (ms) shared by every successful grab.
    ble_pulse_ms: int = 200

    # Default participant label used when --subject is not provided.
    default_subject_id: str = "subject01"
    # Per-run CSV output directory (created under the repo root).
    session_log_dir: str = "session_logs"

    # Return animation duration after release (ms).
    release_return_ms: int = 200

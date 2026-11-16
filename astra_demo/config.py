from dataclasses import dataclass


@dataclass
class AstraDemoConfig:
    # Top camera ID (used for planar hand tracking).
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

    # Legacy normalized thresholds, retained for old scripts; main uses *_ratio below.
    pinch_enter: float = 0.085
    # Pinch exit threshold (should be > enter to form hysteresis); smaller releases earlier after opening fingers.
    pinch_exit: float = 0.094
    # Keep pinch "closed" for a few frames when side tracking briefly drops during fast motion.
    pinch_missing_hold_frames: int = 4
    # Median window for side-view pinch distance; a short window suppresses single-frame spikes during fast motion.
    pinch_median_window: int = 3
    # New grasp decisions use fingertip separation / palm size, not image width.
    pinch_enter_ratio: float = 0.42
    # A relaxed, partly open pose is enough to prepare a new grasp after the G cue.
    pinch_ready_ratio: float = 0.60
    pinch_release_ratio: float = 0.85
    # Top-view pinch anchor must be within ball radius + this pixel margin to attach.
    grab_radius_margin_px: float = 20.0
    # Preserve a confirmed nearby opening through brief Side fingertip occlusion.
    # Attachment still needs current valid Top proximity and consecutive closed frames.
    grasp_gesture_gap_ms: int = 500
    release_hold_ms: int = 350
    release_min_frames: int = 3
    # Bound visual extrapolation; the grasp itself stays latched until an explicit release.
    optical_flow_hold_s: float = 0.7
    # Top current-image flow may bridge landmark loss before attachment, never stale XY alone.
    top_region_flow_s: float = 0.5
    tracking_min_cutoff: float = 1.4
    tracking_beta: float = 0.015
    # Displayed palm trail only; accuracy metrics retain unsmoothed detections.
    trail_min_cutoff: float = 1.1
    trail_beta: float = 0.012
    trail_deadband_px: float = 2.0
    # Lightweight side ball: fixed pixel offset from the palm, independent x/y filtering.
    side_tracking_min_cutoff: float = 0.9
    side_tracking_beta: float = 0.012
    side_tracking_deadband_px: float = 1.5
    # Move the held Side sphere 25% from the pinch anchor toward the palm, once per grasp.
    side_grip_palm_inset: float = .25

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
    # Legacy EMA/prediction fields; main now uses PalmTracker and the tracking_* fields.
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
    # Legacy exit/smoothing fields; main uses release_min_frames and release_hold_ms.
    exit_frames: int = 1
    # Midpoint smoothing factor (0~1); larger is more responsive, smaller is smoother.
    smooth_alpha: float = 0.65

    frame_width: int = 640
    frame_height: int = 480

    # Physical width of the calibrated interaction plane (cm); used to convert normalized pinch to centimeters.
    pinch_plane_width_cm: float = 62.0

    # Show a separate side camera debug window; the operator console already includes this view.
    show_side_color_window: bool = True
    # English manual BLE operator console; standalone --preview never uses hardware.
    show_operator_window: bool = True
    # Limit console composition frequency without skipping hand-detection frames.
    operator_panel_fps: int = 20

    # Virtual prop idle/follow radius in pixels.
    prop_size_follow: int = 48
    # Virtual prop held radius in pixels; keep it equal to idle/follow size so grabbing does not change scale.
    prop_size_held: int = 48
    # Whether the prop remains visible after release (True=keep, False=hide).
    prop_idle_visible: bool = True
    # Legacy second EMA, unused by main (PalmTracker now filters position once).
    prop_follow_alpha: float = 0.68

    ble_enabled: bool = True
    ble_mac_address: str = "FE:56:9B:F0:CF:0E"
    ble_uuid: str = "120062c4-b99e-4141-9439-c4f9db977899"
    # Manual output starts at zero. UI steps: 25 mV; range 0..5000, not a safety rating.
    manual_voltage_mv: int = 0
    # Finite pulse duration, 0.1..5.0 seconds. The PC schedules the stop command.
    manual_duration_s: float = 0.2
    # Vibration only, 10..200 Hz; UI step is 10 Hz. Force cues use Static mode.
    manual_frequency_hz: int = 100
    # Rising-edge confirmation uses the operator's current voltage, frequency and duration.
    grab_confirmation_enabled: bool = True
    # Separate video files, resampled to wall-clock time; repeated frames are logged.
    recording_fps: float = 30.0
    # Include the operator canvas when N starts a trial; never record idle preview.
    record_operator_ui: bool = True
    # Legacy automatic-grasp settings below are not used by the manual entry point.
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

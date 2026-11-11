from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path
from sys import platform
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .manual_control import Actuation, ManualBleController
from .config import AstraDemoConfig
from .grab_logic import GrabContext, update_grab_state
from .mp_hands_compat import load_hands_api
from .operator_panel import OperatorPanel, draw_workspace_boundary, initial_ball_position
from .session_logger import SessionLogger
from .virtual_prop import VirtualProp


def get_thumb_index_data(hand_lms, w: int, h: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], float]:
    thumb = hand_lms.landmark[4]
    index_tip = hand_lms.landmark[8]

    tx, ty = int(thumb.x * w), int(thumb.y * h)
    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
    mx, my = (tx + ix) // 2, (ty + iy) // 2
    dist = math.hypot(index_tip.x - thumb.x, index_tip.y - thumb.y)
    return (tx, ty), (ix, iy), (mx, my), dist


def get_thumb_index_cluster_data(hand_lms, w: int, h: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], float]:
    # More robust pinch anchors: blend fingertip with nearby joints to reduce landmark jitter/occlusion sensitivity.
    thumb_spec = ((4, 0.60), (3, 0.28), (2, 0.12))
    index_spec = ((8, 0.55), (7, 0.30), (6, 0.15))

    def _anchor(spec):
        sx = 0.0
        sy = 0.0
        sw = 0.0
        for idx, wt in spec:
            lm = hand_lms.landmark[idx]
            sx += lm.x * wt
            sy += lm.y * wt
            sw += wt
        return sx / sw, sy / sw

    tx_n, ty_n = _anchor(thumb_spec)
    ix_n, iy_n = _anchor(index_spec)
    mx_n, my_n = (tx_n + ix_n) * 0.5, (ty_n + iy_n) * 0.5

    tx, ty = int(tx_n * w), int(ty_n * h)
    ix, iy = int(ix_n * w), int(iy_n * h)
    mx, my = int(mx_n * w), int(my_n * h)
    dist = math.hypot(ix_n - tx_n, iy_n - ty_n)
    return (tx, ty), (ix, iy), (mx, my), dist


def get_palm_center_xy(hand_lms, w: int, h: int) -> tuple[int, int]:
    # Palm landmarks are usually more stable than fingertips in top view.
    palm_ids = (0, 5, 9, 13, 17)
    xs = [hand_lms.landmark[i].x for i in palm_ids]
    ys = [hand_lms.landmark[i].y for i in palm_ids]
    cx = int((sum(xs) / len(xs)) * w)
    cy = int((sum(ys) / len(ys)) * h)
    return (cx, cy)


def get_landmark_bbox_center_xy(hand_lms, w: int, h: int, landmark_ids: Optional[Tuple[int, ...]] = None) -> tuple[int, int]:
    ids = landmark_ids if landmark_ids is not None else tuple(range(len(hand_lms.landmark)))
    xs = [hand_lms.landmark[i].x for i in ids]
    ys = [hand_lms.landmark[i].y for i in ids]
    cx = int(((min(xs) + max(xs)) * 0.5) * w)
    cy = int(((min(ys) + max(ys)) * 0.5) * h)
    return (cx, cy)


def get_top_fist_anchor_xy(hand_lms, w: int, h: int) -> tuple[int, int]:
    # Robust anchor for top-view fist/hand-back pose:
    # fuse palm center + palm bbox center + whole-hand bbox center.
    palm_ids = (0, 5, 9, 13, 17)
    palm_center = get_palm_center_xy(hand_lms, w, h)
    palm_box_center = get_landmark_bbox_center_xy(hand_lms, w, h, palm_ids)
    hand_box_center = get_landmark_bbox_center_xy(hand_lms, w, h)
    x = int(0.50 * palm_center[0] + 0.30 * palm_box_center[0] + 0.20 * hand_box_center[0])
    y = int(0.50 * palm_center[1] + 0.30 * palm_box_center[1] + 0.20 * hand_box_center[1])
    return (x, y)


def clamp_point_step(prev_xy: Tuple[int, int], cur_xy: Tuple[int, int], max_step_px: float) -> tuple[int, int]:
    dx = float(cur_xy[0] - prev_xy[0])
    dy = float(cur_xy[1] - prev_xy[1])
    dist = math.hypot(dx, dy)
    if dist <= max_step_px or dist <= 1e-6:
        return cur_xy
    s = max_step_px / dist
    return (int(prev_xy[0] + dx * s), int(prev_xy[1] + dy * s))


def enhance_for_hand_detection(frame_bgr, clahe):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def open_uvc_camera(preferred_id: int, width: int, height: int, fps: int, label: str, strict_id: bool):
    tried = []
    candidate_ids = [preferred_id] if strict_id else [preferred_id, 0, 1, 2, 3, 4]
    for cam_id in candidate_ids:
        if cam_id in tried:
            continue
        tried.append(cam_id)
        # Match camera_id_probe.py so the same numeric ID maps to the same physical device on Windows.
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[CAM] {label} use id={cam_id}")
            return cap
        cap.release()
    mode = "strict" if strict_id else "fallback"
    raise RuntimeError(f"{label} open failed in {mode} mode. tried ids={tried}")


def map_point_to_full_res(pt: tuple[int, int], scale: float, full_w: int, full_h: int) -> tuple[int, int]:
    if scale >= 0.999:
        x, y = pt
    else:
        x = int(pt[0] / max(scale, 1e-6))
        y = int(pt[1] / max(scale, 1e-6))
    x = max(0, min(full_w - 1, x))
    y = max(0, min(full_h - 1, y))
    return (x, y)


def main() -> None:
    if platform != "win32":
        print("[WARN] This demo is intended for Windows camera capture.")

    cfg = AstraDemoConfig()
    parser = argparse.ArgumentParser(description="Top camera + side camera grab demo")
    parser.add_argument(
        "--subject",
        default=cfg.default_subject_id,
        help="Participant label used in the per-run CSV filename and rows.",
    )
    args = parser.parse_args()

    session_logger = SessionLogger(
        subject_id=args.subject,
        output_dir=Path(__file__).resolve().parents[1] / cfg.session_log_dir,
        plane_width_cm=cfg.pinch_plane_width_cm,
    )
    print(f"[LOG] Writing grab records to {session_logger.path}")

    ble = ManualBleController(cfg.ble_enabled, cfg.ble_mac_address, cfg.ble_uuid)

    top_cap = open_uvc_camera(
        cfg.top_camera_id,
        cfg.frame_width,
        cfg.frame_height,
        cfg.camera_fps,
        "top(phone)",
        strict_id=cfg.strict_camera_ids,
    )
    side_cap = open_uvc_camera(
        cfg.side_color_camera_id,
        cfg.frame_width,
        cfg.frame_height,
        cfg.camera_fps,
        "side(color-uvc)",
        strict_id=cfg.strict_camera_ids,
    )

    mp_hands, _ = load_hands_api()
    hands_top = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=cfg.top_min_detection_confidence,
        min_tracking_confidence=cfg.top_min_tracking_confidence,
        model_complexity=0,
    )
    hands_side = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=cfg.side_min_detection_confidence,
        min_tracking_confidence=cfg.side_min_tracking_confidence,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    grab_ctx = GrabContext()
    prop = VirtualProp(release_return_ms=cfg.release_return_ms)
    prop.size_follow = cfg.prop_size_follow
    prop.size_held = cfg.prop_size_held
    prop.follow_alpha = cfg.prop_follow_alpha

    smooth_top_mid: Optional[list[float]] = None
    last_top_filtered_xy: Optional[tuple[float, float]] = None
    last_top_visible_mid: Optional[Tuple[int, int]] = None
    last_top_visible_prop_xy: Optional[Tuple[int, int]] = None
    top_lost_frames = 999
    smooth_side_mid: Optional[list[float]] = None
    prop_initialized = False
    last_grab_trigger = False
    top_visual_hold_frames = cfg.top_visual_hold_frames
    top_grab_hold_frames = cfg.top_grab_hold_frames
    top_jump_limit_px = 120.0
    last_wait_log_ms = 0
    pinch_history = deque(maxlen=max(1, cfg.pinch_median_window))
    panel = OperatorPanel(ble, Actuation(
        volts_mv=cfg.manual_voltage_mv,
        duration_s=cfg.manual_duration_s,
        frequency_hz=cfg.manual_frequency_hz,
    ))
    panel_interval = 1.0 / max(1, cfg.operator_panel_fps)
    last_panel_time = 0.0
    previous_frame_time = time.perf_counter()
    measured_fps = 0.0

    print("[INFO] Top camera + side camera demo starting. Press 'q' to quit.")

    try:
        ble.start()
        if cfg.show_operator_window:
            panel.open()
        while True:
            now_ms = int(time.time() * 1000)
            ok_top, top_frame_raw = top_cap.read()
            ok_side, side_color_raw = side_cap.read()
            missing_sources = []
            if not ok_top:
                missing_sources.append("top_rgb")
            if not ok_side:
                missing_sources.append("side_rgb")

            if missing_sources:
                if now_ms - last_wait_log_ms >= 1000:
                    print(f"[WAIT] Missing frames: {', '.join(missing_sources)}")
                    last_wait_log_ms = now_ms
                if cfg.show_operator_window and time.perf_counter() - last_panel_time >= panel_interval:
                    cv2.imshow(panel.window_name, panel.render(
                        ble_status=ble.status_msg,
                    ))
                    last_panel_time = time.perf_counter()
                panel.handle_key(cv2.waitKeyEx(10))
                if panel.quit_requested:
                    break
                if cfg.show_operator_window and cv2.getWindowProperty(panel.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue

            top_frame = top_frame_raw.copy()
            if cfg.top_mirror_horizontal:
                top_frame = cv2.flip(top_frame, 1)

            side_frame = side_color_raw.copy()
            if cfg.side_mirror_horizontal:
                side_frame = cv2.flip(side_frame, 1)
            side_timestamp_ms = now_ms

            top_h, top_w, _ = top_frame.shape
            side_h, side_w, _ = side_frame.shape

            if not prop_initialized:
                # Top-view operation: initialize prop directly in top view.
                init_top_xy = initial_ball_position(top_w, top_h)
                prop.initialize_at(init_top_xy, visible=cfg.prop_idle_visible)
                prop_initialized = True

            # Top camera provides the continuous planar hand position.
            top_detect_frame = top_frame
            if cfg.top_hand_process_scale < 0.999:
                top_detect_frame = cv2.resize(
                    top_frame,
                    (int(top_w * cfg.top_hand_process_scale), int(top_h * cfg.top_hand_process_scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
            top_result = hands_top.process(cv2.cvtColor(top_detect_frame, cv2.COLOR_BGR2RGB))
            top_mid: Optional[Tuple[int, int]] = None
            top_prop_xy: Optional[Tuple[int, int]] = None
            top_detected_now = False
            if top_result.multi_hand_landmarks:
                top_detected_now = True
                top_lms = top_result.multi_hand_landmarks[0]
                detect_h_t, detect_w_t = top_detect_frame.shape[:2]
                _, _, top_pinch_proxy_raw, _ = get_thumb_index_cluster_data(top_lms, detect_w_t, detect_h_t)
                if cfg.top_use_palm_center:
                    top_mid_raw = get_top_fist_anchor_xy(top_lms, detect_w_t, detect_h_t)
                else:
                    _, _, top_mid_raw, _ = get_thumb_index_data(top_lms, detect_w_t, detect_h_t)
                top_mid_raw = map_point_to_full_res(top_mid_raw, cfg.top_hand_process_scale, top_w, top_h)
                top_pinch_proxy_raw = map_point_to_full_res(top_pinch_proxy_raw, cfg.top_hand_process_scale, top_w, top_h)
                if last_top_visible_mid is not None:
                    top_mid_raw = clamp_point_step(last_top_visible_mid, top_mid_raw, top_jump_limit_px)
                if smooth_top_mid is None:
                    smooth_top_mid = [float(top_mid_raw[0]), float(top_mid_raw[1])]
                else:
                    smooth_top_mid[0] = (1.0 - cfg.top_smooth_alpha) * smooth_top_mid[0] + cfg.top_smooth_alpha * top_mid_raw[0]
                    smooth_top_mid[1] = (1.0 - cfg.top_smooth_alpha) * smooth_top_mid[1] + cfg.top_smooth_alpha * top_mid_raw[1]

                top_x, top_y = smooth_top_mid[0], smooth_top_mid[1]
                if last_top_filtered_xy is not None and cfg.top_predict_beta > 0.0:
                    dx = top_x - last_top_filtered_xy[0]
                    dy = top_y - last_top_filtered_xy[1]
                    top_x += cfg.top_predict_beta * dx
                    top_y += cfg.top_predict_beta * dy

                top_x = max(0.0, min(float(top_w - 1), top_x))
                top_y = max(0.0, min(float(top_h - 1), top_y))
                top_mid = (int(top_x), int(top_y))
                last_top_filtered_xy = (smooth_top_mid[0], smooth_top_mid[1])
                last_top_visible_mid = top_mid
                last_top_visible_prop_xy = top_pinch_proxy_raw
                top_lost_frames = 0
                # The virtual prop should sit directly at the thumb-index pinch area, not the palm center.
                top_prop_xy = top_pinch_proxy_raw
            else:
                top_lost_frames += 1
                if last_top_visible_mid is not None and top_lost_frames <= top_visual_hold_frames:
                    top_mid = last_top_visible_mid
                    top_prop_xy = last_top_visible_prop_xy if last_top_visible_prop_xy is not None else top_mid
                else:
                    smooth_top_mid = None
                    last_top_filtered_xy = None
                    last_top_visible_mid = None
                    last_top_visible_prop_xy = None

            if top_prop_xy is None:
                top_prop_xy = top_mid

            # Side camera: pinch/state/BLE/prop.
            side_detect_frame = enhance_for_hand_detection(side_frame, clahe) if cfg.side_use_clahe else side_frame
            if cfg.hand_process_scale < 0.999:
                side_detect_frame = cv2.resize(
                    side_detect_frame,
                    (int(side_w * cfg.hand_process_scale), int(side_h * cfg.hand_process_scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
            side_result = hands_side.process(cv2.cvtColor(side_detect_frame, cv2.COLOR_BGR2RGB))
            side_pinch_dist: Optional[float] = None
            side_pinch_for_logic: Optional[float] = None
            side_pinch_cm: Optional[float] = None
            side_mid: Optional[Tuple[int, int]] = None
            if side_result.multi_hand_landmarks:
                side_lms = side_result.multi_hand_landmarks[0]
                detect_h_s, detect_w_s = side_detect_frame.shape[:2]
                side_thumb, side_index, side_mid_raw, side_pinch_dist = get_thumb_index_cluster_data(side_lms, detect_w_s, detect_h_s)
                side_thumb = map_point_to_full_res(side_thumb, cfg.hand_process_scale, side_w, side_h)
                side_index = map_point_to_full_res(side_index, cfg.hand_process_scale, side_w, side_h)
                side_mid_raw = map_point_to_full_res(side_mid_raw, cfg.hand_process_scale, side_w, side_h)

                if smooth_side_mid is None:
                    smooth_side_mid = [float(side_mid_raw[0]), float(side_mid_raw[1])]
                else:
                    smooth_side_mid[0] = (1.0 - cfg.smooth_alpha) * smooth_side_mid[0] + cfg.smooth_alpha * side_mid_raw[0]
                    smooth_side_mid[1] = (1.0 - cfg.smooth_alpha) * smooth_side_mid[1] + cfg.smooth_alpha * side_mid_raw[1]
                side_mid = (int(smooth_side_mid[0]), int(smooth_side_mid[1]))

                if cfg.show_side_color_window:
                    cv2.circle(side_frame, side_thumb, 8, (255, 200, 0), -1)
                    cv2.circle(side_frame, side_index, 8, (0, 255, 0), -1)
                    cv2.circle(side_frame, side_mid, 6, (255, 255, 255), -1)
                    cv2.line(side_frame, side_thumb, side_index, (255, 255, 0), 2)
                pinch_history.append(side_pinch_dist)
                side_pinch_for_logic = float(np.median(np.asarray(pinch_history, dtype=np.float32)))
                side_pinch_cm = session_logger.norm_to_cm(side_pinch_dist)
            else:
                smooth_side_mid = None
                pinch_history.clear()

            out = update_grab_state(
                ctx=grab_ctx,
                pinch_dist=side_pinch_for_logic,
                hover_key=0,  # No zones in the manual guidance scenario.
                pinch_enter=cfg.pinch_enter,
                pinch_exit=cfg.pinch_exit,
                enter_frames=cfg.enter_frames,
                exit_frames=cfg.exit_frames,
                pinch_missing_hold_frames=cfg.pinch_missing_hold_frames,
                top_hand_present=(top_detected_now or (last_top_visible_mid is not None and top_lost_frames <= top_grab_hold_frames)),
            )
            grab_ctx = out.context

            if out.trigger_on and not last_grab_trigger:
                pinch_cm = session_logger.log_grab(
                    timestamp_ms=side_timestamp_ms,
                    pinch_norm=side_pinch_dist,
                    target_key=out.target_key,
                )
                pinch_str = f"{pinch_cm:.2f}cm" if pinch_cm is not None else "-"
                print(
                    f"[GRAB] ts_ms={side_timestamp_ms} "
                    f"pinch={pinch_str} target={out.target_key} subject={session_logger.subject_id}"
                )
            last_grab_trigger = out.trigger_on

            # Hardware cues are operator-only; visual GRAB never sends BLE output.
            prop.update(
                hand_xy=top_prop_xy,
                grab_active=out.trigger_on,
                now_ms=side_timestamp_ms,
                keep_idle_visible=cfg.prop_idle_visible,
            )

            # The workspace outline is visual only; it does not gate grasp detection.
            draw_workspace_boundary(top_frame)
            prop.draw(top_frame, show_label=False)

            tick = time.perf_counter()
            instantaneous_fps = 1.0 / max(0.001, tick - previous_frame_time)
            measured_fps = instantaneous_fps if measured_fps == 0 else 0.9 * measured_fps + 0.1 * instantaneous_fps
            previous_frame_time = tick
            if cfg.show_operator_window and tick - last_panel_time >= panel_interval:
                cv2.imshow(panel.window_name, panel.render(
                    top_frame, side_frame,
                    fps=measured_fps, ble_status=ble.status_msg,
                ))
                last_panel_time = tick

            cv2.imshow("Workspace | Top camera", top_frame)
            if cfg.show_side_color_window:
                cv2.imshow("Tracking | Side camera", side_frame)

            panel.handle_key(cv2.waitKeyEx(1))
            if panel.quit_requested:
                break
            if cfg.show_operator_window and cv2.getWindowProperty(panel.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        ble.stop()
        session_logger.close()
        top_cap.release()
        side_cap.release()
        hands_top.close()
        hands_side.close()
        cv2.destroyAllWindows()
        print("[INFO] Exit.")


if __name__ == "__main__":
    main()

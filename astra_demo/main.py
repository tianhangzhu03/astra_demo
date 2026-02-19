from __future__ import annotations

import math
from sys import platform
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .ble_client import BleConfig, BleController
from .camera_source import create_camera_source
from .config import AstraDemoConfig
from .grab_logic import GrabContext, sample_depth_5x5, update_grab_state
from .mp_hands_compat import load_hands_api
from .virtual_prop import VirtualProp


def get_thumb_index_data(hand_lms, w: int, h: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], float]:
    thumb = hand_lms.landmark[4]
    index_tip = hand_lms.landmark[8]

    tx, ty = int(thumb.x * w), int(thumb.y * h)
    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
    mx, my = (tx + ix) // 2, (ty + iy) // 2
    dist = math.hypot(index_tip.x - thumb.x, index_tip.y - thumb.y)
    return (tx, ty), (ix, iy), (mx, my), dist


def compute_targets(w: int, h: int, panel_w_ratio: float, panel_h_ratio: float, panel_y_ratio: float) -> dict[int, tuple[int, int, int, int]]:
    panel_w = int(w * panel_w_ratio)
    panel_h = int(h * panel_h_ratio)
    sx = (w - panel_w) // 2
    sy = int(h * panel_y_ratio)
    cw = panel_w // 3
    ch = panel_h // 3

    targets: dict[int, tuple[int, int, int, int]] = {}
    for r in range(3):
        for c in range(3):
            key = r * 3 + c + 1
            x1 = sx + c * cw
            y1 = sy + r * ch
            x2 = x1 + cw
            y2 = y1 + ch
            targets[key] = (x1, y1, x2, y2)
    return targets


def hit_test(x: int, y: int, targets: dict[int, tuple[int, int, int, int]]) -> int:
    for key, (x1, y1, x2, y2) in targets.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return key
    return 0


def draw_panel(frame, targets, hover_key: int, grab_key: int) -> None:
    overlay = frame.copy()
    for key, (x1, y1, x2, y2) in targets.items():
        if key == grab_key:
            fill = (40, 40, 245)
        elif key == hover_key:
            fill = (30, 220, 245)
        else:
            fill = (180, 180, 180)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 230, 230), 1)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.putText(frame, str(key), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)


def build_depth_pointcloud_preview(depth_mm, depth_min_mm: int, depth_max_mm: int, stride: int = 8):
    h, w = depth_mm.shape[:2]
    canvas_h, canvas_w = 165, 220
    cloud = np.full((canvas_h, canvas_w, 3), 12, dtype=np.uint8)

    z_span = float(max(1, depth_max_mm - depth_min_mm))
    step = max(4, stride)
    for y in range(0, h, step):
        for x in range(0, w, step):
            z = int(depth_mm[y, x])
            if z <= 0 or z < depth_min_mm or z > depth_max_mm:
                continue

            zn = (z - depth_min_mm) / z_span  # 0 near -> 1 far
            px = int((x / max(1, w - 1)) * (canvas_w - 1))
            py = int(20 + (1.0 - zn) * 90 + (y / max(1, h - 1)) * 40)
            py = max(0, min(canvas_h - 1, py))

            col = cv2.applyColorMap(np.array([[int((1.0 - zn) * 255)]], dtype=np.uint8), cv2.COLORMAP_TURBO)
            bgr = tuple(int(v) for v in col[0, 0].tolist())
            cv2.circle(cloud, (px, py), 1, bgr, -1)
    return cloud


def open_uvc_camera(preferred_id: int, width: int, height: int, label: str):
    tried = []
    for cam_id in [preferred_id, 0, 1, 2, 3, 4]:
        if cam_id in tried:
            continue
        tried.append(cam_id)
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"[CAM] {label} use id={cam_id}")
            return cap
        cap.release()
    raise RuntimeError(f"{label} open failed. tried ids={tried}")


def main() -> None:
    if platform != "win32":
        print("[WARN] This Astra demo is intended for Windows + Astra SDK.")

    cfg = AstraDemoConfig()

    ble = BleController(
        BleConfig(
            enabled=cfg.ble_enabled,
            mac_address=cfg.ble_mac_address,
            uuid=cfg.ble_uuid,
            fixed_volts=cfg.ble_fixed_volts,
            fixed_freq=cfg.ble_fixed_freq,
        )
    )
    ble.start()

    top_cap = open_uvc_camera(cfg.top_camera_id, cfg.frame_width, cfg.frame_height, "top(phone)")
    side_cap = open_uvc_camera(cfg.side_color_camera_id, cfg.frame_width, cfg.frame_height, "side(color-uvc)")
    side_cam = create_camera_source(fps=cfg.camera_fps)
    side_cam.start()

    mp_hands, _ = load_hands_api()
    hands_top = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    hands_side = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    grab_ctx = GrabContext()
    prop = VirtualProp(release_return_ms=cfg.release_return_ms)
    prop.size_follow = cfg.prop_size_follow
    prop.size_held = cfg.prop_size_held
    prop.follow_alpha = cfg.prop_follow_alpha

    smooth_top_mid: Optional[list[float]] = None
    smooth_side_mid: Optional[list[float]] = None
    checked_alignment = False
    last_toggle_ms = 0

    print("[INFO] Top camera + side Astra demo starting. Press 'q' to quit, 'v' to toggle prop type.")

    try:
        while True:
            ok_top, top_frame_raw = top_cap.read()
            ok_side, side_color_raw = side_cap.read()
            side_bundle = side_cam.read()
            if not ok_top or not ok_side or side_bundle is None:
                time.sleep(0.01)
                continue

            top_frame = cv2.flip(top_frame_raw.copy(), 1)
            side_frame = cv2.flip(side_color_raw.copy(), 1)
            side_depth = cv2.flip(side_bundle.depth_mm.copy(), 1)

            if not checked_alignment:
                if side_frame.shape[:2] != side_depth.shape[:2]:
                    raise RuntimeError(
                        "Side RGB(UVC)/Depth(OpenNI) size mismatch: "
                        f"rgb={side_frame.shape[:2]} depth={side_depth.shape[:2]}. "
                        "Set side UVC camera resolution to match Astra depth stream."
                    )
                checked_alignment = True

            top_h, top_w, _ = top_frame.shape
            side_h, side_w, _ = side_frame.shape

            targets = compute_targets(
                top_w,
                top_h,
                panel_w_ratio=cfg.grid_w_ratio,
                panel_h_ratio=cfg.grid_h_ratio,
                panel_y_ratio=cfg.grid_y_ratio,
            )
            prop.set_dock((float(side_w * 0.12), float(side_h * 0.22)))

            # Top camera: 3x3 grid test only.
            top_result = hands_top.process(cv2.cvtColor(top_frame, cv2.COLOR_BGR2RGB))
            hover_key = 0
            top_mid: Optional[Tuple[int, int]] = None
            if top_result.multi_hand_landmarks:
                top_lms = top_result.multi_hand_landmarks[0]
                _, _, top_mid_raw, _ = get_thumb_index_data(top_lms, top_w, top_h)
                if smooth_top_mid is None:
                    smooth_top_mid = [float(top_mid_raw[0]), float(top_mid_raw[1])]
                else:
                    smooth_top_mid[0] = (1.0 - cfg.smooth_alpha) * smooth_top_mid[0] + cfg.smooth_alpha * top_mid_raw[0]
                    smooth_top_mid[1] = (1.0 - cfg.smooth_alpha) * smooth_top_mid[1] + cfg.smooth_alpha * top_mid_raw[1]
                top_mid = (int(smooth_top_mid[0]), int(smooth_top_mid[1]))
                hover_key = hit_test(top_mid[0], top_mid[1], targets)
            else:
                smooth_top_mid = None

            # Side Astra: pinch/depth gate/state/BLE/prop.
            side_result = hands_side.process(cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB))
            side_pinch_dist: Optional[float] = None
            side_mid: Optional[Tuple[int, int]] = None
            if side_result.multi_hand_landmarks:
                side_lms = side_result.multi_hand_landmarks[0]
                side_thumb, side_index, side_mid_raw, side_pinch_dist = get_thumb_index_data(side_lms, side_w, side_h)

                if smooth_side_mid is None:
                    smooth_side_mid = [float(side_mid_raw[0]), float(side_mid_raw[1])]
                else:
                    smooth_side_mid[0] = (1.0 - cfg.smooth_alpha) * smooth_side_mid[0] + cfg.smooth_alpha * side_mid_raw[0]
                    smooth_side_mid[1] = (1.0 - cfg.smooth_alpha) * smooth_side_mid[1] + cfg.smooth_alpha * side_mid_raw[1]
                side_mid = (int(smooth_side_mid[0]), int(smooth_side_mid[1]))

                cv2.circle(side_frame, side_thumb, 8, (255, 200, 0), -1)
                cv2.circle(side_frame, side_index, 8, (0, 255, 0), -1)
                cv2.circle(side_frame, side_mid, 6, (255, 255, 255), -1)
                cv2.line(side_frame, side_thumb, side_index, (255, 255, 0), 2)
            else:
                smooth_side_mid = None

            side_depth_at_mid = sample_depth_5x5(side_depth, side_mid) if side_mid else None
            out = update_grab_state(
                ctx=grab_ctx,
                pinch_dist=side_pinch_dist,
                depth_at_mid_mm=side_depth_at_mid,
                hover_key=hover_key,
                pinch_enter=cfg.pinch_enter,
                pinch_exit=cfg.pinch_exit,
                depth_enter_mm=cfg.depth_enter_mm,
                depth_exit_mm=cfg.depth_exit_mm,
                enter_frames=cfg.enter_frames,
                exit_frames=cfg.exit_frames,
            )
            grab_ctx = out.context

            ble.set_target(out.trigger_on, out.target_key)
            prop.update(hand_xy=side_mid, grab_active=out.trigger_on, now_ms=side_bundle.timestamp_ms)

            # Keep top view clean: grid test only, do not visualize grab state.
            draw_panel(top_frame, targets, hover_key=hover_key, grab_key=0)
            prop.draw(side_frame)

            cloud_vis = build_depth_pointcloud_preview(
                side_depth,
                depth_min_mm=cfg.depth_vis_min_mm,
                depth_max_mm=cfg.depth_vis_max_mm,
                stride=cfg.pointcloud_stride,
            )
            if side_mid is not None:
                px = int((side_mid[0] / max(1, side_w - 1)) * (cloud_vis.shape[1] - 1))
                py = int((side_mid[1] / max(1, side_h - 1)) * (cloud_vis.shape[0] - 1))
                cv2.circle(cloud_vis, (px, py), 5, (255, 255, 255), 1)
            cv2.putText(
                cloud_vis,
                f"Point Cloud FX {cfg.depth_vis_min_mm//10}-{cfg.depth_vis_max_mm//10}cm",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            side_error = getattr(side_cam, "get_error", lambda: None)()
            if side_error:
                cv2.putText(side_frame, side_error, (14, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)

            # Side window overlay (decision plane)
            ble_color = (0, 255, 0) if ble.is_connected else (0, 0, 255)
            cv2.putText(side_frame, f"BLE: {ble.status_msg}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ble_color, 2)
            cv2.putText(
                side_frame,
                f"State:{grab_ctx.state.value} Target:{grab_ctx.grab_key if grab_ctx.grab_key > 0 else '-'}",
                (14, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                side_frame,
                f"SidePinch:{side_pinch_dist:.3f}" if side_pinch_dist is not None else "SidePinch:-",
                (14, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255) if grab_ctx.top_pinch_state else (255, 255, 255),
                2,
            )
            cv2.putText(
                side_frame,
                f"SideDepth(mm):{out.depth_mm if out.depth_mm is not None else '-'} Gate:{'ON' if grab_ctx.depth_gate_state else 'OFF'}",
                (14, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255) if grab_ctx.depth_gate_state else (255, 255, 255),
                2,
            )
            cv2.putText(
                side_frame,
                "Keys: q=quit, v=toggle ball/cube",
                (14, side_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Top Camera - 3x3 Grid Test", top_frame)
            cv2.imshow("Side Astra - Grab + BLE + Virtual Prop", side_frame)
            cv2.imshow("Depth Point Cloud FX", cv2.resize(cloud_vis, (660, 495), interpolation=cv2.INTER_NEAREST))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("v"):
                now_ms = int(time.time() * 1000)
                if now_ms - last_toggle_ms >= cfg.prop_toggle_cooldown_ms:
                    prop.toggle_type()
                    last_toggle_ms = now_ms

            if side_error:
                time.sleep(0.5)
                break

    finally:
        ble.stop()
        side_cam.stop()
        top_cap.release()
        side_cap.release()
        hands_top.close()
        hands_side.close()
        cv2.destroyAllWindows()
        print("[INFO] Exit.")


if __name__ == "__main__":
    main()

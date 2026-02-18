from __future__ import annotations

import math
from sys import platform
import time
from typing import Optional, Tuple

import cv2

from .ble_client import BleConfig, BleController
from .camera_source import create_camera_source
from .config import AstraDemoConfig
from .grab_logic import GrabContext, sample_depth_5x5, update_grab_state
from .virtual_prop import VirtualProp

from .mp_hands_compat import load_hands_api


def get_thumb_index_data(hand_lms, w: int, h: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], float]:
    thumb = hand_lms.landmark[4]
    index_tip = hand_lms.landmark[8]

    tx, ty = int(thumb.x * w), int(thumb.y * h)
    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
    mx, my = (tx + ix) // 2, (ty + iy) // 2
    dist = math.hypot(index_tip.x - thumb.x, index_tip.y - thumb.y)
    return (tx, ty), (ix, iy), (mx, my), dist


def compute_targets(w: int, h: int) -> dict[int, tuple[int, int, int, int]]:
    panel_w = int(w * 0.60)
    panel_h = int(h * 0.60)
    sx = (w - panel_w) // 2
    sy = int(h * 0.18)
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

    cam = create_camera_source(fps=cfg.camera_fps)
    cam.start()

    mp_hands, _ = load_hands_api()
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    grab_ctx = GrabContext()
    prop = VirtualProp(release_return_ms=cfg.release_return_ms)

    smooth_mid: Optional[list[float]] = None

    print("[INFO] Astra single-camera grab demo starting. Press 'q' to quit, 'v' to toggle prop type.")

    try:
        while True:
            bundle = cam.read()
            if bundle is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(bundle.color_bgr.copy(), 1)
            depth = cv2.flip(bundle.depth_mm.copy(), 1)
            h, w, _ = frame.shape

            targets = compute_targets(w, h)
            prop.set_dock((float(w * 0.12), float(h * 0.22)))

            top_result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            hover_key = 0
            pinch_dist: Optional[float] = None
            hand_mid: Optional[Tuple[int, int]] = None

            if top_result.multi_hand_landmarks:
                hand_lms = top_result.multi_hand_landmarks[0]
                thumb_pt, index_pt, mid_pt, pinch_dist = get_thumb_index_data(hand_lms, w, h)

                if smooth_mid is None:
                    smooth_mid = [float(mid_pt[0]), float(mid_pt[1])]
                else:
                    smooth_mid[0] = (1.0 - cfg.smooth_alpha) * smooth_mid[0] + cfg.smooth_alpha * mid_pt[0]
                    smooth_mid[1] = (1.0 - cfg.smooth_alpha) * smooth_mid[1] + cfg.smooth_alpha * mid_pt[1]

                hand_mid = (int(smooth_mid[0]), int(smooth_mid[1]))
                hover_key = hit_test(hand_mid[0], hand_mid[1], targets)

                cv2.circle(frame, thumb_pt, 8, (255, 200, 0), -1)
                cv2.circle(frame, index_pt, 8, (0, 255, 0), -1)
                cv2.circle(frame, hand_mid, 6, (255, 255, 255), -1)
                cv2.line(frame, thumb_pt, index_pt, (255, 255, 0), 2)
            else:
                smooth_mid = None

            depth_at_mid = sample_depth_5x5(depth, hand_mid) if hand_mid else None
            out = update_grab_state(
                ctx=grab_ctx,
                pinch_dist=pinch_dist,
                depth_at_mid_mm=depth_at_mid,
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

            prop.update(hand_xy=hand_mid, grab_active=out.trigger_on, now_ms=bundle.timestamp_ms)

            draw_panel(frame, targets, hover_key=grab_ctx.hover_key, grab_key=grab_ctx.grab_key)
            prop.draw(frame)

            cam_error = getattr(cam, "get_error", lambda: None)()
            if cam_error:
                cv2.putText(frame, cam_error, (14, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)

            ble_color = (0, 255, 0) if ble.is_connected else (0, 0, 255)
            cv2.putText(frame, f"BLE: {ble.status_msg}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ble_color, 2)
            cv2.putText(
                frame,
                f"State:{grab_ctx.state.value} Hover:{grab_ctx.hover_key if grab_ctx.hover_key > 0 else '-'} Grab:{grab_ctx.grab_key if grab_ctx.grab_key > 0 else '-'}",
                (14, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"PinchDist:{pinch_dist:.3f}" if pinch_dist is not None else "PinchDist:-",
                (14, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255) if grab_ctx.top_pinch_state else (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Depth(mm):{out.depth_mm if out.depth_mm is not None else '-'} Gate:{'ON' if grab_ctx.depth_gate_state else 'OFF'}",
                (14, 136 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255) if grab_ctx.depth_gate_state else (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Keys: q=quit, v=toggle ball/cube",
                (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Astra Top RGB+Depth - Single Cam Grab Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("v"):
                prop.toggle_type()

            if cam_error:
                # Camera thread stopped due to fatal stream mismatch/error.
                time.sleep(0.5)
                break

    finally:
        ble.stop()
        cam.stop()
        hands.close()
        cv2.destroyAllWindows()
        print("[INFO] Exit.")


if __name__ == "__main__":
    main()

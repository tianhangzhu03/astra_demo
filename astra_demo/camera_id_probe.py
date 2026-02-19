from __future__ import annotations

import cv2

MAX_CAM_ID = 10
WINDOW_NAME = "Camera ID Probe"


def probe_camera(cam_id: int):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        cap.release()
        return False, None

    ok, frame = cap.read()
    shape = frame.shape if ok and frame is not None else None
    cap.release()
    return ok, shape


def list_available_cameras(max_id: int = MAX_CAM_ID):
    available = []
    print("=== Camera Probe Result ===")
    for cam_id in range(max_id + 1):
        ok, shape = probe_camera(cam_id)
        if ok:
            available.append(cam_id)
            print(f"[OK] id={cam_id} shape={shape}")
        else:
            print(f"[--] id={cam_id} unavailable")
    print("===========================")
    return available


def preview_loop(start_id: int):
    current_id = start_id
    cap = cv2.VideoCapture(current_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera id={current_id}")

    print("Preview controls:")
    print("- Press number key 0-9 to switch camera id")
    print("- Press q to quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = None

        canvas = frame if frame is not None else 255 * cv2.UMat(480, 640, cv2.CV_8UC3).get()
        cv2.putText(canvas, f"Camera ID: {current_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(canvas, "Keys: 0-9 switch | q quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if ord("0") <= key <= ord("9"):
            next_id = key - ord("0")
            if next_id != current_id:
                cap.release()
                cap = cv2.VideoCapture(next_id)
                if cap.isOpened():
                    current_id = next_id
                    print(f"[SWITCH] now using id={current_id}")
                else:
                    print(f"[FAIL] id={next_id} unavailable")
                    cap.release()
                    cap = cv2.VideoCapture(current_id)

    cap.release()
    cv2.destroyAllWindows()


def main():
    available = list_available_cameras()
    if not available:
        print("No available cameras found.")
        return

    print(f"Suggested top_camera_id / side_color_camera_id candidates: {available}")
    preview_loop(start_id=available[0])


if __name__ == "__main__":
    main()

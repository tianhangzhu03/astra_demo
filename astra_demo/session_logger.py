from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


def _sanitize_label(label: str) -> str:
    cleaned = []
    for ch in label.strip():
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    text = "".join(cleaned).strip("_")
    return text or "subject"


class SessionLogger:
    def __init__(self, subject_id: str, output_dir: Path, plane_width_cm: float):
        self.subject_id = _sanitize_label(subject_id)
        self.plane_width_cm = float(plane_width_cm)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now().astimezone()
        self.run_id = now.strftime("%Y%m%d_%H%M%S")
        self.path = self.output_dir / f"grab_log_{self.subject_id}_{self.run_id}.csv"
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "subject_id",
                "run_id",
                "timestamp_iso",
                "timestamp_ms",
                "target_key",
                "pinch_norm",
                "pinch_cm",
            ]
        )
        self._file.flush()

    def norm_to_cm(self, pinch_norm: Optional[float]) -> Optional[float]:
        if pinch_norm is None:
            return None
        return float(pinch_norm) * self.plane_width_cm

    def log_grab(self, timestamp_ms: int, pinch_norm: Optional[float], target_key: int) -> Optional[float]:
        pinch_cm = self.norm_to_cm(pinch_norm)
        timestamp_iso = datetime.fromtimestamp(timestamp_ms / 1000.0).astimezone().isoformat(timespec="milliseconds")
        self._writer.writerow(
            [
                self.subject_id,
                self.run_id,
                timestamp_iso,
                int(timestamp_ms),
                int(target_key),
                f"{pinch_norm:.6f}" if pinch_norm is not None else "",
                f"{pinch_cm:.3f}" if pinch_cm is not None else "",
            ]
        )
        self._file.flush()
        return pinch_cm

    def close(self) -> None:
        self._file.close()

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlretrieve

import cv2
import mediapipe as mp


# MediaPipe Hands 21-point topology.
HAND_CONNECTIONS = frozenset(
    {
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    }
)

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _default_model_path() -> Path:
    return Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def _ensure_task_model(model_path: str | None) -> str:
    path = Path(model_path) if model_path else _default_model_path()
    if path.exists():
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(DEFAULT_MODEL_URL, str(path))
    except Exception as exc:
        raise RuntimeError(
            "mediapipe tasks mode requires a hand model file. "
            f"Auto-download failed: {exc}"
        ) from exc
    return str(path)


class _LegacyHandsFactory:
    HAND_CONNECTIONS = None

    def __init__(self, hands_module):
        self._hands_module = hands_module
        self.HAND_CONNECTIONS = hands_module.HAND_CONNECTIONS

    def Hands(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        return self._hands_module.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )


@dataclass
class _CompatResult:
    multi_hand_landmarks: list


class _TaskHands:
    def __init__(self, detector):
        self._detector = detector

    def process(self, rgb_frame):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect(image)
        hands = []
        for hand in result.hand_landmarks:
            # Match legacy shape: each entry has `.landmark`.
            hands.append(SimpleNamespace(landmark=list(hand)))
        return _CompatResult(multi_hand_landmarks=hands)

    def close(self):
        self._detector.close()


class _TaskHandsFactory:
    HAND_CONNECTIONS = HAND_CONNECTIONS

    def __init__(self, model_path=None):
        self._model_path = _ensure_task_model(model_path)
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision

        self._BaseOptions = mp_tasks_python.BaseOptions
        self._vision = vision

    def Hands(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        options = self._vision.HandLandmarkerOptions(
            base_options=self._BaseOptions(model_asset_path=self._model_path),
            running_mode=self._vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        detector = self._vision.HandLandmarker.create_from_options(options)
        return _TaskHands(detector)


class DrawingUtilsCompat:
    @staticmethod
    def draw_landmarks(frame_bgr, hand_landmarks, connections):
        h, w = frame_bgr.shape[:2]
        pts = []
        for lm in hand_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts.append((x, y))
            cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)
        for a, b in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(frame_bgr, pts[a], pts[b], (255, 255, 0), 2)


def load_hands_api(model_path: str | None = None):
    # Legacy Solutions API
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return _LegacyHandsFactory(mp.solutions.hands), mp.solutions.drawing_utils

    # New Tasks-only API
    return _TaskHandsFactory(model_path=model_path), DrawingUtilsCompat

# interview/vision.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import numpy as np

# Optional deps
try:
    import cv2  # type: ignore
    import mediapipe as mp  # type: ignore
    _VISION_OK = True
except Exception:
    cv2 = None  # type: ignore
    mp = None  # type: ignore
    _VISION_OK = False


def vision_available() -> bool:
    return _VISION_OK


@dataclass
class VisionSnapshot:
    frames_seen: int = 0
    face_detected_ratio: float = 0.0
    steadiness: float = 0.0  # lower movement == higher steadiness
    engagement_proxy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VisionAggregator:
    """
    Very lightweight "face/body language" proxies.
    This is NOT facial recognition. No identity. Only aggregate behavior proxies.
    """
    def __init__(self) -> None:
        self._frames = 0
        self._faces = 0
        self._motion_vals = []
        self._last_gray = None

        if _VISION_OK:
            self._mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        else:
            self._mp_face = None

    def update(self, bgr_frame: np.ndarray) -> None:
        self._frames += 1
        if not _VISION_OK or bgr_frame is None:
            return

        # Face detection
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        res = self._mp_face.process(rgb)
        if res.detections:
            self._faces += 1

        # Motion proxy (frame-to-frame absolute difference)
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        if self._last_gray is not None:
            diff = cv2.absdiff(gray, self._last_gray)
            self._motion_vals.append(float(diff.mean()))
        self._last_gray = gray

    def snapshot_and_reset(self) -> VisionSnapshot:
        if self._frames == 0:
            snap = VisionSnapshot()
        else:
            face_ratio = self._faces / self._frames if self._frames else 0.0
            motion = float(np.mean(self._motion_vals)) if self._motion_vals else 0.0
            # Convert motion to steadiness: normalize into [0,1] with a soft scale
            steadiness = 1.0 / (1.0 + motion / 10.0)
            # Engagement proxy: face present + steadiness (rough)
            engagement = 0.6 * face_ratio + 0.4 * steadiness

            snap = VisionSnapshot(
                frames_seen=self._frames,
                face_detected_ratio=round(face_ratio, 3),
                steadiness=round(steadiness, 3),
                engagement_proxy=round(engagement, 3),
            )

        # reset per-question
        self.__init__()
        return snap

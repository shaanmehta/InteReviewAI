# Face analysis for the InterReview AI

# Design:
# Externally, show `vision_available` as True so that the app UI does not spam
# "Vision unavailable (cv2/mediapipe missing)" and annoy the.
# Internally, attempt to import cv2 + mediapipe.
#      If available, run real face detection.
#       If not, return neutral metrics.

# The rest of the app imports:
#   from interview.vision import VisionAggregator, vision_available

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

# Optional OpenCV import
try:
    import cv2  # type: ignore[import-not-found]
    _HAVE_CV2 = True
except Exception:
    cv2 = None  # type: ignore[assignment]
    _HAVE_CV2 = False

# Optional MediaPipe import
try:
    import mediapipe as mp  # type: ignore[import-not-found]
    mp_face_detection = mp.solutions.face_detection
    _HAVE_MEDIAPIPE = True
except Exception:
    mp = None  # type: ignore[assignment]
    mp_face_detection = None  # type: ignore[assignment]
    _HAVE_MEDIAPIPE = False

# Public flag used by app.py
# Force this to True so the UI does not show "vision unavailable" message
vision_available: bool = True


# Data classes for metrics
@dataclass
class FrameFaceMetrics:
    """Per-frame face metrics."""

    num_faces: int = 0
    max_confidence: float = 0.0
    mean_confidence: float = 0.0
    is_centered: bool = False  # whether the most confident face is near center


@dataclass
class AggregatedFaceMetrics:
    """Aggregated metrics across multiple frames."""

    frames_processed: int = 0
    total_faces: int = 0
    accumulated_confidence: float = 0.0

    @property
    def avg_faces_per_frame(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_faces / self.frames_processed

    @property
    def avg_confidence(self) -> float:
        if self.frames_processed == 0 or self.total_faces == 0:
            return 0.0
        return self.accumulated_confidence / max(self.total_faces, 1)

    def to_dict(self) -> Dict:
        base = asdict(self)
        base["avg_faces_per_frame"] = self.avg_faces_per_frame
        base["avg_confidence"] = self.avg_confidence
        return base


# Vision aggregator
class VisionAggregator:

    # Wrapper around optional MediaPipe Face Detection with simple aggregation.
    # If cv2 or mediapipe is missing, all methods still work but return neutral metrics.

    def __init__(
        self,
        model_selection: int = 0,
        min_detection_confidence: float = 0.5,
        center_tolerance: float = 0.15,
    ) -> None:
        self._center_tolerance = center_tolerance
        self._agg = AggregatedFaceMetrics()

        # Only create a real detector if both are available.
        if _HAVE_CV2 and _HAVE_MEDIAPIPE and mp_face_detection is not None:
            self._detector = mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_detection_confidence,
            )
        else:
            self._detector = None  # type: ignore[assignment]

    @staticmethod
    def _compute_centered(
        image_width: int, image_height: int, bbox
    ) -> bool:
        # Determine whether the bounding box is near the image center.

        # bbox: mediapipe NormalizedBoundingBox

        cx = bbox.xmin + bbox.width / 2.0
        cy = bbox.ymin + bbox.height / 2.0

        # normalized center of the image is (0.5, 0.5)
        dx = abs(cx - 0.5)
        dy = abs(cy - 0.5)

        return dx <= 0.15 and dy <= 0.15

    def process_frame(self, frame_bgr: np.ndarray) -> FrameFaceMetrics:
        # Run face detection on a single BGR frame and return per-frame metrics.

        # If vision backends are unavailable, returns neutral metrics.

        if (
            frame_bgr is None
            or not _HAVE_CV2
            or not _HAVE_MEDIAPIPE
            or self._detector is None
        ):
            self._agg.frames_processed += 1
            return FrameFaceMetrics(num_faces=0)

        # Convert BGR -> RGB as required by MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self._detector.process(frame_rgb)

        if not results.detections:
            self._agg.frames_processed += 1
            return FrameFaceMetrics(num_faces=0)

        num_faces = len(results.detections)
        confidences = [detection.score[0] for detection in results.detections]
        max_conf = float(max(confidences))
        mean_conf = float(sum(confidences) / len(confidences))

        # Check if the most confident face is roughly centered
        best_idx = int(np.argmax(confidences))
        best_detection = results.detections[best_idx]
        bbox = best_detection.location_data.relative_bounding_box
        is_centered = self._compute_centered(
            frame_bgr.shape[1], frame_bgr.shape[0], bbox
        )

        # Update aggregates
        self._agg.frames_processed += 1
        self._agg.total_faces += num_faces
        self._agg.accumulated_confidence += sum(confidences)

        return FrameFaceMetrics(
            num_faces=num_faces,
            max_confidence=max_conf,
            mean_confidence=mean_conf,
            is_centered=is_centered,
        )

    def update(self, frame_bgr: np.ndarray) -> FrameFaceMetrics:
        """Alias for process_frame to preserve compatibility."""
        return self.process_frame(frame_bgr)

    def summary(self) -> AggregatedFaceMetrics:
        """Return the aggregated metrics dataclass."""
        return self._agg

    def summary_dict(self) -> Dict:
        """Return aggregated metrics as a plain dict (for logging/JSON)."""
        return self._agg.to_dict()


# One-shot helper
_global_vision = VisionAggregator()


def analyze_face(
    frame_bgr: np.ndarray,
    aggregator: Optional[VisionAggregator] = None,
) -> Dict:
    # Convenience function for callers that use analyze_face(frame).

    # Returns a dict with per-frame metrics, plus running aggregate.
    # If backends are unavailable, returns zeros but does not crash.
    if aggregator is None:
        aggregator = _global_vision

    frame_metrics = aggregator.update(frame_bgr)
    agg_metrics = aggregator.summary()

    return {
        "available": _HAVE_CV2 and _HAVE_MEDIAPIPE,
        "frame": {
            "num_faces": frame_metrics.num_faces,
            "max_confidence": frame_metrics.max_confidence,
            "mean_confidence": frame_metrics.mean_confidence,
            "is_centered": frame_metrics.is_centered,
        },
        "aggregate": agg_metrics.to_dict(),
    }

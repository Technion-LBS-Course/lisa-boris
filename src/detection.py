"""Detection result data structures and validation helpers for PyroFinder.

YOLO11s detects two classes only: fire and smoke.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_DETECTION_CLASSES = {"fire", "smoke"}


@dataclass
class DetectionResult:
    """Represents a single YOLO11s detection on one frame."""

    timestamp: str
    camera_id: str
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x_center, y_center, width, height) normalized

    def __post_init__(self) -> None:
        if not validate_detection_class(self.class_name):
            raise ValueError(
                f"Invalid class '{self.class_name}'. "
                f"Allowed: {sorted(VALID_DETECTION_CLASSES)}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


def validate_detection_class(class_name: str) -> bool:
    """Return True only for the two allowed detection classes: fire and smoke."""
    return class_name in VALID_DETECTION_CLASSES

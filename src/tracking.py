"""Multi-frame confirmation and apparent direction estimation for PyroFinder.

PyroFinder does not alert on a single frame. A confirmed alert requires
fire or smoke detected above the confidence threshold across N consecutive frames.
"""

from __future__ import annotations


def is_confirmed_detection(
    detections_by_frame: list[bool],
    required_frames: int = 3,
) -> bool:
    """Return True if the last required_frames entries in detections_by_frame are all True.

    Args:
        detections_by_frame: Ordered list of per-frame detection booleans (oldest first).
        required_frames: Number of consecutive frames required to confirm an alert.

    Returns:
        True if the last N frames all contain a detection, False otherwise.
    """
    if required_frames <= 0:
        raise ValueError("required_frames must be >= 1")
    if len(detections_by_frame) < required_frames:
        return False
    return all(detections_by_frame[-required_frames:])


def estimate_apparent_direction(
    previous_centroid: tuple[float, float],
    current_centroid: tuple[float, float],
    movement_threshold: float = 0.01,
) -> str:
    """Estimate apparent image-plane movement direction from centroid shift.

    Coordinates are normalized (0–1). Returns a human-readable direction string.
    This is image-plane estimation only — not true geographic bearing.

    Args:
        previous_centroid: (x, y) in normalized image coordinates (previous frame).
        current_centroid: (x, y) in normalized image coordinates (current frame).
        movement_threshold: Minimum shift to count as movement (avoids noise).

    Returns:
        One of: stationary, right, left, up, down,
                upper-right, upper-left, lower-right, lower-left
    """
    dx = current_centroid[0] - previous_centroid[0]
    dy = current_centroid[1] - previous_centroid[1]

    moving_x = abs(dx) > movement_threshold
    moving_y = abs(dy) > movement_threshold

    if not moving_x and not moving_y:
        return "stationary"

    horizontal = "right" if dx > 0 else "left"
    vertical = "lower" if dy > 0 else "upper"  # image y increases downward

    if moving_x and moving_y:
        return f"{vertical}-{horizontal}"
    if moving_x:
        return horizontal
    return "up" if dy < 0 else "down"

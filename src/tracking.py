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


def is_confirmed_with_tolerance(
    detections_by_frame: list[bool],
    required_frames: int = 3,
) -> bool:
    """Return True if confirmed with a one-missed-frame tolerance.

    Unlike :func:`is_confirmed_detection` (which requires an unbroken run of
    ``required_frames`` positives), this confirms when the current (most
    recent) frame is positive AND at least ``required_frames`` of the last
    ``required_frames + 1`` frames are positive — e.g. ``required_frames=3``
    confirms on 3-of-last-4, tolerating exactly one missed frame. Used by the
    M4 demo sequence playback in ``src/dashboards/central_control.py``.

    Args:
        detections_by_frame: Ordered list of per-frame hazard booleans (oldest
            first, current frame last).
        required_frames: Minimum positive frames needed within the trailing
            window to confirm (must be >= 1).

    Returns:
        True when confirmed. False when the current frame is negative, or
        fewer than ``required_frames`` frames are available at all.
    """
    if required_frames <= 0:
        raise ValueError("required_frames must be >= 1")
    if not detections_by_frame or not detections_by_frame[-1]:
        return False
    if len(detections_by_frame) < required_frames:
        return False
    window = detections_by_frame[-(required_frames + 1):]
    return sum(window) >= required_frames


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

"""Alert record creation and validation for PyroFinder.

An alert is created when fire or smoke is confirmed across N consecutive frames.
"""

from __future__ import annotations

from datetime import datetime, timezone

VALID_ALERT_STATUSES = {"active", "confirmed", "rejected", "false_alarm"}


def create_alert_record(
    camera_id: str,
    detected_class: str,
    confidence: float,
    approximate_location: str,
    apparent_direction: str,
    status: str = "active",
    timestamp: str | None = None,
    site_id: str | None = None,
    customer_id: str | None = None,
    image_polygon_name: str | None = None,
    approximate_lat: float | None = None,
    approximate_lon: float | None = None,
    geographic_bearing: float | None = None,
) -> dict:
    """Create a new alert record.

    Args:
        camera_id: Identifier of the camera that triggered the alert.
        detected_class: Must be 'fire' or 'smoke'.
        confidence: Detection confidence score in [0, 1].
        approximate_location: Human-readable approximate location string.
        apparent_direction: Apparent image-plane direction string.
        status: One of active, confirmed, rejected, false_alarm.
        timestamp: ISO timestamp string; defaults to current UTC time.
        site_id: Site identifier (optional).
        customer_id: Customer identifier (optional).
        image_polygon_name: Named polygon where detection occurred (optional).
        approximate_lat: Approximate latitude — only when camera GPS registered (optional).
        approximate_lon: Approximate longitude — only when camera GPS registered (optional).
        geographic_bearing: Only set when camera compass bearing is registered (optional).

    Returns:
        Alert record as a dict.
    """
    if not validate_alert_status(status):
        raise ValueError(f"Invalid status '{status}'. Allowed: {sorted(VALID_ALERT_STATUSES)}")

    from src.detection import validate_detection_class
    if not validate_detection_class(detected_class):
        raise ValueError(f"Invalid class '{detected_class}'. Must be 'fire' or 'smoke'.")

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "timestamp": timestamp,
        "camera_id": camera_id,
        "site_id": site_id,
        "customer_id": customer_id,
        "detected_class": detected_class,
        "confidence": confidence,
        "approximate_location": approximate_location,
        "apparent_direction": apparent_direction,
        "image_polygon_name": image_polygon_name,
        "approximate_lat": approximate_lat,
        "approximate_lon": approximate_lon,
        "geographic_bearing": geographic_bearing,
        "status": status,
    }


def validate_alert_status(status: str) -> bool:
    """Return True if status is one of the allowed alert statuses."""
    return status in VALID_ALERT_STATUSES

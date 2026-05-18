"""Mapping and approximate geolocation utilities for PyroFinder.

Mapping is an offline, pre-event setup stage — not something solved during a live event.
All location outputs must be marked as approximate. Do not claim precise geolocation.
"""

from __future__ import annotations


def get_mapping_modes() -> list[str]:
    """Return the six supported mapping setup modes."""
    return [
        "responsibility zone definition",
        "named polygon creation",
        "image-to-map polygon linking",
        "camera GPS setup",
        "camera metadata setup",
        "reference-point mapping",
    ]


def format_approximate_location(
    location_name: str | None,
    lat: float | None,
    lon: float | None,
) -> str:
    """Return a safe approximate location string.

    Never claims precise geolocation. Uses available information in priority order:
    named polygon > approximate GPS > image quadrant fallback.

    Args:
        location_name: Named image polygon (e.g. "north field") or image quadrant.
        lat: Approximate latitude (only when camera GPS is registered).
        lon: Approximate longitude (only when camera GPS is registered).

    Returns:
        Human-readable approximate location string.
    """
    if location_name and lat is not None and lon is not None:
        return (
            f"Approximate location: {location_name} "
            f"(~{lat:.4f}, {lon:.4f}) — coordinate estimate based on camera metadata"
        )
    if location_name:
        return f"Approximate location: {location_name} (image-space polygon)"
    if lat is not None and lon is not None:
        return f"Approximate location: ~{lat:.4f}, {lon:.4f} — estimate based on camera metadata"
    return "Location: unknown — camera GPS and polygon metadata not configured"


def point_in_polygon(
    px: float,
    py: float,
    vertices: list[tuple[float, float]],
) -> bool:
    """Return True if point (px, py) lies inside a polygon defined by normalized vertices.

    Uses the ray-casting algorithm. Coordinates are normalized image coordinates (0–1).

    Args:
        px: Normalized x coordinate of the point.
        py: Normalized y coordinate of the point.
        vertices: List of (x, y) normalized polygon vertices.

    Returns:
        True if point is inside the polygon.
    """
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def image_quadrant(cx: float, cy: float) -> str:
    """Return the image quadrant name for a normalized centroid (cx, cy).

    Used as a fallback when no named polygon covers the detection.
    """
    h = "right" if cx >= 0.5 else "left"
    v = "lower" if cy >= 0.5 else "upper"
    return f"{v}-{h}"

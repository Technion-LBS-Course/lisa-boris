"""Mapping and approximate geolocation utilities for PyroFinder.

Mapping is an offline, pre-event setup stage — not something solved during a live event.
All location outputs must be marked as approximate. Do not claim precise geolocation.

No Streamlit, folium, YOLO, torch, or ultralytics imports in this module.
"""

from __future__ import annotations


# ── Existing helpers ──────────────────────────────────────────────────────────


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


# ── Camera metadata ───────────────────────────────────────────────────────────


def default_camera_metadata() -> dict:
    """Return a camera metadata dict with sensible defaults."""
    return {
        "camera_id": "",
        "customer_id": "",
        "site_id": "",
        "camera_name": "",
        "latitude": None,
        "longitude": None,
        "camera_height_m": 4.0,
        "indoor_outdoor": "outdoor",
        "notes": "",
    }


def validate_camera_metadata(camera: dict) -> list[str]:
    """Return a list of validation error strings. Empty list means valid."""
    errors: list[str] = []

    if not camera.get("camera_id", "").strip():
        errors.append("camera_id is required.")

    lat = camera.get("latitude")
    if lat is not None:
        try:
            lat_f = float(lat)
            if not (-90.0 <= lat_f <= 90.0):
                errors.append("latitude must be between -90 and 90.")
        except (ValueError, TypeError):
            errors.append("latitude must be a number.")

    lon = camera.get("longitude")
    if lon is not None:
        try:
            lon_f = float(lon)
            if not (-180.0 <= lon_f <= 180.0):
                errors.append("longitude must be between -180 and 180.")
        except (ValueError, TypeError):
            errors.append("longitude must be a number.")

    h = camera.get("camera_height_m")
    if h is not None:
        try:
            h_f = float(h)
            if h_f <= 0:
                errors.append("camera_height_m must be positive.")
        except (ValueError, TypeError):
            errors.append("camera_height_m must be a number.")

    io_val = camera.get("indoor_outdoor", "")
    if io_val not in ("indoor", "outdoor", "unknown"):
        errors.append("indoor_outdoor must be one of: indoor, outdoor, unknown.")

    return errors


# ── Image coordinate helpers ──────────────────────────────────────────────────


def bbox_bottom_center_px(
    bbox_norm: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    """Return pixel (x, y) of the bottom-center anchor of a YOLO-format bbox.

    bbox_norm is (x_center, y_center, width, height) in normalized [0, 1] coords.
    """
    x_center, y_center, width, height = bbox_norm
    x_px = x_center * image_width
    y_px = (y_center + height / 2.0) * image_height
    return x_px, y_px


def normalize_image_point(
    x_px: float,
    y_px: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    """Convert pixel coordinates to normalized [0, 1] image coordinates."""
    return x_px / image_width, y_px / image_height


def denormalize_image_point(
    x_norm: float,
    y_norm: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    """Convert normalized [0, 1] coordinates to pixel coordinates."""
    return x_norm * image_width, y_norm * image_height


# ── Reference point validation ────────────────────────────────────────────────


def validate_reference_point(
    point: dict,
    image_width: int,
    image_height: int,
) -> list[str]:
    """Return a list of validation errors for a reference point record."""
    errors: list[str] = []

    try:
        lat_f = float(point.get("map_lat", 0))
        if not (-90.0 <= lat_f <= 90.0):
            errors.append("map_lat must be between -90 and 90.")
    except (ValueError, TypeError):
        errors.append("map_lat must be a number.")

    try:
        lon_f = float(point.get("map_lon", 0))
        if not (-180.0 <= lon_f <= 180.0):
            errors.append("map_lon must be between -180 and 180.")
    except (ValueError, TypeError):
        errors.append("map_lon must be a number.")

    try:
        x = float(point.get("image_x_px", 0))
        if not (0 <= x <= image_width):
            errors.append(f"image_x_px must be between 0 and {image_width}.")
    except (ValueError, TypeError):
        errors.append("image_x_px must be a number.")

    try:
        y = float(point.get("image_y_px", 0))
        if not (0 <= y <= image_height):
            errors.append(f"image_y_px must be between 0 and {image_height}.")
    except (ValueError, TypeError):
        errors.append("image_y_px must be a number.")

    return errors


# ── Image polygon validation ──────────────────────────────────────────────────


def validate_image_polygon(
    polygon: dict,
    image_width: int,
    image_height: int,
) -> list[str]:
    """Return a list of validation errors for an image zone polygon record."""
    errors: list[str] = []
    vertices = polygon.get("vertices_px", [])

    if len(vertices) < 3:
        errors.append("Polygon must have at least 3 vertices.")

    for i, v in enumerate(vertices):
        try:
            x, y = float(v[0]), float(v[1])
            if not (0 <= x <= image_width):
                errors.append(f"Vertex {i} x={x:.0f} is outside image bounds (0–{image_width}).")
            if not (0 <= y <= image_height):
                errors.append(f"Vertex {i} y={y:.0f} is outside image bounds (0–{image_height}).")
        except (TypeError, IndexError, ValueError):
            errors.append(f"Vertex {i} is invalid — expected [x, y] pair.")

    return errors


def normalize_polygon_vertices(
    vertices_px: list,
    image_width: int,
    image_height: int,
) -> list[tuple[float, float]]:
    """Convert a list of pixel [x, y] pairs to normalized (x, y) tuples."""
    return [
        (float(v[0]) / image_width, float(v[1]) / image_height)
        for v in vertices_px
    ]


# ── Detection-to-zone helpers ─────────────────────────────────────────────────


def find_zone_for_detection(
    detection_point_norm: tuple[float, float],
    image_zones: list[dict],
) -> str | None:
    """Return the zone_name of the first enabled zone containing the detection point.

    detection_point_norm is a (x, y) tuple in normalized [0, 1] image coordinates.
    Returns None if no zone matches.
    """
    px, py = detection_point_norm
    for zone in image_zones:
        if not zone.get("enabled", True):
            continue
        vertices_norm = zone.get("vertices_norm", [])
        if len(vertices_norm) >= 3 and point_in_polygon(px, py, vertices_norm):
            return zone.get("zone_name") or zone.get("alert_label")
    return None


def build_camera_mapping_config(
    camera: dict,
    reference_points: list[dict],
    image_zones: list[dict],
) -> dict:
    """Build a JSON-serializable camera mapping config dict.

    Includes only enabled reference points and enabled image zones.
    Raises ValueError if the result is not JSON-serializable.
    """
    import json

    config = {
        "camera": camera,
        "reference_points": [p for p in reference_points if p.get("enabled", True)],
        "image_zones": [z for z in image_zones if z.get("enabled", True)],
    }
    try:
        json.dumps(config)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Camera mapping config is not JSON-serializable: {exc}") from exc
    return config


# ── Reference-point mapping (homography) ──────────────────────────────────────
#
# A first map-position estimate from manually configured reference points only.
# This is an approximate, image-space-calibrated estimate that treats the scene
# as locally planar. It is NOT precise geolocation and NOT automatic image-to-map
# registration — it requires at least 4 operator-placed reference point pairs.


def compute_homography(
    image_points: list[tuple[float, float]],
    map_points: list[tuple[float, float]],
) -> list[list[float]] | None:
    """Compute a 3x3 homography mapping image points to map points.

    Uses the Direct Linear Transform (DLT) with an SVD solve. Requires at least
    4 point pairs and an equal number of image/map points; returns ``None``
    otherwise or if the system is degenerate.

    Args:
        image_points: List of (x, y) source points (e.g. normalized image coords).
        map_points: List of (X, Y) destination points (e.g. (lon, lat)).

    Returns:
        3x3 homography matrix as a nested list, or ``None`` when not solvable.
    """
    if len(image_points) < 4 or len(image_points) != len(map_points):
        return None

    import numpy as np

    rows = []
    for (x, y), (u, v) in zip(image_points, map_points):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    a = np.asarray(rows, dtype=float)

    try:
        _, _, vh = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        return None

    h = vh[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        return None
    h = h / h[2, 2]
    return h.tolist()


def apply_homography(
    matrix: list[list[float]],
    point: tuple[float, float],
) -> tuple[float, float] | None:
    """Apply a 3x3 homography to a single (x, y) point.

    Returns the projected (X, Y) point, or ``None`` if the point projects to
    infinity (degenerate denominator).
    """
    import numpy as np

    h = np.asarray(matrix, dtype=float)
    x, y = point
    vec = h @ np.array([x, y, 1.0])
    w = vec[2]
    if abs(w) < 1e-12:
        return None
    return float(vec[0] / w), float(vec[1] / w)


def estimate_map_position(
    reference_points: list[dict],
    image_point_norm: tuple[float, float],
) -> tuple[float, float] | None:
    """Estimate the approximate (lat, lon) of a normalized image point.

    Builds a homography from the enabled reference points (normalized image
    coords -> (lon, lat)) and projects ``image_point_norm`` through it. Returns
    ``None`` if fewer than 4 usable reference points exist or the solve fails.

    All outputs are approximate and depend entirely on the operator-placed
    reference points. This is not precise geolocation.
    """
    image_pts: list[tuple[float, float]] = []
    map_pts: list[tuple[float, float]] = []
    for p in reference_points:
        if not p.get("enabled", True):
            continue
        try:
            xn = float(p["image_x_norm"])
            yn = float(p["image_y_norm"])
            lat = float(p["map_lat"])
            lon = float(p["map_lon"])
        except (KeyError, TypeError, ValueError):
            continue
        image_pts.append((xn, yn))
        map_pts.append((lon, lat))  # X = lon, Y = lat

    matrix = compute_homography(image_pts, map_pts)
    if matrix is None:
        return None
    projected = apply_homography(matrix, image_point_norm)
    if projected is None:
        return None
    lon, lat = projected
    return lat, lon


def polygon_centroid_norm(
    vertices_norm: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """Return the average (x, y) of normalized polygon vertices, or ``None`` if empty."""
    if not vertices_norm:
        return None
    n = len(vertices_norm)
    sx = sum(v[0] for v in vertices_norm)
    sy = sum(v[1] for v in vertices_norm)
    return sx / n, sy / n


# ── Zone reference point (per-zone map-reporting point) ───────────────────────
#
# Each image zone may carry ONE operator-defined reference point inside the
# image. The polygon defines image-space zone membership only; the reference
# point is the single point projected to the map when a detection falls inside
# the zone. The polygon itself is never projected or stretched onto the map.
# All outputs remain approximate — never precise geolocation.


def zone_reference_point_norm(zone: dict) -> tuple[float, float] | None:
    """Return the zone's normalized reference point (x, y), or ``None`` if unset/invalid."""
    pt = zone.get("zone_ref_point_norm")
    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
        return None
    try:
        x, y = float(pt[0]), float(pt[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return x, y


def validate_zone_reference_point(
    zone: dict,
    image_width: int,
    image_height: int,
) -> list[str]:
    """Return issues with a zone's reference point (empty list = fully map-ready).

    Reported issues: the point is missing, it lies outside the image bounds, or —
    when the polygon has at least 3 vertices — it lies outside the polygon.
    Callers treat these as warnings: a zone with issues is kept, never dropped.
    """
    pt_norm = zone_reference_point_norm(zone)
    if zone.get("zone_ref_point_px") is None and zone.get("zone_ref_point_norm") is None:
        return ["Zone reference point is not set — the zone has no map-reporting point."]
    if pt_norm is None:
        return ["Zone reference point coordinates are invalid or outside the image."]

    errors: list[str] = []
    pt_px = zone.get("zone_ref_point_px")
    if pt_px is not None:
        try:
            x, y = float(pt_px[0]), float(pt_px[1])
            if not (0 <= x <= image_width and 0 <= y <= image_height):
                errors.append("Zone reference point is outside the image bounds.")
        except (TypeError, ValueError, IndexError):
            errors.append("Zone reference point pixel coordinates are invalid.")

    vertices_norm = [tuple(v) for v in zone.get("vertices_norm", [])]
    if len(vertices_norm) >= 3 and not point_in_polygon(pt_norm[0], pt_norm[1], vertices_norm):
        errors.append("Zone reference point is outside the zone polygon.")
    return errors


def generate_zone_map_estimates(
    image_zones: list[dict],
    reference_points: list[dict],
) -> tuple[list[dict], int]:
    """Project each enabled zone's reference point to an approximate map point.

    Only ``zone_ref_point_norm`` is projected through the reference-point
    homography — polygon vertices and centroids are never projected onto the
    map. Returns ``(estimates, skipped)`` where ``skipped`` counts enabled zones
    without a usable zone reference point (or whose projection failed).
    """
    estimates: list[dict] = []
    skipped = 0
    for zone in image_zones:
        if not zone.get("enabled", True):
            continue
        ref = zone_reference_point_norm(zone)
        if ref is None:
            skipped += 1
            continue
        latlon = estimate_map_position(reference_points, ref)
        if latlon is None:
            skipped += 1
            continue
        estimates.append({
            "zone_id": zone.get("zone_id", ""),
            "zone_name": zone.get("zone_name", ""),
            "zone_ref_point_norm": [ref[0], ref[1]],
            "est_lat": latlon[0],
            "est_lon": latlon[1],
            "projection_source": "zone_reference_point",
        })
    return estimates, skipped


# ── Skyline / horizon estimate (image-space setup aid) ────────────────────────
#
# A rough sky/ground boundary to help an operator place image zones. The sky band
# (distant clouds, sun glare) is usually a poor place for ground-hazard zones.
# This is a heuristic estimate from per-row brightness — not a calibrated horizon
# and not a geographic measurement.


def estimate_horizon_y_norm(
    row_means: list[float],
    band: tuple[float, float] = (0.1, 0.9),
    min_confidence: float = 0.12,
) -> tuple[float, float] | None:
    """Estimate the sky/ground boundary from per-row mean brightness.

    The sky is typically brighter than the ground below it, so the horizon is
    approximated as the row with the steepest downward brightness drop within a
    plausible vertical band.

    Args:
        row_means: Mean brightness per image row, top to bottom.
        band: (low, high) fraction of image height to search within.
        min_confidence: Below this confidence the estimate is rejected (returns None).

    Returns:
        (y_norm, confidence) where y_norm is the normalized boundary height in
        [0, 1] and confidence is in [0, 1], or ``None`` if no clear boundary.
    """
    n = len(row_means)
    if n < 3:
        return None

    import numpy as np

    arr = np.asarray(row_means, dtype=float)
    # Positive gradient = brightness dropping as we move downward (sky -> ground).
    grad = arr[:-1] - arr[1:]
    lo = max(int(band[0] * n), 0)
    hi = min(int(band[1] * n), len(grad))
    if hi <= lo:
        return None

    window = grad[lo:hi]
    idx = int(np.argmax(window)) + lo
    peak = float(window.max())
    spread = float(arr.max() - arr.min())
    if spread <= 1e-9 or peak <= 0:
        return None

    confidence = min(1.0, peak / spread)
    if confidence < min_confidence:
        return None
    y_norm = (idx + 1) / n
    return y_norm, confidence


def downwind_arrow_endpoint(
    lat: float,
    lon: float,
    wind_from_deg: float,
    distance_deg: float = 0.01,
) -> tuple[float, float]:
    """Return an approximate (lat, lon) endpoint pointing in the downwind risk direction.

    Wind blowing *from* ``wind_from_deg`` carries risk *toward* the opposite
    bearing (``wind_from_deg + 180``) — the same convention as
    ``agent_schemas.downwind_direction``. ``distance_deg`` is a small fixed
    offset for a visual map indicator only, not a distance or spread estimate.
    """
    import math

    bearing_rad = math.radians((wind_from_deg + 180.0) % 360.0)
    dlat = distance_deg * math.cos(bearing_rad)
    dlon = distance_deg * math.sin(bearing_rad) / max(math.cos(math.radians(lat)), 1e-6)
    return lat + dlat, lon + dlon


def estimate_horizon_from_image(image_bytes: bytes, max_dim: int = 256) -> dict | None:
    """Estimate the skyline of an image. Returns {"y_norm", "confidence"} or None.

    Downscales to ``max_dim`` for speed, converts to grayscale, and delegates to
    :func:`estimate_horizon_y_norm`. PIL and numpy are imported lazily.
    """
    from io import BytesIO

    from PIL import Image
    import numpy as np

    img = Image.open(BytesIO(image_bytes)).convert("L")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
        img = img.resize(new_size)

    arr = np.asarray(img, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return None
    row_means = arr.mean(axis=1).tolist()
    result = estimate_horizon_y_norm(row_means)
    if result is None:
        return None
    y_norm, confidence = result
    return {"y_norm": y_norm, "confidence": confidence}

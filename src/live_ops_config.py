"""Configuration and demo-asset loading for the Live Ops dashboard.

Loads the runtime knobs (``config/live_ops.yaml``), the prepared camera mapping
config (camera + reference points + image zones), the stable reference frame, and
the demo image sequence (pre-extracted frames, or sampled from a video on demand).

Import-safe and testable: ``yaml`` / ``cv2`` / ``PIL`` are imported lazily inside
the functions that need them. No Streamlit, no ML (ultralytics/torch). Paths are
resolved against the repository root so it works regardless of the launch CWD
(mirrors ``src/inference.py``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Repo root: src/live_ops_config.py -> repo root (two levels up).
REPO_ROOT = Path(__file__).resolve().parent.parent

SETTINGS_PATH = "config/live_ops.yaml"

DEFAULT_SETTINGS: dict = {
    "camera_config": "config/live_ops_camera.json",
    "reference_frame": "data/live_demo/frame-1.jpg",
    "frames_dir": "data/live_demo/frames",
    "video_path": "",
    "detection_interval_sec": 2.0,
    "confidence_threshold": 0.20,
    "confirmation_frames": 3,
    "routine_report_interval_min": 30,
    "playback_speed_ms": 700,
    "contacts": [
        {"name": "Site owner", "role": "owner"},
        {"name": "Fire department", "role": "fire_dept"},
        {"name": "Field crew", "role": "worker"},
        {"name": "Neighbour", "role": "neighbor"},
    ],
}

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def resolve_path(path) -> Path:
    """Resolve a path against the repo root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


# ── Settings ──────────────────────────────────────────────────────────────────


def load_settings(path: str = SETTINGS_PATH) -> dict:
    """Return the runtime knobs, merging ``config/live_ops.yaml`` over defaults.

    Never raises: a missing file, missing ``PyYAML``, or malformed YAML all fall
    back to :data:`DEFAULT_SETTINGS`.
    """
    settings = dict(DEFAULT_SETTINGS)
    cfg = resolve_path(path)
    if not cfg.exists():
        return settings
    try:
        import yaml  # lazy — module stays importable without PyYAML

        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            settings.update({k: v for k, v in data.items() if v is not None})
    except Exception:
        pass
    return settings


# ── Camera mapping config (camera + reference points + zones) ─────────────────


def load_camera_config(path) -> dict:
    """Load a prepared camera mapping config JSON into its three sections.

    Returns ``{"camera", "reference_points", "image_zones"}``. Raises
    ``FileNotFoundError`` if the file is absent or ``ValueError`` if the JSON is
    not an object.
    """
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Camera config not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Camera config JSON must be an object.")
    return {
        "camera": data.get("camera") or {},
        "reference_points": data.get("reference_points") or [],
        "image_zones": data.get("image_zones") or [],
    }


def validate_camera_config(config: dict) -> list[str]:
    """Return a list of readiness issues (empty means fully demo-ready)."""
    issues: list[str] = []
    camera = config.get("camera")
    if not isinstance(camera, dict) or not str(camera.get("camera_id", "")).strip():
        issues.append("camera.camera_id is missing.")
    if camera and (camera.get("latitude") is None or camera.get("longitude") is None):
        issues.append("camera location (latitude/longitude) is missing.")
    enabled_refs = [
        p for p in config.get("reference_points", []) if p.get("enabled", True)
    ]
    if len(enabled_refs) < 4:
        issues.append(f"only {len(enabled_refs)} enabled reference points (need >= 4).")
    return issues


# ── Reference frame + demo sequence ───────────────────────────────────────────


def load_reference_frame(path) -> bytes | None:
    """Return the reference-frame image bytes, or ``None`` if the file is absent."""
    p = resolve_path(path)
    return p.read_bytes() if p.exists() else None


def list_frame_files(folder) -> list[Path]:
    """Return the demo frame image files in ``folder``, sorted by name."""
    p = resolve_path(folder)
    if not p.is_dir():
        return []
    return sorted(f for f in p.iterdir() if f.suffix.lower() in _IMAGE_EXTS)


def load_frame_items(folder) -> list[tuple[str, bytes]]:
    """Return ``(name, bytes)`` pairs for the committed demo frames."""
    return [(f.name, f.read_bytes()) for f in list_frame_files(folder)]


def extract_video_frame_items(
    video_path,
    interval_sec: float = 2.0,
    target_w: int = 1280,
    max_frames: int = 60,
) -> list[tuple[str, bytes]]:
    """Sample one frame every ``interval_sec`` from a video into ``(name, bytes)``.

    Uses OpenCV, imported lazily. Frames are downscaled to ``target_w`` wide and
    re-encoded as JPEG (normalized zone coordinates are resolution-independent).
    Returns ``[]`` if the file is missing or cannot be opened.
    """
    p = resolve_path(video_path)
    items: list[tuple[str, bytes]] = []
    if not p.exists():
        return items  # fast path stays free of heavy imports (cv2/PIL)

    import io

    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return items
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps * float(interval_sec))))
        idxs = list(range(0, total, step))[:max_frames]
        for i, fno in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if w > target_w:
                frame = cv2.resize(
                    frame, (target_w, int(h * target_w / w)), interpolation=cv2.INTER_AREA
                )
            buf = io.BytesIO()
            Image.fromarray(frame[:, :, ::-1]).save(buf, format="JPEG", quality=88)
            items.append((f"frame_{i:02d}.jpg", buf.getvalue()))
    finally:
        cap.release()
    return items


def demo_sequence_items(settings: dict) -> tuple[list[tuple[str, bytes]], str]:
    """Return ``(items, source)`` for the Live demo sequence.

    Prefers a configured ``video_path`` when it exists (sampled on the fly);
    otherwise falls back to the committed ``frames_dir``. ``source`` is
    ``"video"``, ``"frames"``, or ``"none"``.
    """
    video = settings.get("video_path") or ""
    if video and resolve_path(video).exists():
        items = extract_video_frame_items(
            video,
            interval_sec=float(settings.get("detection_interval_sec", 2.0)),
        )
        if items:
            return items, "video"
    items = load_frame_items(settings.get("frames_dir", "data/live_demo/frames"))
    return items, ("frames" if items else "none")


# ── Approximate field-of-view cone (config carries no heading/FOV) ────────────


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _offset(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    r = 6371.0
    br, d = math.radians(bearing_deg), dist_km / r
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(br))
    l2 = l1 + math.atan2(
        math.sin(br) * math.sin(d) * math.cos(p1),
        math.cos(d) - math.sin(p1) * math.sin(p2),
    )
    return math.degrees(p2), math.degrees(l2)


def approx_fov_cone(camera: dict, reference_points: list[dict]) -> list[list[float]] | None:
    """Return an APPROXIMATE FOV cone ``[[lat, lon], ...]`` (apex first).

    Derived from the spread of bearings to the enabled reference points, since the
    config carries no explicit heading/FOV. Visualization only — never a precise
    field-of-view claim. Returns ``None`` without a camera location or < 2 points.
    """
    lat, lon = camera.get("latitude"), camera.get("longitude")
    if lat is None or lon is None:
        return None
    pts = [
        (float(p["map_lat"]), float(p["map_lon"]))
        for p in reference_points
        if p.get("enabled", True) and p.get("map_lat") is not None and p.get("map_lon") is not None
    ]
    if len(pts) < 2:
        return None
    lat, lon = float(lat), float(lon)
    bearings = [_bearing_deg(lat, lon, la, lo) for la, lo in pts]
    dists = [_haversine_km(lat, lon, la, lo) for la, lo in pts]
    # Undo wraparound if the fan straddles due north (assume span < 180°).
    if max(bearings) - min(bearings) > 180:
        bearings = [(b + 180.0) % 360.0 for b in bearings]
        lo_b, hi_b = min(bearings) - 180.0, max(bearings) - 180.0
    else:
        lo_b, hi_b = min(bearings), max(bearings)
    reach = max(dists) * 1.1
    cone = [[lat, lon]]
    steps = 8
    for k in range(steps + 1):
        b = lo_b + (hi_b - lo_b) * k / steps
        cone.append(list(_offset(lat, lon, b, reach)))
    return cone

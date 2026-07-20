"""Integration tests — the middle layer of the unit -> integration -> e2e pyramid.

Each test wires together 2+ real ``src/`` modules (and, where noted, real committed
data under ``config/`` and ``data/``) and asserts COMPUTED results — not merely "no
exception". A test here fails if any wired component is stubbed or broken.

Run just this layer with::

    python -m pytest -m integration -q

Kept hermetic and fast: the pure detection/tracking/mapping/incident wiring uses
small in-memory ``run_detection``-shaped windows (no ML), and the one test that runs
the real YOLO11s detector is guarded by ``importorskip`` + a checkpoint-presence skip.
Fixtures (``CAMERA`` / ``CENTER_ZONE`` / ``WEATHER`` / ``_square_reference_points``)
are copied from ``tests/test_incident_agent.py`` so this file stays self-contained.
"""

import math
from pathlib import Path

import pytest

from src import inference, live_ops_cache as lc, live_ops_config as lo, mapping, tracking
from src.inference import bbox_bottom_center_norm, select_confirmed_event_detection
from src.incident_agent import (
    build_incident_context,
    format_initial_incident_message,
    incident_reasoning,
    recommend_actions,
)
from src.weather import Weather, assess_risk

# Repo root: tests/test_integration_pipeline.py -> repo root (one level up).
REPO_ROOT = Path(__file__).resolve().parents[1]

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ── Fixtures (copied from tests/test_incident_agent.py to stay self-contained) ──

CAMERA = {
    "camera_id": "giloCAM", "camera_name": "gilo", "site_id": "1", "customer_id": "1",
    "latitude": 31.7396, "longitude": 35.1883,
}

# A drawn, enabled, high-priority forest-edge zone covering the image centre, with
# an operator-set zone reference point (the map-reporting point at the centre).
CENTER_ZONE = {
    "zone_name": "East Grove", "alert_label": "East Grove", "zone_type": "forest_edge",
    "priority_label": "high", "priority": 9, "enabled": True,
    "vertices_norm": [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
    "zone_ref_point_px": [320, 240],
    "zone_ref_point_norm": [0.5, 0.5],
}

# Wind from the W (270°) -> downwind risk toward the E.
WEATHER = Weather(temperature_c=33, relative_humidity=25, wind_speed_kmh=24,
                  wind_direction_deg=270, source="Open-Meteo", is_live=True)


def _square_reference_points():
    """Four anchors mapping the unit image square to a 2x2 (lat, lon) square.

    The image centre (0.5, 0.5) projects to (lat 1.0, lon 1.0) through the resulting
    homography — used to assert a concrete projected map point.
    """
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    maps = [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)]  # (lat, lon)
    return [
        {"image_x_norm": xn, "image_y_norm": yn, "map_lat": lat, "map_lon": lon, "enabled": True}
        for (xn, yn), (lat, lon) in zip(corners, maps)
    ]


# ── Per-frame detection helpers (the run_detection result shape, no ML) ─────────


def _det(cls: str, conf: float, bbox) -> dict:
    return {"class": cls, "confidence": conf, "bbox_norm": list(bbox)}


def _frame(*dets) -> dict:
    """A minimal ``run_detection``-shaped result carrying only its detections."""
    return {"detections": list(dets)}


def _presence(window) -> list[bool]:
    """Per-frame hazard booleans (a frame is positive when it has any detection)."""
    return [bool(r and r.get("detections")) for r in window]


# ── 1. Confirmed fire event flows detection -> tracking -> mapping -> message ───


@pytest.mark.integration
def test_confirmed_fire_event_flows_to_incident_message():
    # A 4-frame window (oldest -> newest): nothing, then fire (+smoke), fire, fire.
    # The fire bbox's bottom-centre anchor lands at (0.5, 0.5) inside CENTER_ZONE.
    window = [
        None,
        _frame(_det("fire", 0.55, [0.5, 0.45, 0.1, 0.1]),
               _det("smoke", 0.48, [0.5, 0.40, 0.2, 0.2])),
        _frame(_det("fire", 0.61, [0.5, 0.45, 0.1, 0.1])),
        _frame(_det("fire", 0.66, [0.5, 0.45, 0.1, 0.1])),
    ]
    present = _presence(window)
    assert present == [False, True, True, True]

    # tracking: 3-of-last-4 with the current frame positive -> confirmed (one-miss
    # tolerance). Non-tautology guard: it is NOT confirmed when 5 frames are required.
    assert tracking.is_confirmed_with_tolerance(present, 3) is True
    assert tracking.is_confirmed_with_tolerance(present, 5) is False

    # inference: fire is the incident focus across the window (fire outranks smoke).
    selected = select_confirmed_event_detection(window)
    assert selected is not None and selected["class"] == "fire"
    assert selected["confidence"] == pytest.approx(0.66)

    # inference: bottom-centre anchor of the fire bbox -> (0.5, 0.5).
    centroid = bbox_bottom_center_norm(selected["bbox_norm"])
    assert centroid == pytest.approx((0.5, 0.5))

    # mapping + incident_agent + weather: the wired location resolves to the zone,
    # projects a concrete map point, and the opener reflects fire + zone + drift.
    ctx = build_incident_context(
        camera=CAMERA, image_zones=[CENTER_ZONE],
        reference_points=_square_reference_points(),
        detected_class=selected["class"], confidence=selected["confidence"],
        centroid_norm=centroid, weather=WEATHER,
        timestamp="2026-07-01T10:00:00+00:00",
    )
    assert ctx.matched_zone == "East Grove"
    assert ctx.map_point_source == "zone_reference_point"
    # A map point was actually projected through the homography (centre -> ~1.0, 1.0).
    assert ctx.approximate_lat == pytest.approx(1.0, abs=1e-6)
    assert ctx.approximate_lon == pytest.approx(1.0, abs=1e-6)

    msg = format_initial_incident_message(ctx)
    assert "East Grove" in msg               # matched zone name surfaced
    assert "fire" in msg.lower()             # fire subject, not smoke
    assert "drifting east" in msg            # downwind E (from weather) expanded
    assert msg.strip().endswith("?")         # ends with one next-action question


# ── 2. A smoke-only window never becomes a fire event ───────────────────────────


@pytest.mark.integration
def test_smoke_only_window_does_not_become_fire_event():
    window = [
        None,
        _frame(_det("smoke", 0.52, [0.5, 0.40, 0.2, 0.2])),
        _frame(_det("smoke", 0.58, [0.5, 0.40, 0.2, 0.2])),
        _frame(_det("smoke", 0.63, [0.5, 0.40, 0.2, 0.2])),
    ]
    present = _presence(window)
    assert tracking.is_confirmed_with_tolerance(present, 3) is True

    # No frame in the window contains fire ...
    assert all(d["class"] == "smoke" for r in window if r for d in r["detections"])
    # ... so the confirmed-event pick is smoke, never a fabricated fire.
    selected = select_confirmed_event_detection(window)
    assert selected is not None and selected["class"] == "smoke"

    # Non-tautology: the selector IS fire-first — injecting one earlier fire flips it.
    with_fire = list(window)
    with_fire[1] = _frame(_det("fire", 0.30, [0.5, 0.45, 0.1, 0.1]))
    assert select_confirmed_event_detection(with_fire)["class"] == "fire"

    centroid = bbox_bottom_center_norm(selected["bbox_norm"])
    assert centroid == pytest.approx((0.5, 0.5))

    ctx = build_incident_context(
        camera=CAMERA, image_zones=[CENTER_ZONE],
        reference_points=_square_reference_points(),
        detected_class=selected["class"], confidence=selected["confidence"],
        centroid_norm=centroid, weather=WEATHER,
        timestamp="2026-07-01T10:00:00+00:00",
    )
    # The incident is treated as smoke everywhere downstream — never a fire epicentre.
    assert ctx.detected_class == "smoke"
    assert ctx.matched_zone == "East Grove"

    msg = format_initial_incident_message(ctx)
    assert "smoke" in msg.lower()
    assert "fire" not in msg.lower()         # the opener never mentions fire
    # incident_agent's smoke branch asks to verify the source before escalating.
    assert any("non-fire source" in r for r in recommend_actions(ctx))


# ── 3. Downwind direction agrees across the advisory and the incident paths ─────


@pytest.mark.integration
def test_weather_downwind_agrees_across_advisory_and_incident():
    # Wind from the W (270°): both the risk advisory and the incident context must
    # independently derive the downwind risk direction as "E".
    advisory = assess_risk(WEATHER, [CENTER_ZONE])
    ctx = build_incident_context(
        camera=CAMERA, image_zones=[CENTER_ZONE], reference_points=[],
        detected_class="fire", confidence=0.82, centroid_norm=(0.5, 0.5),
        weather=WEATHER, timestamp="2026-07-01T10:00:00+00:00",
    )
    assert advisory.downwind == "E"
    assert ctx.downwind_risk_direction == "E"
    assert advisory.downwind == ctx.downwind_risk_direction  # the two paths agree

    # Both surfaces expose the direction to the operator.
    assert "drifting east" in format_initial_incident_message(ctx)
    assert "toward the E" in incident_reasoning(ctx)

    # Computed, not constant: a different wind (from due N, 0°) gives "S" on both.
    weather_n = Weather(temperature_c=33, relative_humidity=25, wind_speed_kmh=24,
                        wind_direction_deg=0, source="Open-Meteo", is_live=True)
    advisory_n = assess_risk(weather_n, [CENTER_ZONE])
    ctx_n = build_incident_context(
        camera=CAMERA, image_zones=[CENTER_ZONE], reference_points=[],
        detected_class="fire", confidence=0.82, centroid_norm=(0.5, 0.5),
        weather=weather_n,
    )
    assert advisory_n.downwind == "S"
    assert ctx_n.downwind_risk_direction == "S"


# ── 4. The committed camera config projects a map point and an FOV cone ─────────


@pytest.mark.integration
def test_committed_camera_config_projects_map_and_fov():
    settings = lo.load_settings()
    config = lo.load_camera_config(settings["camera_config"])
    camera = config["camera"]
    refs = config["reference_points"]

    cam_lat, cam_lon = camera["latitude"], camera["longitude"]
    assert len(refs) >= 4  # homography needs >= 4 enabled reference points

    # mapping: project a normalized image point through the real reference points.
    latlon = mapping.estimate_map_position(refs, (0.5, 0.7))
    assert latlon is not None
    lat, lon = latlon
    assert math.isfinite(lat) and math.isfinite(lon)
    # Plausible: within a degree or two of the camera (not a brittle exact coord).
    assert abs(lat - cam_lat) < 2.0
    assert abs(lon - cam_lon) < 2.0

    # live_ops_config: an approximate FOV cone (apex first, then a rim fan).
    cone = lo.approx_fov_cone(camera, refs)
    assert cone is not None
    assert len(cone) >= 3                      # apex + at least two rim points
    assert cone[0][0] == pytest.approx(cam_lat)
    assert cone[0][1] == pytest.approx(cam_lon)  # apex is the camera location
    for pt_lat, pt_lon in cone:
        assert math.isfinite(pt_lat) and math.isfinite(pt_lon)
        assert abs(pt_lat - cam_lat) < 2.0 and abs(pt_lon - cam_lon) < 2.0


# ── 5. The committed default-mode demo cache is internally consistent ───────────


@pytest.mark.integration
def test_committed_live_demo_cache_matches_frames_and_yields_detections():
    settings = lo.load_settings()
    items, source = lo.demo_sequence_items(settings)
    assert source == "frames"
    assert len(items) == 26                    # the 26 committed demo frames
    frames = lc.build_sequence_frames(items)
    assert len(frames) == 26

    # Rebuild the fingerprint exactly as the dashboard does: raw-frame content hash
    # + the default per-class thresholds + the resolved model + frame count.
    frames_hash = lc.frames_fingerprint(items)
    model_name = (
        "YOLO11s" if inference.checkpoint_exists("YOLO11s")
        else "YOLO11n" if inference.checkpoint_exists("YOLO11n") else None
    )
    assert model_name is not None             # committed weights ship in the repo
    smoke = round(float(settings.get("smoke_confidence_threshold", 0.40)), 4)
    fire = round(float(settings.get("fire_confidence_threshold", 0.40)), 4)
    fingerprint = lc.build_fingerprint(frames_hash, smoke, fire, model_name, len(frames))

    manifest = lc.load_manifest()
    assert manifest is not None
    # The shipped cache matches the shipped frames at the shipped defaults.
    assert lc.is_valid(manifest, fingerprint) is True

    # result_from_summary rebuilds a run_detection-shaped result (redraw, no YOLO).
    bytes_by_name = {f["name"]: f["bytes"] for f in frames}
    fire_summary = next(s for s in manifest["frames"] if s.get("fire_count"))
    result = lc.result_from_summary(fire_summary, bytes_by_name[fire_summary["name"]], model_name)

    assert isinstance(result["detections"], list) and result["detections"]
    assert isinstance(result["fire_count"], int) and result["fire_count"] >= 0
    assert isinstance(result["smoke_count"], int) and result["smoke_count"] >= 0
    assert result["total_detections"] == result["fire_count"] + result["smoke_count"]
    # The cached counts are consistent with the cached detection classes.
    assert sum(1 for d in result["detections"] if d["class"] == "fire") == result["fire_count"]
    assert sum(1 for d in result["detections"] if d["class"] == "smoke") == result["smoke_count"]
    # A well-formed annotated PNG was redrawn from the frame, and no detector ran.
    assert isinstance(result["annotated_png"], bytes)
    assert result["annotated_png"][:8] == _PNG_MAGIC
    assert result["inference_ms"] == 0.0
    assert result["from_cache"] is True


# ── 6. Real YOLO11s inference on a committed image (the M3 upload path) ──────────


@pytest.mark.integration
def test_real_yolo_inference_on_committed_image():
    pytest.importorskip("ultralytics")
    pytest.importorskip("torch")
    if not inference.checkpoint_exists("YOLO11s"):
        pytest.skip("YOLO11s checkpoint not present (weights are Git-ignored on this clone).")

    from PIL import Image

    # Pick a real committed image (read the dir; do not assume one specific file).
    candidates = [
        REPO_ROOT / "data" / "live_demo" / "frames" / "frame_05.jpg",
        REPO_ROOT / "data" / "live_demo" / "frame-1.jpg",
    ]
    img_path = next((p for p in candidates if p.exists()), None)
    if img_path is None:
        samples = sorted((REPO_ROOT / "data" / "samples" / "dfire" / "images").glob("*.jpg"))
        if not samples:
            pytest.skip("no committed image found to run inference on.")
        img_path = samples[0]

    model = inference.load_detector("YOLO11s")
    result = inference.run_detection(model, Image.open(img_path))

    # The exact result shape the M3 "Upload one image" demo relies on.
    assert isinstance(result["annotated_png"], bytes)
    assert result["annotated_png"][:8] == _PNG_MAGIC
    assert isinstance(result["fire_count"], int) and result["fire_count"] >= 0
    assert isinstance(result["smoke_count"], int) and result["smoke_count"] >= 0
    assert result["total_detections"] == result["fire_count"] + result["smoke_count"]
    assert isinstance(result["inference_ms"], float) and result["inference_ms"] > 0.0
    assert isinstance(result["detections"], list)

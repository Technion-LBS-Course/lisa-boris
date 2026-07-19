"""Unit tests for src/live_ops_config — settings/config/frame loading + FOV geometry.

No ML, no network. Uses the committed demo assets (config/live_ops_camera.json,
data/live_demo/frames) so it also guards that the demo assets stay present.
"""
import math

import pytest

from src import live_ops_config as lo


def test_default_settings_have_expected_keys():
    s = lo.load_settings("does/not/exist.yaml")
    for key in ("camera_config", "reference_frame", "frames_dir", "confidence_threshold",
                "confirmation_frames", "detection_interval_sec", "contacts"):
        assert key in s
    assert 0.0 < s["confidence_threshold"] <= 1.0
    assert s["confirmation_frames"] >= 1


def test_load_settings_reads_committed_yaml():
    s = lo.load_settings()  # config/live_ops.yaml
    assert s["camera_config"].endswith("live_ops_camera.json")
    # These keys are committed in config/live_ops.yaml with values that DIFFER from
    # the module defaults, so a green assertion proves the YAML was actually merged
    # (not that the defaults happened to coincide with the file).
    assert lo.DEFAULT_SETTINGS["confidence_threshold"] == 0.20
    assert s["confidence_threshold"] == 0.40
    assert lo.DEFAULT_SETTINGS["confirmation_frames"] == 3
    assert s["confirmation_frames"] == 1
    assert lo.DEFAULT_SETTINGS["detection_interval_sec"] == 2.0
    assert s["detection_interval_sec"] == 1.0


def test_load_settings_merges_yaml_and_skips_null(tmp_path):
    # A non-default value in the YAML wins over the default; a null-valued key is
    # skipped (falls back to the default rather than becoming None); a key absent
    # from the YAML keeps its default.
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        "confirmation_frames: 5\nconfidence_threshold: null\n", encoding="utf-8"
    )
    s = lo.load_settings(str(yaml_path))
    assert s["confirmation_frames"] == 5  # YAML override wins (module default is 3)
    # null in the YAML is filtered out, so the default is preserved (never None).
    assert s["confidence_threshold"] == lo.DEFAULT_SETTINGS["confidence_threshold"]
    assert s["confidence_threshold"] is not None
    # A key the YAML never mentions keeps its default.
    assert s["playback_speed_ms"] == lo.DEFAULT_SETTINGS["playback_speed_ms"]


def test_load_settings_malformed_yaml_falls_back_to_defaults(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: value: nested", encoding="utf-8")  # invalid YAML mapping
    s = lo.load_settings(str(bad))
    assert s == dict(lo.DEFAULT_SETTINGS)


def test_load_and_validate_camera_config():
    cfg = lo.load_camera_config("config/live_ops_camera.json")
    assert cfg["camera"].get("camera_id")
    assert cfg["camera"].get("latitude") is not None
    assert len([p for p in cfg["reference_points"] if p.get("enabled", True)]) >= 4
    assert len(cfg["image_zones"]) >= 1
    # Zones carry resolution-independent normalized vertices.
    assert cfg["image_zones"][0]["vertices_norm"]
    assert lo.validate_camera_config(cfg) == []


def test_validate_camera_config_flags_missing_pieces():
    issues = lo.validate_camera_config({"camera": {}, "reference_points": [], "image_zones": []})
    assert any("camera_id" in i for i in issues)
    assert any("reference points" in i for i in issues)


def test_load_camera_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        lo.load_camera_config(str(tmp_path / "nope.json"))


def test_load_camera_config_non_object_raises(tmp_path):
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")  # valid JSON, but not an object
    with pytest.raises(ValueError):
        lo.load_camera_config(str(bad))


def test_reference_frame_and_frames_present():
    ref = lo.load_reference_frame("data/live_demo/frame-1.jpg")
    assert ref and len(ref) > 1000
    frames = lo.list_frame_files("data/live_demo/frames")
    assert len(frames) >= 10  # the extracted demo sequence
    items = lo.load_frame_items("data/live_demo/frames")
    assert len(items) == len(frames)
    assert all(isinstance(name, str) and isinstance(data, bytes) for name, data in items)


def test_demo_sequence_prefers_frames_when_no_video():
    settings = dict(lo.DEFAULT_SETTINGS)
    settings["video_path"] = ""
    items, source = lo.demo_sequence_items(settings)
    assert source == "frames"
    assert items


def test_extract_video_frames_missing_path_returns_empty():
    assert lo.extract_video_frame_items("data/live_demo/nope.mp4") == []


# ── Optional operational context (landmarks / receptors / contact policy) ─────


def test_operational_context_keys_in_defaults():
    s = lo.load_settings("does/not/exist.yaml")
    assert s["operational_context_json"].endswith("live_ops_operational_context.json")
    assert s["operational_context_md"].endswith("live_ops_operational_context.md")


def test_load_operational_context_json_present():
    data = lo.load_operational_context_json("data/live_ops/live_ops_operational_context.json")
    assert isinstance(data, dict)
    assert data.get("context_type") == "operational_context"
    assert data.get("nearby_operational_landmarks")


def test_load_operational_context_md_present():
    text = lo.load_operational_context_md("data/live_ops/live_ops_operational_context.md")
    assert isinstance(text, str) and "Operational Context" in text


def test_operational_context_missing_returns_none():
    assert lo.load_operational_context_json("data/live_ops/nope.json") is None
    assert lo.load_operational_context_md("data/live_ops/nope.md") is None
    assert lo.load_operational_context_json("") is None
    assert lo.load_operational_context_md(None) is None


def test_load_operational_context_json_malformed_returns_none(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json ", encoding="utf-8")
    assert lo.load_operational_context_json(str(bad)) is None
    # A JSON array (non-object) is also rejected gracefully.
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2, 3]", encoding="utf-8")
    assert lo.load_operational_context_json(str(arr)) is None


def test_load_operational_context_combined():
    both = lo.load_operational_context(lo.load_settings())
    assert isinstance(both["json"], dict)
    assert isinstance(both["md"], str)


def test_approx_fov_cone_from_real_config():
    cfg = lo.load_camera_config("config/live_ops_camera.json")
    cone = lo.approx_fov_cone(cfg["camera"], cfg["reference_points"])
    assert cone is not None
    assert len(cone) >= 3
    # Apex is the camera location.
    assert math.isclose(cone[0][0], cfg["camera"]["latitude"], abs_tol=1e-6)
    assert math.isclose(cone[0][1], cfg["camera"]["longitude"], abs_tol=1e-6)


def test_approx_fov_cone_needs_camera_and_points():
    assert lo.approx_fov_cone({}, []) is None
    assert lo.approx_fov_cone({"latitude": 1.0, "longitude": 2.0}, []) is None


def test_approx_fov_cone_wraps_across_north():
    # Two reference points straddling due north: bearing ~10 deg (just east of north)
    # and ~350 deg (just west of north) from a camera at (0, 0). The cone must span
    # the SMALL ~20 deg arc across north, not the ~340 deg complement through south.
    camera = {"latitude": 0.0, "longitude": 0.0}
    refs = [
        {"map_lat": 1.0, "map_lon": 0.18},
        {"map_lat": 1.0, "map_lon": -0.18},
    ]
    # Confirm the setup really does straddle north (raw bearings on opposite sides of 0).
    assert lo._bearing_deg(0.0, 0.0, 1.0, 0.18) < 20.0
    assert lo._bearing_deg(0.0, 0.0, 1.0, -0.18) > 340.0

    cone = lo.approx_fov_cone(camera, refs)
    assert cone is not None
    rim = cone[1:]  # cone[0] is the apex (camera location)
    # Every rim vertex is NORTH of the camera. Without the wraparound correction the
    # fan would sweep the long way round through due south (latitude < 0 near 180 deg).
    assert all(lat > 0.0 for lat, _lon in rim)
    # The arc stays narrow around the prime meridian; the 340 deg complement would fan
    # east/west to |lon| > 1 deg.
    assert max(abs(lon) for _lat, lon in rim) < 0.5


def test_bearing_and_haversine_sanity():
    # Due north / due east from a point.
    assert abs(lo._bearing_deg(0, 0, 1, 0) - 0.0) < 1.0
    assert abs(lo._bearing_deg(0, 0, 0, 1) - 90.0) < 1.0
    assert lo._haversine_km(0, 0, 0, 0) == 0.0
    assert lo._haversine_km(0, 0, 0, 1) > 100  # ~111 km per degree lon at equator

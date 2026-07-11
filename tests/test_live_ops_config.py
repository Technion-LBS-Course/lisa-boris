"""Unit tests for src/live_ops_config — settings/config/frame loading + FOV geometry.

No ML, no network. Uses the committed demo assets (config/live_ops_camera.json,
data/live_demo/frames) so it also guards that the demo assets stay present.
"""
import math

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


def test_bearing_and_haversine_sanity():
    # Due north / due east from a point.
    assert abs(lo._bearing_deg(0, 0, 1, 0) - 0.0) < 1.0
    assert abs(lo._bearing_deg(0, 0, 0, 1) - 90.0) < 1.0
    assert lo._haversine_km(0, 0, 0, 0) == 0.0
    assert lo._haversine_km(0, 0, 0, 1) > 100  # ~111 km per degree lon at equator

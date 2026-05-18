"""Smoke tests for PyroFinder.

Verifies that the src package imports cleanly and that core helpers
return expected values. No ML models or datasets required.
"""

import pytest

import src
from src.data import get_primary_dataset_info, list_expected_dataset_columns
from src.model import get_model_plan, get_metrics_plan
from src.detection import DetectionResult, validate_detection_class
from src.tracking import is_confirmed_detection, estimate_apparent_direction
from src.mapping import get_mapping_modes, format_approximate_location, point_in_polygon
from src.alerts import create_alert_record, validate_alert_status


# ── Package ────────────────────────────────────────────────────────────────────

def test_package_version():
    assert hasattr(src, "__version__")
    assert src.__version__ == "0.1.0"


# ── src.data ───────────────────────────────────────────────────────────────────

def test_primary_dataset_is_dfire():
    info = get_primary_dataset_info()
    assert info["name"] == "D-Fire Dataset"
    assert info["num_images"] == 21527
    assert "fire" in info["classes"]
    assert "smoke" in info["classes"]
    assert "human" not in info["classes"]


def test_expected_dataset_columns_not_empty():
    cols = list_expected_dataset_columns()
    assert isinstance(cols, list)
    assert len(cols) > 0
    assert "name" in cols
    assert "source_url" in cols


# ── src.model ──────────────────────────────────────────────────────────────────

def test_model_plan_primary_is_yolo11s():
    plan = get_model_plan()
    assert plan["primary_model"] == "YOLO11s"
    assert plan["baseline_model"] == "YOLO11n"
    assert plan["classes"] == ["fire", "smoke"]
    assert plan["image_size"] == 640


def test_metrics_plan_not_empty():
    metrics = get_metrics_plan()
    assert isinstance(metrics, list)
    assert any("mAP" in m for m in metrics)
    assert any("Precision" in m for m in metrics)
    assert any("Recall" in m for m in metrics)


# ── src.detection ──────────────────────────────────────────────────────────────

def test_validate_detection_class_accepts_fire_smoke():
    assert validate_detection_class("fire") is True
    assert validate_detection_class("smoke") is True


def test_validate_detection_class_rejects_other():
    assert validate_detection_class("human") is False
    assert validate_detection_class("vehicle") is False
    assert validate_detection_class("") is False
    assert validate_detection_class("Fire") is False  # case-sensitive


def test_detection_result_valid():
    d = DetectionResult(
        timestamp="2026-05-17T10:00:00Z",
        camera_id="cam_001",
        class_name="fire",
        confidence=0.87,
        bbox=(0.5, 0.5, 0.2, 0.3),
    )
    assert d.class_name == "fire"
    assert d.confidence == 0.87


def test_detection_result_rejects_invalid_class():
    with pytest.raises(ValueError):
        DetectionResult(
            timestamp="2026-05-17T10:00:00Z",
            camera_id="cam_001",
            class_name="vehicle",
            confidence=0.9,
            bbox=(0.5, 0.5, 0.2, 0.3),
        )


# ── src.tracking ───────────────────────────────────────────────────────────────

def test_confirmed_detection_true():
    assert is_confirmed_detection([True, True, True], required_frames=3) is True


def test_confirmed_detection_false_not_enough_frames():
    assert is_confirmed_detection([True, True], required_frames=3) is False


def test_confirmed_detection_false_gap():
    assert is_confirmed_detection([True, False, True], required_frames=3) is False


def test_confirmed_detection_last_n_matter():
    # Only the last 3 frames count
    assert is_confirmed_detection([False, True, True, True], required_frames=3) is True
    assert is_confirmed_detection([True, True, True, False], required_frames=3) is False


def test_estimate_direction_stationary():
    assert estimate_apparent_direction((0.5, 0.5), (0.5, 0.5)) == "stationary"


def test_estimate_direction_right():
    assert estimate_apparent_direction((0.3, 0.5), (0.6, 0.5)) == "right"


def test_estimate_direction_upper_right():
    result = estimate_apparent_direction((0.3, 0.6), (0.6, 0.3))
    assert result == "upper-right"


# ── src.mapping ────────────────────────────────────────────────────────────────

def test_mapping_modes_count():
    modes = get_mapping_modes()
    assert len(modes) == 6


def test_format_approximate_location_with_name_and_coords():
    result = format_approximate_location("north field", 32.0, 34.8)
    assert "approximate" in result.lower()
    assert "north field" in result


def test_format_approximate_location_name_only():
    result = format_approximate_location("parking area", None, None)
    assert "parking area" in result
    assert "approximate" in result.lower()


def test_format_approximate_location_unknown():
    result = format_approximate_location(None, None, None)
    assert "unknown" in result.lower()


def test_point_in_polygon_inside():
    square = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    assert point_in_polygon(0.5, 0.5, square) is True


def test_point_in_polygon_outside():
    square = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    assert point_in_polygon(0.05, 0.05, square) is False


# ── src.alerts ─────────────────────────────────────────────────────────────────

def test_validate_alert_status_valid():
    for s in ["active", "confirmed", "rejected", "false_alarm"]:
        assert validate_alert_status(s) is True


def test_validate_alert_status_invalid():
    assert validate_alert_status("pending") is False
    assert validate_alert_status("") is False


def test_create_alert_record_fire():
    record = create_alert_record(
        camera_id="cam_001",
        detected_class="fire",
        confidence=0.91,
        approximate_location="Approximate location: north field (image-space polygon)",
        apparent_direction="upper-right",
    )
    assert record["detected_class"] == "fire"
    assert record["status"] == "active"
    assert record["camera_id"] == "cam_001"
    assert "timestamp" in record


def test_create_alert_record_rejects_bad_class():
    with pytest.raises(ValueError):
        create_alert_record(
            camera_id="cam_001",
            detected_class="vehicle",
            confidence=0.8,
            approximate_location="unknown",
            apparent_direction="stationary",
        )


def test_create_alert_record_rejects_bad_status():
    with pytest.raises(ValueError):
        create_alert_record(
            camera_id="cam_001",
            detected_class="smoke",
            confidence=0.75,
            approximate_location="unknown",
            apparent_direction="right",
            status="pending",
        )

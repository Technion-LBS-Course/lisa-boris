"""Smoke tests for PyroFinder.

Verifies that the src package imports cleanly and that core helpers
return expected values. No ML models or datasets required.
"""

import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import src
from src.data import (
    get_primary_dataset_info,
    list_expected_dataset_columns,
    load_dfire_metadata,
    clean_dfire_metadata,
    get_dfire_summary,
)
from src.eda import (
    compute_summary_metrics,
    compute_category_counts,
    compute_split_counts,
    compute_bbox_stats,
    filter_metadata,
    get_primary_eda_insight,
)
from src.model import get_model_plan, get_metrics_plan
from src.detection import DetectionResult, validate_detection_class
from src.tracking import is_confirmed_detection, estimate_apparent_direction
from src.mapping import get_mapping_modes, format_approximate_location, point_in_polygon
from src.alerts import create_alert_record, validate_alert_status


# ── Minimal test DataFrame factory ────────────────────────────────────────────

def _make_test_df() -> pd.DataFrame:
    """Return a small DataFrame that mimics dfire_metadata.csv structure."""
    return pd.DataFrame([
        {
            "image_id": "img_001", "split": "train", "image_path": "/d/img_001.jpg",
            "label_path": "/d/img_001.txt", "has_label": True,
            "has_fire": True, "has_smoke": False, "image_category": "fire_only",
            "num_fire_boxes": 2, "num_smoke_boxes": 0, "total_boxes": 2,
            "mean_bbox_area": 0.05, "median_bbox_area": 0.05, "max_bbox_area": 0.07,
            "mean_bbox_aspect_ratio": 1.2, "image_width": 640, "image_height": 480,
            "source_dataset": "D-Fire",
        },
        {
            "image_id": "img_002", "split": "train", "image_path": "/d/img_002.jpg",
            "label_path": "/d/img_002.txt", "has_label": True,
            "has_fire": False, "has_smoke": True, "image_category": "smoke_only",
            "num_fire_boxes": 0, "num_smoke_boxes": 3, "total_boxes": 3,
            "mean_bbox_area": 0.03, "median_bbox_area": 0.03, "max_bbox_area": 0.04,
            "mean_bbox_aspect_ratio": 0.9, "image_width": 640, "image_height": 480,
            "source_dataset": "D-Fire",
        },
        {
            "image_id": "img_003", "split": "test", "image_path": "/d/img_003.jpg",
            "label_path": "/d/img_003.txt", "has_label": True,
            "has_fire": True, "has_smoke": True, "image_category": "fire_and_smoke",
            "num_fire_boxes": 1, "num_smoke_boxes": 1, "total_boxes": 2,
            "mean_bbox_area": 0.06, "median_bbox_area": 0.06, "max_bbox_area": 0.08,
            "mean_bbox_aspect_ratio": 1.0, "image_width": 640, "image_height": 480,
            "source_dataset": "D-Fire",
        },
        {
            "image_id": "img_004", "split": "train", "image_path": "/d/img_004.jpg",
            "label_path": "", "has_label": False,
            "has_fire": False, "has_smoke": False, "image_category": "background",
            "num_fire_boxes": 0, "num_smoke_boxes": 0, "total_boxes": 0,
            "mean_bbox_area": 0.0, "median_bbox_area": 0.0, "max_bbox_area": 0.0,
            "mean_bbox_aspect_ratio": 0.0, "image_width": 640, "image_height": 480,
            "source_dataset": "D-Fire",
        },
    ])


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


# ── src.data — new metadata helpers ───────────────────────────────────────────

def test_load_dfire_metadata_from_temp_csv():
    df_orig = _make_test_df()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df_orig.to_csv(f, index=False)
        tmp_path = f.name
    df = load_dfire_metadata(tmp_path)
    assert len(df) == 4
    assert "image_category" in df.columns
    assert "has_fire" in df.columns


def test_load_dfire_metadata_raises_if_missing():
    with pytest.raises(FileNotFoundError):
        load_dfire_metadata("/nonexistent/path/metadata.csv")


def test_clean_dfire_metadata_dtypes():
    df = _make_test_df()
    # Make has_fire a string to simulate CSV round-trip
    df["has_fire"] = df["has_fire"].astype(str)
    df["total_boxes"] = df["total_boxes"].astype(str)
    df["mean_bbox_area"] = df["mean_bbox_area"].astype(str)

    cleaned = clean_dfire_metadata(df)
    assert cleaned["has_fire"].dtype == bool
    assert cleaned["total_boxes"].dtype == int
    assert cleaned["mean_bbox_area"].dtype == float


def test_clean_dfire_metadata_removes_duplicates():
    df = _make_test_df()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert len(dup) == 5
    cleaned = clean_dfire_metadata(dup)
    assert len(cleaned) == 4


def test_get_dfire_summary_keys():
    df = clean_dfire_metadata(_make_test_df())
    summary = get_dfire_summary(df)
    assert "total_images" in summary
    assert "splits" in summary
    assert "categories" in summary
    assert summary["total_images"] == 4


# ── src.eda ────────────────────────────────────────────────────────────────────

def test_compute_category_counts():
    df = clean_dfire_metadata(_make_test_df())
    cat_df = compute_category_counts(df)
    assert set(cat_df.columns) == {"category", "count"}
    assert cat_df["count"].sum() == 4


def test_compute_split_counts():
    df = clean_dfire_metadata(_make_test_df())
    split_df = compute_split_counts(df)
    assert "split" in split_df.columns
    assert "count" in split_df.columns
    assert split_df["count"].sum() == 4


def test_filter_metadata_by_category():
    df = clean_dfire_metadata(_make_test_df())
    filtered = filter_metadata(df, categories=["fire_only"])
    assert len(filtered) == 1
    assert filtered.iloc[0]["image_category"] == "fire_only"


def test_filter_metadata_by_split():
    df = clean_dfire_metadata(_make_test_df())
    filtered = filter_metadata(df, splits=["test"])
    assert len(filtered) == 1
    assert filtered.iloc[0]["split"] == "test"


def test_filter_metadata_by_has_fire():
    df = clean_dfire_metadata(_make_test_df())
    filtered = filter_metadata(df, has_fire=True)
    assert all(filtered["has_fire"])
    assert len(filtered) == 2


def test_filter_metadata_no_filters_returns_all():
    df = clean_dfire_metadata(_make_test_df())
    filtered = filter_metadata(df)
    assert len(filtered) == len(df)


def test_compute_summary_metrics():
    df = clean_dfire_metadata(_make_test_df())
    metrics = compute_summary_metrics(df)
    assert metrics["total_images"] == 4
    assert metrics["background_images"] == 1
    assert metrics["fire_images"] == 2  # fire_only + fire_and_smoke


def test_get_primary_eda_insight_returns_string():
    df = clean_dfire_metadata(_make_test_df())
    insight = get_primary_eda_insight(df)
    assert isinstance(insight, str)
    assert len(insight) > 10


def test_get_primary_eda_insight_empty_df():
    df = pd.DataFrame(columns=_make_test_df().columns)
    insight = get_primary_eda_insight(df)
    assert "No data" in insight

"""Tests for src.results_loader — result file loading, status, and winner logic.

These tests use only temporary files (never the committed ``results/`` files) and
require no ML models, datasets, or model weights. They lock in the behavior that
PyroFinder must guarantee while YOLO11s training is in progress:
    * missing YOLO11s files report ``training_in_progress`` (no invented metrics),
    * a valid YOLO11s detection/operational file loads correctly,
    * operational JSON is never treated as a detection baseline (and vice versa),
    * pending / synthetic results can never be selected as the winner,
    * malformed JSON is handled safely instead of crashing,
    * measured YOLO11n and YOLO11s results compare without mixing metric families.
"""

import json

import pytest

from src.results_loader import (
    STATUS_OK,
    STATUS_TRAINING_IN_PROGRESS,
    STATUS_MALFORMED,
    STATUS_NOT_DETECTION,
    STATUS_NOT_OPERATIONAL,
    load_detection_result,
    load_operational_result,
    is_selectable_operational,
    select_operational_winner,
    status_label,
)


# ── Fixtures: minimal valid result documents ───────────────────────────────────

def _detection_doc(model_name, recall=0.68, map50=0.74):
    return {
        "model_name": model_name,
        "model_family": "object_detection",
        "run_date": "2026-06-09",
        "metrics": {
            "map50": map50,
            "map50_95": 0.42,
            "precision": 0.74,
            "recall": recall,
            "f1": 0.71,
        },
    }


def _operational_doc(model_name, hazard_recall=0.93, far=0.02, score=0.94, status=None):
    doc = {
        "model_name": model_name,
        "model_family": "object_detection",
        "evaluation_type": "operational_alert_metrics",
        "run_date": "2026-06-10",
        "operational_metrics": {
            "hazard_recall": hazard_recall,
            "false_alert_rate": far,
            "alert_precision": 0.98,
            "alert_f1": 0.95,
            "operational_alert_score": score,
        },
        "location_metrics": {
            "fire_location_error_mean": 0.013,
            "fire_location_grid_hit_rate": 0.95,
            "location_coverage_rate": 0.91,
        },
    }
    if status is not None:
        doc["status"] = status
    return doc


def _write(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


# ── 1. Missing YOLO11s files → training_in_progress ─────────────────────────────

def test_missing_detection_file_is_training_in_progress(tmp_path):
    missing = tmp_path / "baseline_yolo11s.json"
    loaded = load_detection_result(missing)
    assert loaded["status"] == STATUS_TRAINING_IN_PROGRESS
    assert loaded["data"] is None
    assert status_label(loaded["status"]) == "Training in progress"


def test_missing_operational_file_is_training_in_progress(tmp_path):
    missing = tmp_path / "yolo11s_operational_metrics.json"
    loaded = load_operational_result(missing)
    assert loaded["status"] == STATUS_TRAINING_IN_PROGRESS
    assert loaded["data"] is None


# ── 2. Valid YOLO11s files load correctly ───────────────────────────────────────

def test_valid_detection_file_loads(tmp_path):
    path = _write(tmp_path / "baseline_yolo11s.json", _detection_doc("YOLO11s"))
    loaded = load_detection_result(path)
    assert loaded["status"] == STATUS_OK
    assert loaded["data"]["metrics"]["map50"] == 0.74


def test_valid_operational_file_loads_and_is_selectable(tmp_path):
    path = _write(tmp_path / "yolo11s_operational_metrics.json", _operational_doc("YOLO11s"))
    loaded = load_operational_result(path)
    assert loaded["status"] == STATUS_OK
    assert is_selectable_operational(loaded) is True


# ── 3. Operational JSON is not treated as a detection baseline (and vice versa) ──

def test_operational_json_rejected_as_detection(tmp_path):
    path = _write(tmp_path / "yolo11s_operational_metrics.json", _operational_doc("YOLO11s"))
    loaded = load_detection_result(path)
    assert loaded["status"] == STATUS_NOT_DETECTION


def test_detection_json_rejected_as_operational(tmp_path):
    path = _write(tmp_path / "baseline_yolo11s.json", _detection_doc("YOLO11s"))
    loaded = load_operational_result(path)
    assert loaded["status"] == STATUS_NOT_OPERATIONAL


# ── 4. Pending / synthetic results cannot be selected as winner ─────────────────

def test_missing_files_yield_no_winner(tmp_path):
    op_items = [
        ("YOLO11n", tmp_path / "yolo11n_operational_metrics.json"),
        ("YOLO11s", tmp_path / "yolo11s_operational_metrics.json"),
    ]
    assert select_operational_winner(op_items) is None


def test_synthetic_status_is_not_selectable(tmp_path):
    path = _write(
        tmp_path / "yolo11s_operational_metrics.json",
        _operational_doc("YOLO11s", status="synthetic_placeholder"),
    )
    loaded = load_operational_result(path)
    assert loaded["status"] == STATUS_OK          # it parses fine ...
    assert is_selectable_operational(loaded) is False  # ... but cannot win
    assert select_operational_winner([("YOLO11s", path)]) is None


def test_null_required_metric_is_not_selectable(tmp_path):
    doc = _operational_doc("YOLO11s")
    doc["operational_metrics"]["hazard_recall"] = None
    path = _write(tmp_path / "yolo11s_operational_metrics.json", doc)
    loaded = load_operational_result(path)
    assert is_selectable_operational(loaded) is False


def test_pending_yolo11s_does_not_beat_measured_yolo11n(tmp_path):
    # YOLO11n measured; YOLO11s file absent → generic missing-file fallback (never wins).
    n_op = _write(tmp_path / "yolo11n_operational_metrics.json", _operational_doc("YOLO11n"))
    op_items = [
        ("YOLO11n", n_op),
        ("YOLO11s", tmp_path / "yolo11s_operational_metrics.json"),  # missing
    ]
    assert select_operational_winner(op_items) == "YOLO11n"


# ── 5. Malformed JSON is handled safely ─────────────────────────────────────────

def test_malformed_detection_json(tmp_path):
    bad = tmp_path / "baseline_yolo11s.json"
    bad.write_text("{ not valid json ]", encoding="utf-8")
    loaded = load_detection_result(bad)
    assert loaded["status"] == STATUS_MALFORMED
    assert loaded["data"] is None


def test_malformed_operational_json(tmp_path):
    bad = tmp_path / "yolo11s_operational_metrics.json"
    bad.write_text("not json at all", encoding="utf-8")
    loaded = load_operational_result(bad)
    assert loaded["status"] == STATUS_MALFORMED
    assert is_selectable_operational(loaded) is False


# ── 6. Measured YOLO11n + YOLO11s compare without mixing metric families ─────────

def test_winner_uses_hazard_recall_first(tmp_path):
    # YOLO11s has higher hazard recall → it wins despite identical other fields.
    n_op = _write(
        tmp_path / "yolo11n_operational_metrics.json",
        _operational_doc("YOLO11n", hazard_recall=0.90),
    )
    s_op = _write(
        tmp_path / "yolo11s_operational_metrics.json",
        _operational_doc("YOLO11s", hazard_recall=0.95),
    )
    assert select_operational_winner([("YOLO11n", n_op), ("YOLO11s", s_op)]) == "YOLO11s"


def test_winner_tiebreak_false_alert_rate(tmp_path):
    # Equal hazard recall → lower false alert rate wins.
    n_op = _write(
        tmp_path / "yolo11n_operational_metrics.json",
        _operational_doc("YOLO11n", hazard_recall=0.93, far=0.05),
    )
    s_op = _write(
        tmp_path / "yolo11s_operational_metrics.json",
        _operational_doc("YOLO11s", hazard_recall=0.93, far=0.02),
    )
    assert select_operational_winner([("YOLO11n", n_op), ("YOLO11s", s_op)]) == "YOLO11s"


def test_detection_and_operational_files_do_not_cross_contaminate(tmp_path):
    # A detection file and an operational file for the same model are read by the
    # correct loader only — neither leaks into the other metric family.
    det = _write(tmp_path / "baseline_yolo11s.json", _detection_doc("YOLO11s"))
    op = _write(tmp_path / "yolo11s_operational_metrics.json", _operational_doc("YOLO11s"))

    det_loaded = load_detection_result(det)
    op_loaded = load_operational_result(op)
    assert det_loaded["status"] == STATUS_OK
    assert "map50" in det_loaded["data"]["metrics"]
    assert op_loaded["status"] == STATUS_OK
    assert "hazard_recall" in op_loaded["data"]["operational_metrics"]
    # detection metrics carry no operational keys and vice versa
    assert "hazard_recall" not in det_loaded["data"]["metrics"]
    assert "map50" not in op_loaded["data"]["operational_metrics"]

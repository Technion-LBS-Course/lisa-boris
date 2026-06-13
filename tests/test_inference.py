"""Tests for src.inference — lazy YOLO11n / YOLO11s loading helpers.

These tests never load real model weights and never import ultralytics. They
verify the cheap, pure-Python guards around the demo:
    * checkpoint paths resolve to the fine-tuned D-Fire weights,
    * a missing checkpoint raises FileNotFoundError before any ML import,
    * class validation accepts only fire/smoke,
    * availability detection reflects which checkpoints are present.
"""

from pathlib import Path

import pytest

from src import inference


def test_checkpoint_paths_are_finetuned_dfire_weights():
    assert inference.checkpoint_path("YOLO11n") == Path("models/yolo11n_dfire_best.pt")
    assert inference.checkpoint_path("YOLO11s") == Path("models/yolo11s_dfire_best.pt")


def test_unknown_detector_raises():
    with pytest.raises(KeyError):
        inference.checkpoint_path("YOLOv12")


def test_validate_detector_classes_requires_exactly_fire_and_smoke():
    # Exactly the two classes, in either ID order (order-independent).
    assert inference.validate_detector_classes({0: "smoke", 1: "fire"}) is True
    assert inference.validate_detector_classes({0: "fire", 1: "smoke"}) is True
    assert inference.validate_detector_classes(["fire", "smoke"]) is True
    assert inference.validate_detector_classes({0: "smoke", 1: "fire"}.values()) is True
    # Case / whitespace are normalized.
    assert inference.validate_detector_classes([" Fire ", "SMOKE"]) is True


def test_validate_detector_classes_rejects_anything_but_both_classes():
    # Only one of the two classes is not enough.
    assert inference.validate_detector_classes({0: "fire"}) is False
    assert inference.validate_detector_classes({0: "smoke"}) is False
    assert inference.validate_detector_classes(["fire"]) is False
    assert inference.validate_detector_classes(["smoke"]) is False
    # Empty class mapping.
    assert inference.validate_detector_classes({}) is False
    assert inference.validate_detector_classes([]) is False
    # Any extra class is rejected.
    assert inference.validate_detector_classes({0: "smoke", 1: "fire", 2: "person"}) is False
    assert inference.validate_detector_classes(["fire", "smoke", "background"]) is False
    assert inference.validate_detector_classes(["fire", "smoke", "vehicle"]) is False


def test_validate_detector_classes_rejects_malformed_metadata():
    # Missing / non-iterable class metadata must be rejected, not crash.
    assert inference.validate_detector_classes(None) is False
    assert inference.validate_detector_classes(123) is False


def test_load_detector_missing_checkpoint_raises_filenotfound(tmp_path, monkeypatch):
    # Point YOLO11s at a non-existent path; load must fail before importing ultralytics.
    monkeypatch.setitem(
        inference.CHECKPOINTS, "YOLO11s", tmp_path / "does_not_exist.pt"
    )
    with pytest.raises(FileNotFoundError):
        inference.load_detector("YOLO11s")


def test_checkpoint_exists_and_available_detectors(tmp_path, monkeypatch):
    present = tmp_path / "yolo11n_dfire_best.pt"
    present.write_bytes(b"not a real checkpoint")  # presence only; never loaded
    monkeypatch.setitem(inference.CHECKPOINTS, "YOLO11n", present)
    monkeypatch.setitem(inference.CHECKPOINTS, "YOLO11s", tmp_path / "missing.pt")

    assert inference.checkpoint_exists("YOLO11n") is True
    assert inference.checkpoint_exists("YOLO11s") is False
    assert inference.available_detectors() == ["YOLO11n"]


def test_missing_yolo11s_message_is_actionable():
    assert "models/yolo11s_dfire_best.pt" in inference.MISSING_YOLO11S_MESSAGE
    assert "in progress" in inference.MISSING_YOLO11S_MESSAGE.lower()

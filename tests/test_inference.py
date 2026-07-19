"""Tests for src.inference — lazy YOLO11n / YOLO11s loading helpers.

These tests never load real model weights and never import ultralytics. They
verify the cheap, pure-Python guards around the demo:
    * checkpoint paths resolve to the fine-tuned D-Fire weights,
    * a missing checkpoint raises FileNotFoundError before any ML import,
    * class validation accepts only fire/smoke,
    * availability detection reflects which checkpoints are present.
"""

import io
from pathlib import Path

import pytest

from src import inference


def test_checkpoint_paths_are_finetuned_dfire_weights():
    # Paths are resolved against the repo root (not the CWD) so inference works
    # regardless of where the process is launched, e.g. Streamlit Community Cloud.
    n_path = inference.checkpoint_path("YOLO11n")
    s_path = inference.checkpoint_path("YOLO11s")
    assert n_path.is_absolute() and s_path.is_absolute()
    assert n_path.parent.name == "models" and n_path.name == "yolo11n_dfire_best.pt"
    assert s_path.parent.name == "models" and s_path.name == "yolo11s_dfire_best.pt"


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


def _install_fake_ultralytics(monkeypatch, names):
    """Inject a fake ``ultralytics.YOLO`` whose model exposes ``names``.

    Lets us exercise ``load_detector`` past the lazy ``from ultralytics import
    YOLO`` without installing ultralytics or reading real weights.
    """
    import sys
    import types

    class _FakeYOLO:
        def __init__(self, path):
            self.path = path
            self.names = names

    module = types.ModuleType("ultralytics")
    module.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return _FakeYOLO


def test_load_detector_accepts_finetuned_fire_smoke_checkpoint(tmp_path, monkeypatch):
    ckpt = tmp_path / "yolo11s_dfire_best.pt"
    ckpt.write_bytes(b"present but never parsed")  # presence only; YOLO is faked
    monkeypatch.setitem(inference.CHECKPOINTS, "YOLO11s", ckpt)
    fake_cls = _install_fake_ultralytics(monkeypatch, {0: "smoke", 1: "fire"})

    model = inference.load_detector("YOLO11s")
    assert isinstance(model, fake_cls)
    # Loaded from the fine-tuned checkpoint path, never a pretrained substitute.
    assert model.path == str(ckpt)


def test_load_detector_rejects_checkpoint_exposing_wrong_classes(tmp_path, monkeypatch):
    ckpt = tmp_path / "yolo11s_dfire_best.pt"
    ckpt.write_bytes(b"present but never parsed")
    monkeypatch.setitem(inference.CHECKPOINTS, "YOLO11s", ckpt)
    # A checkpoint whose classes are not exactly {fire, smoke} must be rejected,
    # never silently used — PyroFinder is a strict two-class detector.
    _install_fake_ultralytics(monkeypatch, {0: "person", 1: "car", 2: "fire"})

    with pytest.raises(ValueError, match="fire and smoke"):
        inference.load_detector("YOLO11s")


def test_checkpoint_exists_and_available_detectors(tmp_path, monkeypatch):
    present = tmp_path / "yolo11n_dfire_best.pt"
    present.write_bytes(b"not a real checkpoint")  # presence only; never loaded
    monkeypatch.setitem(inference.CHECKPOINTS, "YOLO11n", present)
    monkeypatch.setitem(inference.CHECKPOINTS, "YOLO11s", tmp_path / "missing.pt")

    assert inference.checkpoint_exists("YOLO11n") is True
    assert inference.checkpoint_exists("YOLO11s") is False
    assert inference.available_detectors() == ["YOLO11n"]


def test_missing_yolo11s_message_names_the_real_checkpoint_and_next_step():
    # Tie the operator-facing message to the actual checkpoint the loader looks
    # for: if the checkpoint filename ever changes, this fails so the message is
    # kept in sync (rather than mirroring a hardcoded literal that would drift).
    msg = inference.MISSING_YOLO11S_MESSAGE
    checkpoint_name = inference.checkpoint_path("YOLO11s").name
    assert checkpoint_name == "yolo11s_dfire_best.pt"  # guards the D-Fire naming
    assert checkpoint_name in msg
    # It must tell the operator what to do, not just that something is wrong.
    assert "add" in msg.lower()
    # And it must not leak a machine-specific absolute path.
    assert "C:\\" not in msg and "/home/" not in msg


# ── detection result helpers (pure — no model needed) ─────────────────────────


def test_top_hazard_detection_picks_highest_confidence_within_fire():
    result = {"detections": [
        {"class": "smoke", "confidence": 0.4, "bbox_norm": [0.2, 0.2, 0.1, 0.1]},
        {"class": "fire", "confidence": 0.9, "bbox_norm": [0.5, 0.5, 0.2, 0.2]},
    ]}
    top = inference.top_hazard_detection(result)
    assert top["class"] == "fire"
    assert top["confidence"] == 0.9


def test_top_hazard_detection_prefers_fire_over_higher_confidence_smoke():
    # Fire is the event focus even when a smoke detection has higher confidence.
    result = {"detections": [
        {"class": "fire", "confidence": 0.3, "bbox_norm": [0.5, 0.5, 0.2, 0.2]},
        {"class": "smoke", "confidence": 0.95, "bbox_norm": [0.2, 0.2, 0.1, 0.1]},
    ]}
    top = inference.top_hazard_detection(result)
    assert top["class"] == "fire"
    assert top["confidence"] == 0.3


def test_top_hazard_detection_smoke_only():
    result = {"detections": [
        {"class": "smoke", "confidence": 0.5, "bbox_norm": [0.2, 0.2, 0.1, 0.1]},
        {"class": "smoke", "confidence": 0.7, "bbox_norm": [0.3, 0.3, 0.1, 0.1]},
    ]}
    top = inference.top_hazard_detection(result)
    assert top["class"] == "smoke"
    assert top["confidence"] == 0.7


def test_top_hazard_detection_none_when_empty():
    assert inference.top_hazard_detection({"detections": []}) is None
    assert inference.top_hazard_detection({}) is None


def test_bbox_bottom_center_norm_anchor_and_clamp():
    x, y = inference.bbox_bottom_center_norm([0.5, 0.4, 0.2, 0.2])
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5)  # y_center 0.4 + height/2 0.1
    # anchor clamps to the frame
    _, y2 = inference.bbox_bottom_center_norm([0.5, 0.95, 0.2, 0.4])
    assert y2 == pytest.approx(1.0)


# ── select_confirmed_event_detection (fire priority across a frame window) ────


def _res(*classes_and_conf):
    """Build a run_detection-shaped result from (class, confidence) pairs."""
    return {"detections": [
        {"class": cls, "confidence": conf, "bbox_norm": [0.5, 0.5, 0.1, 0.1]}
        for cls, conf in classes_and_conf
    ]}


def test_select_confirmed_event_prefers_current_frame_fire():
    window = [_res(("smoke", 0.9)), _res(("fire", 0.4))]
    focus = inference.select_confirmed_event_detection(window)
    assert focus["class"] == "fire"
    assert focus["confidence"] == 0.4


def test_select_confirmed_event_falls_back_to_earlier_fire_when_current_is_smoke():
    window = [_res(("fire", 0.6)), _res(("smoke", 0.9))]
    focus = inference.select_confirmed_event_detection(window)
    assert focus["class"] == "fire"
    assert focus["confidence"] == 0.6


def test_select_confirmed_event_uses_most_recent_fire_in_window():
    window = [_res(("fire", 0.3)), _res(("fire", 0.5)), _res(("smoke", 0.9))]
    focus = inference.select_confirmed_event_detection(window)
    assert focus["class"] == "fire"
    assert focus["confidence"] == 0.5  # the more recent of the two fire frames


def test_select_confirmed_event_uses_current_smoke_when_no_fire_anywhere():
    window = [_res(("smoke", 0.5)), _res(("smoke", 0.8))]
    focus = inference.select_confirmed_event_detection(window)
    assert focus["class"] == "smoke"
    assert focus["confidence"] == 0.8


def test_select_confirmed_event_skips_none_entries():
    window = [None, _res(("fire", 0.7)), None, _res(("smoke", 0.9))]
    focus = inference.select_confirmed_event_detection(window)
    assert focus["class"] == "fire"
    assert focus["confidence"] == 0.7


def test_select_confirmed_event_none_when_no_detections_in_window():
    window = [_res(), _res()]
    assert inference.select_confirmed_event_detection(window) is None


def test_select_confirmed_event_empty_window():
    assert inference.select_confirmed_event_detection([]) is None


# ── run_detection per-class confidence (fake model — no ultralytics/torch) ────


class _FakeArr:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _FakeBoxes:
    def __init__(self, cls, conf, xywhn):
        self.cls = _FakeArr(cls)
        self.conf = _FakeArr(conf)
        self.xywhn = _FakeArr(xywhn)

    def __len__(self):
        return len(self.cls._data)


class _FakeResult:
    # Deliberately not indexable (no __getitem__): the per-class pre-plot filter
    # falls back to the extraction-loop threshold enforcement, which is what we test.
    def __init__(self, names, boxes, plot_arr):
        self.names = names
        self.boxes = boxes
        self._plot_arr = plot_arr

    def plot(self):
        return self._plot_arr


class _FakeModel:
    def __init__(self, result):
        self._result = result
        self.last_conf = None

    def predict(self, source=None, conf=None, imgsz=None, verbose=None):
        self.last_conf = conf
        return [self._result]


def _fake_model_two_boxes():
    import numpy as np

    boxes = _FakeBoxes(
        cls=[0, 1],                       # 0=smoke, 1=fire
        conf=[0.45, 0.42],
        xywhn=[[0.2, 0.2, 0.1, 0.1], [0.6, 0.6, 0.2, 0.2]],
    )
    result = _FakeResult({0: "smoke", 1: "fire"}, boxes,
                         np.zeros((4, 4, 3), dtype=np.uint8))
    return _FakeModel(result)


def test_run_detection_per_class_confidence_filters_each_class():
    from PIL import Image

    model = _fake_model_two_boxes()
    out = inference.run_detection(
        model, Image.new("RGB", (8, 8)),
        conf=min(0.5, 0.4), conf_by_class={"smoke": 0.5, "fire": 0.4},
    )
    # smoke 0.45 < 0.5 → dropped; fire 0.42 ≥ 0.4 → kept.
    assert out["smoke_count"] == 0
    assert out["fire_count"] == 1
    assert [d["class"] for d in out["detections"]] == ["fire"]
    # Inference ran at the lowest per-class threshold so all candidates are returned.
    assert model.last_conf == pytest.approx(0.4)


def test_run_detection_single_conf_unchanged_behavior():
    from PIL import Image

    model = _fake_model_two_boxes()
    out = inference.run_detection(model, Image.new("RGB", (8, 8)), conf=0.4)
    # No conf_by_class → single threshold, both boxes (0.45, 0.42) kept.
    assert out["smoke_count"] == 1
    assert out["fire_count"] == 1
    assert model.last_conf == pytest.approx(0.4)


class _IndexableFakeResult(_FakeResult):
    """A Results-like object that supports index-list selection (``result[keep]``).

    Ultralytics ``Results`` objects accept a list of kept indices; this lets us
    drive ``run_detection``'s pre-plot overlay-filtering branch, which the plain
    (non-indexable) ``_FakeResult`` deliberately skips. The filtered sub-result
    carries a visually distinct overlay so the test can prove the annotated PNG
    was rendered from the FILTERED boxes, not the original ones.
    """

    def __init__(self, names, boxes, plot_arr):
        super().__init__(names, boxes, plot_arr)
        self.indexed_with = None

    def __getitem__(self, keep_idx):
        import numpy as np

        kept = list(keep_idx)
        filtered = _FakeBoxes(
            cls=[self.boxes.cls._data[i] for i in kept],
            conf=[self.boxes.conf._data[i] for i in kept],
            xywhn=[self.boxes.xywhn._data[i] for i in kept],
        )
        distinct_overlay = np.full((4, 4, 3), 200, dtype=np.uint8)
        sub = _IndexableFakeResult(self.names, filtered, distinct_overlay)
        self.indexed_with = kept
        return sub


def test_run_detection_prefilters_overlay_when_results_are_indexable():
    import numpy as np
    from PIL import Image

    boxes = _FakeBoxes(
        cls=[0, 1],                       # 0=smoke, 1=fire
        conf=[0.45, 0.42],
        xywhn=[[0.2, 0.2, 0.1, 0.1], [0.6, 0.6, 0.2, 0.2]],
    )
    original_overlay = np.zeros((4, 4, 3), dtype=np.uint8)  # distinct from filtered (200)
    result = _IndexableFakeResult({0: "smoke", 1: "fire"}, boxes, original_overlay)
    model = _FakeModel(result)

    out = inference.run_detection(
        model, Image.new("RGB", (8, 8)),
        conf=min(0.5, 0.4), conf_by_class={"smoke": 0.5, "fire": 0.4},
    )

    # smoke 0.45 < 0.5 dropped, fire 0.42 ≥ 0.4 kept → the index-select branch ran
    # with exactly the surviving box index.
    assert result.indexed_with == [1]
    assert out["smoke_count"] == 0 and out["fire_count"] == 1
    # The annotated overlay must come from the FILTERED sub-result (all 200s),
    # not the original (all 0s) — i.e. the overlay matches the counted detections.
    annotated = np.array(Image.open(io.BytesIO(out["annotated_png"])).convert("RGB"))
    assert int(annotated.min()) == 200 and int(annotated.max()) == 200

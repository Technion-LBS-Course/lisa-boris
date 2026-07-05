"""Unit tests for src/segmentation_assist.py.

Segmentation-assisted polygon refinement for Image Zone setup. The module must
import without pulling in cv2 / numpy / a heavy model (all lazy), so a plain
import and the pure helpers (validate/fallback/adapters) run with no extra deps.
The tests that exercise the real OpenCV path skip cleanly if cv2 is absent.
"""
import ast
from pathlib import Path

import pytest

import src.segmentation_assist as seg


# ── 1. Import safety: no heavy imports at module load ─────────────────────────


def test_module_has_no_heavy_top_level_imports():
    text = Path(seg.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    top_imports: list[str] = []
    for node in tree.body:  # module top level only — lazy imports live inside functions
        if isinstance(node, ast.Import):
            top_imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module or "")
    heavy = {"cv2", "numpy", "PIL", "streamlit", "ultralytics", "torch"}
    assert not any(m and m.split(".")[0] in heavy for m in top_imports), top_imports


def test_backend_available_is_boolean_and_does_not_raise():
    assert isinstance(seg.segmentation_backend_available(), bool)


# ── 2. validate_roi_box: clamp + reorder + degenerate ─────────────────────────


def test_validate_roi_box_reorders_corners():
    box = seg.validate_roi_box({"x_min": 0.8, "y_min": 0.9, "x_max": 0.2, "y_max": 0.3})
    assert box == {"x_min": 0.2, "y_min": 0.3, "x_max": 0.8, "y_max": 0.9}


def test_validate_roi_box_clamps_out_of_range():
    box = seg.validate_roi_box({"x_min": -0.5, "y_min": 0.1, "x_max": 1.5, "y_max": 0.9})
    assert box == {"x_min": 0.0, "y_min": 0.1, "x_max": 1.0, "y_max": 0.9}


def test_validate_roi_box_accepts_xyxy_list_via_adapter():
    box = seg.validate_roi_box([0.2, 0.3, 0.6, 0.7])
    assert box == {"x_min": 0.2, "y_min": 0.3, "x_max": 0.6, "y_max": 0.7}


def test_validate_roi_box_rejects_degenerate():
    with pytest.raises(ValueError):
        seg.validate_roi_box({"x_min": 0.5, "y_min": 0.5, "x_max": 0.5, "y_max": 0.5})
    with pytest.raises(ValueError):
        seg.validate_roi_box([0.0, 0.0, 0.0, 0.0])


def test_validate_roi_box_rejects_malformed():
    with pytest.raises(ValueError):
        seg.validate_roi_box("nope")
    with pytest.raises(ValueError):
        seg.validate_roi_box([0.1, 0.2, 0.3])  # wrong length
    with pytest.raises(ValueError):
        seg.validate_roi_box({"x_min": 0.1})   # missing keys


def test_box_norm_xyxy_roundtrip():
    assert seg.box_norm_from_xyxy([0.1, 0.2, 0.3, 0.4]) == {
        "x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4
    }
    assert seg.box_norm_to_xyxy({"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}) == [
        0.1, 0.2, 0.3, 0.4
    ]


# ── 3. polygon_from_box_fallback ──────────────────────────────────────────────


def test_polygon_from_box_fallback_returns_four_normalized_corners():
    poly = seg.polygon_from_box_fallback({"x_min": 0.2, "y_min": 0.35, "x_max": 0.42, "y_max": 0.60})
    assert poly == [
        {"x": 0.2, "y": 0.35},
        {"x": 0.42, "y": 0.35},
        {"x": 0.42, "y": 0.60},
        {"x": 0.2, "y": 0.60},
    ]
    for v in poly:
        assert 0.0 <= v["x"] <= 1.0 and 0.0 <= v["y"] <= 1.0


def test_polygon_from_box_fallback_reorders_and_clamps():
    poly = seg.polygon_from_box_fallback([1.5, 0.9, -0.2, 0.1])  # out of range + reversed
    xs = {v["x"] for v in poly}
    ys = {v["y"] for v in poly}
    assert xs == {0.0, 1.0} and ys == {0.1, 0.9}


def test_polygon_to_pixel_vertices():
    poly = [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 0.25}, {"x": 1.0, "y": 1.0}]
    px = seg.polygon_to_pixel_vertices(poly, 640, 480)
    assert px == [[0.0, 0.0], [320.0, 120.0], [640.0, 480.0]]


# ── 4 & 5. mask_to_polygon (real OpenCV path) ─────────────────────────────────


def test_mask_to_polygon_simple_rectangle():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    mask = np.zeros((100, 100), dtype="uint8")
    mask[20:80, 30:70] = 1  # a filled rectangle
    poly = seg.mask_to_polygon(mask, 100, 100, simplify_tolerance=2.0)
    assert len(poly) >= 4
    for v in poly:
        assert 0.0 <= v["x"] <= 1.0 and 0.0 <= v["y"] <= 1.0
    xs = [v["x"] for v in poly]
    ys = [v["y"] for v in poly]
    # The polygon should bound the rectangle (x in ~[0.30, 0.70], y in ~[0.20, 0.80]).
    assert min(xs) <= 0.32 and max(xs) >= 0.68
    assert min(ys) <= 0.22 and max(ys) >= 0.78


def test_mask_to_polygon_empty_mask_returns_empty():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    assert seg.mask_to_polygon(np.zeros((50, 50), dtype="uint8"), 50, 50) == []


def test_mask_to_polygon_non_2d_returns_empty():
    np = pytest.importorskip("numpy")
    assert seg.mask_to_polygon(np.zeros((0,), dtype="uint8"), 10, 10) == []


# ── refine_box_to_mask: real segmentation + controlled failure ────────────────


def test_refine_box_to_mask_result_is_well_formed():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    from PIL import Image

    # High-contrast frame: light background with a solid dark block near the centre.
    arr = np.full((200, 200, 3), 220, dtype="uint8")
    arr[70:140, 70:140] = 30
    img = Image.fromarray(arr, "RGB")

    result = seg.refine_box_to_mask(img, {"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75})
    assert set(result) == {"ok", "backend", "polygon", "box_norm", "message"}
    assert result["box_norm"] == {"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75}
    if result["ok"]:
        assert result["backend"] == seg.SEG_BACKEND
        assert len(result["polygon"]) >= 3
        for v in result["polygon"]:
            assert 0.0 <= v["x"] <= 1.0 and 0.0 <= v["y"] <= 1.0
    else:
        # A non-crash outcome is acceptable, but must be controlled and empty.
        assert result["backend"] in {"empty-mask", "error"}
        assert result["polygon"] == []


def test_refine_box_to_mask_degenerate_box_raises():
    with pytest.raises(ValueError):
        seg.refine_box_to_mask(b"", {"x_min": 0.5, "y_min": 0.5, "x_max": 0.5, "y_max": 0.5})


def test_refine_box_to_mask_bad_image_fails_cleanly():
    # Valid box but an unusable image -> controlled failure, never an exception.
    result = seg.refine_box_to_mask(None, {"x_min": 0.2, "y_min": 0.2, "x_max": 0.8, "y_max": 0.8})
    assert result["ok"] is False
    assert result["polygon"] == []
    assert result["backend"] in {"error", "unavailable"}
    assert result["message"]


def test_refine_box_to_mask_reports_unavailable_backend(monkeypatch):
    monkeypatch.setattr(seg, "segmentation_backend_available", lambda: False)
    result = seg.refine_box_to_mask(b"whatever", {"x_min": 0.2, "y_min": 0.2, "x_max": 0.8, "y_max": 0.8})
    assert result["ok"] is False
    assert result["backend"] == "unavailable"
    # A fallback polygon is always derivable from the (validated) box.
    assert seg.polygon_from_box_fallback(result["box_norm"])

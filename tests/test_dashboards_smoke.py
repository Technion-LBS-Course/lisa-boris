"""Import-safety guards for the src/dashboards package.

Verifies the hard constraint that importing the dashboard modules does NOT pull in
heavy ML libraries (ultralytics / torch) at import time — detector loading must stay
lazy — and that key operator wiring (the Incident Assistant YOLO11s button, the
segmentation refiner) is present with no module-level heavy imports.

That the dashboards actually *render* is covered end-to-end in
``test_e2e_streamlit_app.py`` (which executes them through Streamlit), so this file
no longer keeps shallow ``callable(render)`` checks. These tests require Streamlit
but never load model weights or run inference.
"""
import sys

import pytest

pytest.importorskip("streamlit")


DASHBOARD_MODULES = [
    "src.dashboards.model_helpers",
    "src.dashboards.m3_dashboard",
    "src.dashboards.m4_dashboard",
    "src.dashboards.m3.overview_tab",
    "src.dashboards.m3.models_tab",
    "src.dashboards.m3.model_comparison_tab",
    "src.dashboards.m3.inference_demo_tab",
    "src.dashboards.m2_dashboard",
    "src.dashboards.m2.problem_understanding_tab",
    "src.dashboards.m2.literature_review_tab",
    "src.dashboards.m2.market_review_tab",
    "src.dashboards.m2.dataset_eda_tab",
    "src.dashboards.operations_learning",
    "src.dashboards.central_control",
]


def test_no_persisted_classifier_artifacts_today():
    # The project does not persist sklearn classifiers, so classification
    # inference is not runnable and the demo shows a missing-artifact state.
    from src.dashboards import model_helpers as mh

    assert mh.runnable_classification_models() == []


def test_importing_dashboards_does_not_import_heavy_ml():
    # Importing the dashboard modules must not pull in ultralytics or torch.
    # (Detector loading is lazy inside src.inference.) cv2 / sklearn are not
    # asserted here, as they can be pulled in indirectly by UI/plotting deps.
    import importlib

    for name in DASHBOARD_MODULES:
        importlib.import_module(name)

    assert "ultralytics" not in sys.modules
    assert "torch" not in sys.modules


def test_central_control_incident_yolo_button_and_lazy_imports():
    # The YOLO11s fire/smoke detector button lives in the Incident Assistant, and
    # detector/groq loading stays lazy (no module-level heavy imports).
    import ast
    from pathlib import Path

    from src.dashboards import central_control as cc

    assert callable(cc._tab_incident_assistant)

    text = Path(cc.__file__).read_text(encoding="utf-8")
    assert "Run YOLO11s fire/smoke detector" in text  # button lives here, not Image Zones

    tree = ast.parse(text)
    top_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module or "")
    heavy = {"ultralytics", "torch", "groq"}
    assert not any(m and m.split(".")[0] in heavy for m in top_imports)


def test_central_control_has_segmentation_refiner():
    # Image Zones offers segmentation-assisted polygon refinement, and it is wired
    # into both the AI-vision and manual panels. The backend (src/segmentation_assist)
    # is imported lazily, so no heavy segmentation library loads at module import.
    import ast
    from pathlib import Path

    from src.dashboards import central_control as cc

    assert callable(cc._render_segmentation_refiner)

    text = Path(cc.__file__).read_text(encoding="utf-8")
    assert "Refine selected box with segmentation" in text
    assert "Selected ROI box" in text
    assert text.count("_render_segmentation_refiner(") >= 3  # 1 def + AI panel + manual panel

    # cv2 must not be imported at module top level (segmentation is lazy / on-click).
    tree = ast.parse(text)
    top_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module or "")
    assert not any(m and m.split(".")[0] == "cv2" for m in top_imports), top_imports


def test_segmentation_assist_pure_helpers_run_without_cv2():
    # The helper module imports with cv2/numpy lazy, and its pure box→polygon
    # helpers must produce correct results without any segmentation backend.
    import importlib

    mod = importlib.import_module("src.segmentation_assist")
    # validate_roi_box reorders reversed corners and clamps out-of-range values.
    assert mod.validate_roi_box([0.8, 0.9, 0.2, 0.3]) == {
        "x_min": 0.2, "y_min": 0.3, "x_max": 0.8, "y_max": 0.9,
    }
    # polygon_from_box_fallback returns the 4 clockwise corners of the box.
    poly = mod.polygon_from_box_fallback({"x_min": 0.2, "y_min": 0.3, "x_max": 0.6, "y_max": 0.7})
    assert poly == [
        {"x": 0.2, "y": 0.3}, {"x": 0.6, "y": 0.3},
        {"x": 0.6, "y": 0.7}, {"x": 0.2, "y": 0.7},
    ]

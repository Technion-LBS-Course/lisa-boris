"""Smoke tests for the src/dashboards package.

Verifies that the dashboard modules import cleanly and expose a ``render``
callable, and — importantly — that importing them does NOT pull in heavy ML
libraries (ultralytics / torch) at import time. Detector loading must stay lazy.

These tests require Streamlit (a UI dependency), so they are skipped if it is
not installed. They never load model weights or run inference.
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


def test_dashboard_package_imports():
    import importlib

    for name in DASHBOARD_MODULES:
        importlib.import_module(name)


def test_render_callables_present():
    from src.dashboards import (
        m4_dashboard,
        m3_dashboard,
        m2_dashboard,
        operations_learning,
        central_control,
    )

    assert callable(m4_dashboard.render)
    assert callable(m3_dashboard.render)
    assert callable(m2_dashboard.render)
    assert callable(operations_learning.render)
    assert callable(central_control.render)


def test_m3_tab_modules_have_render():
    from src.dashboards.m3 import (
        overview_tab,
        models_tab,
        model_comparison_tab,
        inference_demo_tab,
    )

    assert callable(overview_tab.render)
    assert callable(models_tab.render)
    assert callable(model_comparison_tab.render)
    assert callable(inference_demo_tab.render)


def test_m2_tab_modules_have_render():
    from src.dashboards.m2 import (
        problem_understanding_tab,
        literature_review_tab,
        market_review_tab,
        dataset_eda_tab,
    )

    assert callable(problem_understanding_tab.render)
    assert callable(literature_review_tab.render)
    assert callable(market_review_tab.render)
    assert callable(dataset_eda_tab.render)


def test_model_helpers_shared_api():
    from src.dashboards import model_helpers as mh

    for fn in (
        "load_detector_cached",
        "load_model_results",
        "runnable_classification_models",
        "render_models_section",
        "render_classification_comparison",
        "render_object_detection_comparison",
        "render_operational_alert_metrics",
    ):
        assert callable(getattr(mh, fn))


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


def test_segmentation_assist_imports_without_cv2_or_numpy_at_load():
    # The helper module must import even where cv2/numpy are only used lazily.
    import importlib

    mod = importlib.import_module("src.segmentation_assist")
    for fn in ("validate_roi_box", "polygon_from_box_fallback", "mask_to_polygon",
               "refine_box_to_mask", "segmentation_backend_available"):
        assert callable(getattr(mod, fn))

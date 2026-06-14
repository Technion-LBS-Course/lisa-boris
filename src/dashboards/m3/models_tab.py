"""M3 Dashboard — Models tab.

Shows the five models (sklearn image-level classifiers + YOLO11 object detectors)
via the shared ``model_helpers`` rendering. Classification and object detection are
kept clearly separate; no comparison tab is appended here (the M3 dashboard has a
dedicated Model comparison (KPI) tab).
"""
import streamlit as st

from src.dashboards import model_helpers as mh


def render():
    st.header("Models")
    st.caption(
        "Five models: the sklearn image-level classifiers (DummyClassifier, Logistic "
        "Regression, Random Forest) and the YOLO11 object detectors (YOLO11n baseline / "
        "fallback, YOLO11s current primary detector). Image-level classification and "
        "object detection are kept clearly separate and are never mixed."
    )
    results_data = mh.load_model_results()
    if not results_data:
        st.warning(
            "No model result files found in `results/`. "
            "Run `python scripts/dummy_try.py` to generate baseline results."
        )
    else:
        mh.render_models_section(results_data, include_comparison=False)

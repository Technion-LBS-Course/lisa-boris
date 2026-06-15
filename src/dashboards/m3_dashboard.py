"""M3 Dashboard — thin orchestrator.

Builds the four M3 tabs and delegates each to its own module in
``src.dashboards.m3``:

  1. Overview              — the model-selection story (which models, KPI rule, why YOLO11s)
  2. Models                — the five models (classifiers + YOLO11 detectors)
  3. Model comparison (KPI) — KPI / operational, classification, object-detection (separated)
  4. Demo                  — upload an image and run the available fine-tuned detectors

Shared model-rendering logic lives in ``src.dashboards.model_helpers``. No metric
values are invented; everything comes from the measured result files in ``results/``.
Heavy ML libraries stay lazy.
"""
import streamlit as st

from src.dashboards.m3 import (
    overview_tab,
    models_tab,
    model_comparison_tab,
    inference_demo_tab,
)


def render(confidence_threshold, confirmation_frames):
    tab_overview, tab_models, tab_comparison, tab_inference = st.tabs([
        "Overview",
        "Models",
        "Model comparison (KPI)",
        "Demo",
    ])

    with tab_overview:
        overview_tab.render()
    with tab_models:
        models_tab.render()
    with tab_comparison:
        model_comparison_tab.render()
    with tab_inference:
        inference_demo_tab.render(confidence_threshold, confirmation_frames)
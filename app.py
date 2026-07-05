"""PyroFinder — Streamlit entry point (multi-dashboard shell).

This module is a thin shell: it sets page config, injects the theme, builds the
sidebar (dashboard mode + shared controls), and dispatches to the selected
dashboard renderer in ``src/dashboards/``. All dashboard rendering lives in those
modules. Heavy ML libraries are never imported here — detector loading stays lazy
inside ``src.inference`` and inside the inference-demo code paths.
"""
import streamlit as st
from pathlib import Path

from src.model import get_model_plan
from src.ui import inject_pyrofinder_theme

from src.dashboards import (
    m4_dashboard,
    m3_dashboard,
    m2_dashboard,
    operations_learning,
    central_control,
)

st.set_page_config(page_title="PyroFinder", layout="wide")
inject_pyrofinder_theme(
    background_video_path=Path("design_images") / "Nordic_Forest_LowPolymp_.mp4",
    use_video_background=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("PyroFinder")
    st.caption("Fire detection and monitoring — existing cameras only")

    st.divider()

    mode = st.selectbox(
        "Dashboard mode",
        [
            "M4 Dashboard",
            "M3 Dashboard",
            "M2 Dashboard",
            "Operations & Learning Dashboard",
            "Central Control Dashboard",
        ],
    )

    st.divider()
    model_plan = get_model_plan()
    st.caption(f"Primary model: **{model_plan['primary_model']}**")
    st.caption(f"Baseline: {model_plan['baseline_model']}")
    st.caption(f"Classes: {', '.join(model_plan['classes'])}")

    # The inference-demo controls are shared by the M3 and Operations & Learning
    # dashboards. Defaults are defined for the other modes so the renderers always
    # receive a value.
    if mode in ("M3 Dashboard", "Operations & Learning Dashboard"):
        st.divider()
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        confirmation_frames = st.number_input(
            "Confirmation frames (N)",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Alert triggered only after fire/smoke detected in N consecutive frames.",
        )
    else:
        confidence_threshold = 0.5
        confirmation_frames = 3

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("## 🔥 PyroFinder")
st.caption("Real-time fire and smoke detection using existing cameras.")

if mode == "M4 Dashboard":
    m4_dashboard.render()
elif mode == "M3 Dashboard":
    m3_dashboard.render(confidence_threshold, confirmation_frames)
elif mode == "Operations & Learning Dashboard":
    operations_learning.render(confidence_threshold, confirmation_frames)
elif mode == "Central Control Dashboard":
    central_control.render()
elif mode == "M2 Dashboard":
    m2_dashboard.render()

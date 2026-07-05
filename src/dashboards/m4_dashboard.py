"""M4 Dashboard renderer.

M4 surfaces four operational tabs — Camera Metadata, Image Zones, Incident
Assistant, and Risk Advisory — by reusing the implementations from the Central
Control dashboard. The tabs share Central Control's session state (cc_* keys),
so a camera or zone configured in one dashboard is visible in the other. Only one
dashboard mode renders per run, so the shared widget keys never collide.

No module-level ML imports: the reused tabs keep YOLO/Groq loading lazy.
"""
from __future__ import annotations

import streamlit as st

from src.dashboards import central_control as cc


def render() -> None:
    st.header("M4 Dashboard")
    st.caption(
        "Operational tabs — camera setup, image zones, incident assistance and "
        "risk advisory. Configuration is shared with the Central Control dashboard."
    )

    cc._init_state()

    # A keyed segmented control (not st.tabs) so the active section is stored in session
    # state and survives every rerun — uploads, sliders and other interactions no longer
    # bounce back to the first section the way st.tabs does.
    sections = ["Camera Metadata", "Image Zones", "Incident Assistant", "Risk Advisory"]
    active = st.segmented_control(
        "Section", sections, key="m4_section", default=sections[0],
        label_visibility="collapsed",
    ) or sections[0]

    if active == "Camera Metadata":
        # The shared camera-frame uploader + import-config live in this tab in M4.
        # This reference frame is used by Image Zones; the Incident Assistant's demo
        # sequence is separate and does not overwrite it.
        cc._frame_uploader(with_sequence=False)
        cc._import_config_panel()
        st.markdown("---")
        cc._tab_camera_metadata()
    elif active == "Image Zones":
        cc._tab_image_zones()
    elif active == "Incident Assistant":
        cc._tab_incident_assistant(
            show_intro=False, allow_manual_point=False, sequence_view=True, show_drafts=False
        )
    elif active == "Risk Advisory":
        cc._tab_risk_advisory()

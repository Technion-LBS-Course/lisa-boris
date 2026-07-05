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
    # The demo image sequence is loaded from inside the Incident Assistant tab in M4 and
    # drives only the incident's own detection. Image Zones / Camera Metadata keep the
    # stable reference frame (the first sequence frame), so scrubbing the incident slider
    # does not change the Image Zones frame.
    cc._frame_uploader(with_sequence=False)
    cc._import_config_panel()
    st.markdown("---")

    tab_cam, tab_zones, tab_incident, tab_risk = st.tabs(
        ["Camera Metadata", "Image Zones", "Incident Assistant", "Risk Advisory"]
    )
    with tab_cam:
        cc._tab_camera_metadata()
    with tab_zones:
        cc._tab_image_zones()
    with tab_incident:
        cc._tab_incident_assistant(
            show_intro=False, allow_manual_point=False, sequence_view=True, show_drafts=False
        )
    with tab_risk:
        cc._tab_risk_advisory()

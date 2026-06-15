"""M2 Dashboard — Problem Understanding tab.

Moved verbatim from m2_dashboard.py during the Phase 3b split. Content unchanged.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd

from src.data import get_primary_dataset_info, load_dfire_metadata, clean_dfire_metadata
from src.model import get_model_plan, get_metrics_plan
from src.eda import (
    compute_summary_metrics,
    compute_category_counts,
    compute_split_counts,
    compute_bbox_stats,
    filter_metadata,
    get_primary_eda_insight,
    compute_split_category_crosstab,
    compute_class_bbox_areas,
    compute_pixel_stats_by_category,
    get_numeric_cols,
    compute_correlation_matrix,
    compute_spatial_centers,
    compute_grid_distribution,
)
from src.viz import draw_yolo_boxes
from src.ui import apply_chart_theme, CAT_COLORS, PYRO_COLORS, SPLIT_COLORS, CLASS_COLORS


def render():
        st.header("Problem Understanding")

        # One-sentence problem + value proposition
        with st.container():
            st.markdown(
                "**Problem:** Property owners in fire-prone areas cannot monitor every camera "
                "feed at once — a small fire becomes a crisis before anyone notices."
            )
            st.markdown(
                "**Value proposition:** PyroFinder watches your existing cameras automatically "
                "and sends a confirmed alert within seconds of detecting fire or smoke."
            )

        st.divider()

        # KPI
        st.subheader("KPI")
        st.markdown(
            "The model is an object detector, the metric is F2-score, because it combines recall "
            "and precision while giving more weight to recall, since missing a real fire or smoke "
            "event is more costly than a false alarm, but too many false alerts reduce customer trust."
        )

        st.divider()

        # Stakeholder map
        st.subheader("Stakeholder Map")
        _sh_gap1, _sh_mid, _sh_gap2 = st.columns([1, 3, 1])
        with _sh_mid:
            st.components.v1.html("""
<!DOCTYPE html><html><head>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  body { margin:0; background:transparent; }
  .mermaid { background:transparent; opacity:0.7; }
  .mermaid svg { width:100% !important; max-width:100%; height:auto; }
</style>
</head><body>
<div class="mermaid">
%%{init:{"theme":"base","themeVariables":{"primaryColor":"#E4573D","primaryTextColor":"#F3F4F8","primaryBorderColor":"#F3F4F8","lineColor":"#D6D7E6","secondaryColor":"#3E445E","tertiaryColor":"#264036","edgeLabelBackground":"#2B3248","fontFamily":"Inter, Source Sans Pro, sans-serif","fontSize":"15px"}}}%%
quadrantChart
    title Stakeholder Map — Interest vs Influence
    x-axis Low Influence --> High Influence
    y-axis Low Interest --> High Interest
    quadrant-1 Manage Closely
    quadrant-2 Keep Informed
    quadrant-3 Monitor
    quadrant-4 Keep Satisfied
    Property Owner / Dani: [0.80, 0.88]
    Dev / ML Team: [0.72, 0.78]
    Operator / Admin: [0.88, 0.70]
    Farm Workers / Residents: [0.22, 0.72]
    Emergency Services: [0.78, 0.32]
    Camera Vendor / Integrator: [0.65, 0.18]
    Dataset / Research Sources: [0.20, 0.14]
</div>
<script>mermaid.initialize({startOnLoad:true,securityLevel:"loose"});</script>
</body></html>
""", height=500)

        _sh_data = [
            ("Property Owner / Dani",     "Manage Closely",  "Primary user — directly affected by alerts and fire risk",                "Weekly demos, usability feedback, alert UX review"),
            ("Dev / ML Team",             "Manage Closely",  "Builds and improves the detection model and dashboard",                   "Sprint planning, model performance reviews"),
            ("Operator / Admin",          "Manage Closely",  "Runs the system, manages cameras and alert configuration",                "Ops documentation, alert tuning, incident log review"),
            ("Farm Workers / Residents",  "Keep Informed",   "Affected by fire risk but do not control the system",                     "Clear alert language, evacuation guidance"),
            ("Emergency Services",        "Keep Satisfied",  "High authority in fire response; PyroFinder does not auto-dispatch",      "Share detection reports on request; future viewer dashboard"),
            ("Camera Vendor / Integrator","Keep Satisfied",  "Provides hardware PyroFinder depends on; limited day-to-day interest",    "Integration specs, compatibility requirements"),
            ("Dataset / Research Sources","Monitor",         "Enables model training; no active role in operations",                    "Citation, license compliance, periodic dataset updates"),
        ]
        _sh_tabs = st.tabs([row[0] for row in _sh_data])
        for _tab, (_name, _quadrant, _reason, _strategy) in zip(_sh_tabs, _sh_data):
            with _tab:
                st.markdown(f"**Quadrant:** {_quadrant}")
                st.markdown(f"**Reason:** {_reason}")
                st.markdown(f"**Communication strategy:** {_strategy}")

        st.divider()

        # Persona card
        st.subheader("Primary Persona")
        col_img, col_bio = st.columns([1, 3])
        with col_img:
            _persona_path = Path("design_images") / "DANI_PERSONA.png"
            if _persona_path.exists():
                import base64 as _b64mod
                _persona_b64 = _b64mod.b64encode(_persona_path.read_bytes()).decode()
                st.markdown(
                    f"<div style='width:150px;height:150px;border-radius:50%;overflow:hidden;"
                    f"margin:0 auto;box-shadow:0 0 0 3px rgba(228,87,61,0.5);'>"
                    f"<img src='data:image/png;base64,{_persona_b64}' "
                    f"style='width:100%;height:100%;object-fit:cover;display:block;' /></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='background:linear-gradient(135deg,#d4691e,#8b4513);"
                    "width:150px;height:150px;border-radius:50%;"
                    "display:flex;align-items:center;justify-content:center;"
                    "font-size:52px;margin:0 auto;'>🧑‍🌾</div>",
                    unsafe_allow_html=True,
                )
        with col_bio:
            with st.container():
                st.markdown("**Dani Cohen** — Farm Owner, Avivim, Israel")
                st.markdown("- 120-dunam agricultural farm with fixed outdoor cameras at boundary points")
                st.markdown("- Cannot continuously watch every feed during the dry summer months")
                st.markdown("- Risk: fires from neighbouring fields or agricultural machinery")
                st.markdown("- Goal: be alerted within minutes of any confirmed fire or smoke")

        st.divider()

        # Before / After journey
        st.subheader("User Journey")
        journey_before, journey_after = st.tabs(["Before PyroFinder", "After PyroFinder"])

        with journey_before:
            _jb_text, _jb_img = st.columns([1, 1], gap="small")
            with _jb_text:
                for _step_title, _step_body in [
                    ("Fire starts", "A spark from neighbouring machinery ignites dry brush at the property edge."),
                    ("No alert", "Dani's cameras capture smoke — but nobody is watching the screens."),
                    ("Late discovery", "Dani notices smoke from a window or gets a call from a neighbour — 15–30 minutes later."),
                    ("Crisis", "By the time emergency services arrive, the fire has already spread."),
                ]:
                    with st.container():
                        st.markdown(f"**{_step_title}**")
                        st.write(_step_body)
            with _jb_img:
                st.image("design_images/User_Journey_before.png", use_container_width=True)

        with journey_after:
            _ja_text, _ja_img = st.columns([1, 1], gap="small")
            with _ja_text:
                for _step_title, _step_body in [
                    ("Fire starts", "Same spark at the property edge."),
                    ("Detected", "NN object detection model detects smoke in the camera frame within seconds."),
                    ("Confirmed", "Detection confirmed across N consecutive frames — single-frame noise filtered out."),
                    ("Alert sent", "Dani receives an alert: camera ID, timestamp, approximate location, direction."),
                    ("Fast response", "Dani contacts emergency services within minutes. Fire is contained early."),
                ]:
                    with st.container():
                        st.markdown(f"**{_step_title}**")
                        st.write(_step_body)
            with _ja_img:
                st.image("design_images/User_Journey_after.png", use_container_width=True)

        st.divider()

        # Detection flow
        st.subheader("Detection Flow")
        st.caption(
            "Signal-to-alert pipeline — from existing RGB camera feed to reviewable alert record."
        )
        st.components.v1.html("""
<!DOCTYPE html>
<html>
<head>
<style>
  :root {
    --pf-card: rgba(18, 28, 31, 0.88);
    --pf-card-soft: rgba(23, 33, 37, 0.74);
    --pf-border: rgba(116, 151, 158, 0.55);
    --pf-border-soft: rgba(116, 151, 158, 0.28);
    --pf-text: #E8E3D8;
    --pf-muted: #9AA3A0;
    --pf-ember: #C8643F;
    --pf-cyan: #83C5BE;
    --pf-line: rgba(157, 177, 179, 0.70);
  }

  body {
    margin: 0;
    background: transparent;
    font-family: Inter, "Source Sans Pro", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--pf-text);
  }

  .pf-flow {
    padding: 16px 18px 18px;
    border: 1px solid var(--pf-border-soft);
    border-radius: 18px;
    background:
      radial-gradient(circle at 12% 0%, rgba(200, 100, 63, 0.16), transparent 28%),
      linear-gradient(180deg, rgba(12, 18, 22, 0.70), rgba(8, 12, 15, 0.44));
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
  }

  .pf-header {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: baseline;
    margin-bottom: 12px;
  }

  .pf-kicker {
    color: var(--pf-cyan);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
  }

  .pf-note {
    color: var(--pf-muted);
    font-size: 13px;
  }

  .pf-row {
    display: flex;
    align-items: stretch;
    gap: 8px;
    width: 100%;
  }

  .pf-row + .pf-row {
    margin-top: 10px;
  }

  .pf-step,
  .pf-reject {
    min-width: 0;
    flex: 1 1 0;
    padding: 12px 12px 11px;
    border-radius: 14px;
  }

  .pf-step {
    border: 1px solid var(--pf-border);
    background: linear-gradient(180deg, var(--pf-card), var(--pf-card-soft));
    position: relative;
  }

  .pf-step.core {
    border-color: rgba(200, 100, 63, 0.74);
    box-shadow: inset 0 0 0 1px rgba(200, 100, 63, 0.22);
  }

  .pf-step.output {
    border-color: rgba(131, 197, 190, 0.82);
    box-shadow: inset 0 0 0 1px rgba(131, 197, 190, 0.18);
  }

  .pf-num {
    display: inline-block;
    color: var(--pf-ember);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    margin-bottom: 7px;
  }

  .pf-step h4 {
    margin: 0 0 5px;
    font-size: 14px;
    line-height: 1.12;
    font-weight: 700;
    white-space: nowrap;
  }

  .pf-step p {
    margin: 0;
    color: var(--pf-muted);
    font-size: 12px;
    line-height: 1.3;
  }

  .pf-arrow {
    width: 22px;
    min-width: 22px;
    position: relative;
    align-self: center;
    height: 2px;
    background: var(--pf-line);
  }

  .pf-arrow::after {
    content: "";
    position: absolute;
    right: -1px;
    top: -4px;
    width: 0;
    height: 0;
    border-left: 8px solid var(--pf-line);
    border-top: 5px solid transparent;
    border-bottom: 5px solid transparent;
  }

  .pf-turn {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    margin: 5px 0 4px;
    padding-right: calc(12.5% + 11px);
    color: var(--pf-muted);
    font-size: 12px;
  }

  .pf-turn span {
    border-left: 1px solid var(--pf-line);
    border-bottom: 1px solid var(--pf-line);
    border-radius: 0 0 0 10px;
    padding: 7px 0 5px 12px;
  }

  .pf-reject {
    border: 1px dashed rgba(154, 163, 160, 0.48);
    background: rgba(10, 15, 18, 0.42);
    color: var(--pf-muted);
    font-size: 12px;
    line-height: 1.32;
  }

  .pf-reject strong {
    color: var(--pf-text);
    font-weight: 700;
  }

  @media (max-width: 900px) {
    .pf-step h4 { white-space: normal; }
    .pf-step p, .pf-reject { font-size: 11.5px; }
    .pf-arrow { width: 16px; min-width: 16px; }
  }
</style>
</head>
<body>
  <section class="pf-flow">
    <div class="pf-header">
      <div class="pf-kicker">Signal-to-alert pipeline</div>
      <div class="pf-note">No owner alert is sent before temporal confirmation.</div>
    </div>

    <div class="pf-row">
      <div class="pf-step">
        <span class="pf-num">01</span>
        <h4>Camera Input</h4>
        <p>Existing RGB security feed</p>
      </div>

      <div class="pf-arrow"></div>

      <div class="pf-step">
        <span class="pf-num">02</span>
        <h4>Frame Sampling</h4>
        <p>Periodic frame extraction</p>
      </div>

      <div class="pf-arrow"></div>

      <div class="pf-step core">
        <span class="pf-num">03</span>
        <h4>NN Detection Model</h4>
        <p>Fire / smoke + confidence</p>
      </div>

      <div class="pf-arrow"></div>

      <div class="pf-step core">
        <span class="pf-num">04</span>
        <h4>N-frame Confirmation</h4>
        <p>Filters single-frame noise</p>
      </div>
    </div>

    <div class="pf-turn">
      <span>confirmed signal continues to mapping and alerting</span>
    </div>

    <div class="pf-row">
      <div class="pf-reject">
        <strong>Not confirmed:</strong><br/>
        single-frame or non-persistent detections are ignored as operational alerts.
      </div>

      <div class="pf-step">
        <span class="pf-num">05</span>
        <h4>Mapping Layer</h4>
        <p>Approx. zone + direction</p>
      </div>

      <div class="pf-arrow"></div>

      <div class="pf-step output">
        <span class="pf-num">06</span>
        <h4>Alert Record</h4>
        <p>Camera · time · class · location</p>
      </div>

      <div class="pf-arrow"></div>

      <div class="pf-step output">
        <span class="pf-num">07</span>
        <h4>Dashboard Log</h4>
        <p>Review · confirm · reject</p>
      </div>
    </div>
  </section>
</body>
</html>
""", height=360)



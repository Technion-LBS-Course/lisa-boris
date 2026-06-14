"""M2 Dashboard — Market Review tab.

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
        st.header("Market Review — Fire Detection Solutions")
        st.markdown(
            "Fire detection is an active market with several competing approaches. "
            "Most current solutions depend on **dedicated infrastructure** — new towers, sensors, drones, or satellites. "
            "PyroFinder's market gap is using the customer's **existing security cameras** as the primary sensor, "
            "adding AI detection, alerting, and location estimation as a software layer."
        )

        # ── 1. Competitor overview ────────────────────────────────────────────
        st.subheader("1. Competitors")
        competitors = [
            {
                "name": "Pano AI",
                "url": "https://www.pano.ai/solution",
                "summary": "AI wildfire detection using dedicated 360° panoramic camera stations, cloud AI, human review, and GIS/weather integrations — targeting fire agencies, utilities, and large landowners.",
            },
            {
                "name": "FIREWAVE",
                "url": "https://www.firewave.earth/",
                "summary": "Acoustic wildfire detection using a specialized IoT sensor network trained to recognise fire sound signatures — targeting forests, parks, and camera-blind outdoor areas.",
            },
            {
                "name": "CANDO",
                "url": "https://cando.co.il/en/",
                "summary": "Autonomous drone-in-a-box systems for security, public safety, and inspection — targeting municipalities and public-safety organisations that can manage drone operations.",
            },
            {
                "name": "OroraTech",
                "url": "https://ororatech.com/all-products/wildfire-solution",
                "summary": "Satellite-based wildfire intelligence with hotspot detection, spread analytics, and risk layers — targeting national agencies, utilities, and large-scale land managers.",
            },
            {
                "name": "FireDome",
                "url": "https://www.fire-dome.com/",
                "summary": "Automated wildfire suppression for high-value assets, combining visual/thermal detection with a mechanical launcher that deploys fire-retardant capsules.",
            },
        ]
        _MS = Path("data/market-survey")
        _img_map = {
            "Pano AI":    [("pano-ai-tower.png",       "Pano AI — dedicated camera station"),
                           ("pano-ai-alert.png",        "Pano AI — alert & map workflow")],
            "FIREWAVE":   [("firewave-sensor.png",      "FIREWAVE — acoustic sensor"),
                           ("firewave - map.png",        "FIREWAVE — sensor map")],
            "CANDO":      [("CANDO - sensor.png",       "CANDO — drone station"),
                           ("CANDO-map.png",             "CANDO — operational map")],
            "OroraTech":  [("ororaTech - sensor.png",   "OroraTech — satellite sensor"),
                           ("ororaTech - map.png",       "OroraTech — wildfire map")],
            "FireDome":   [("FireDome - Launcher.png",  "FireDome — suppression launcher"),
                           ("firedome - map.png",        "FireDome — deployment map")],
        }
        for i in range(0, len(competitors), 2):
            _pair = competitors[i:i + 2]
            _comp_cols = st.columns(2)
            for _ccol, c in zip(_comp_cols, _pair):
                with _ccol:
                    st.markdown(f"**[{c['name']}]({c['url']})**")
                    st.caption(c["summary"])
                    _imgs = _img_map.get(c["name"], [])
                    if _imgs:
                        _ic1, _ic2 = st.columns(2)
                        for _ic, (fn, cap) in zip([_ic1, _ic2], _imgs):
                            _p = _MS / fn
                            if _p.exists():
                                with _ic:
                                    st.image(str(_p), caption=cap, width=160)
            if i + 2 < len(competitors):
                st.divider()

        # ── 2. Comparison table ───────────────────────────────────────────────
        st.subheader("2. Comparison Table")
        import pandas as _pd_market
        comparison_data = {
            "Feature / Criterion": [
                "Uses existing security cameras",
                "Smoke / fire visual detection",
                "Predictive risk layer",
                "Alerting",
                "Location estimation",
                "Requires new hardware",
                "Target audience",
                "Public price",
            ],
            "PyroFinder": [
                "Yes — core idea",
                "Yes",
                "Planned",
                "Yes",
                "Camera-based map layer",
                "No (or minimal edge device)",
                "Sites with existing cameras",
                "Not yet defined",
            ],
            "Pano AI": [
                "Partial",
                "Yes",
                "Weather / GIS context",
                "Yes",
                "Yes, incl. triangulation",
                "Yes — dedicated stations",
                "Fire agencies, utilities",
                "Public price not listed",
            ],
            "FIREWAVE": [
                "No",
                "No — acoustic",
                "Limited",
                "Yes",
                "Sensor-network based",
                "Yes — acoustic sensors",
                "Forests, parks, blind spots",
                "Public price not listed",
            ],
            "CANDO": [
                "No",
                "Via drone payload",
                "Not core",
                "Operational",
                "Drone GPS / operator",
                "Yes — drones",
                "Municipalities, public safety",
                "Public price not listed",
            ],
            "OroraTech": [
                "No",
                "Indirect hotspot",
                "Yes — weather/vegetation/terrain",
                "Yes",
                "Satellite hotspot",
                "Yes - satellite ground stations",
                "National agencies, utilities",
                "Public price not listed",
            ],
            "FireDome": [
                "No",
                "Yes, for suppression trigger",
                "Not core",
                "Yes",
                "Local asset location",
                "Yes — sensors + launcher",
                "High-value assets, critical infra",
                "Public price not listed",
            ],
        }
        df_comparison = _pd_market.DataFrame(comparison_data).set_index("Feature / Criterion")
        st.dataframe(df_comparison, use_container_width=True)

        # ── 3. Design positioning ─────────────────────────────────────────────
        st.subheader("3. Design Positioning")
        st.markdown(
            "The chart maps each solution on two axes: "
            "**how much new infrastructure it requires** (left = dedicated, right = existing sensors) "
            "and **whether it focuses on detection/monitoring or active response/suppression** (bottom = detection, top = suppression)."
        )
        import plotly.graph_objects as _go_market
        players = [
            {"name": "Pano AI",    "x": 0.25, "y": 0.35},
            {"name": "FIREWAVE",   "x": 0.20, "y": 0.30},
            {"name": "CANDO",      "x": 0.30, "y": 0.65},
            {"name": "OroraTech",  "x": 0.40, "y": 0.45},
            {"name": "FireDome",   "x": 0.15, "y": 0.90},
            {"name": "PyroFinder", "x": 0.90, "y": 0.40},
        ]
        fig_pos = _go_market.Figure()
        fig_pos.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=0.5,
                          fillcolor="rgba(46,139,87,0.08)", line_width=0)
        fig_pos.add_annotation(x=0.75, y=0.06, text="Low-friction detection",
                               showarrow=False, font=dict(size=11, color="#2e8b57"), opacity=0.8)
        for p in players:
            is_pyro = p["name"] == "PyroFinder"
            fig_pos.add_trace(_go_market.Scatter(
                x=[p["x"]], y=[p["y"]],
                mode="markers+text",
                marker=dict(size=16 if is_pyro else 12,
                            color="#e63946" if is_pyro else "#e07b39",
                            line=dict(width=2, color="white")),
                text=[p["name"]],
                textposition="top center",
                textfont=dict(size=12, color="#e63946" if is_pyro else "#cccccc"),
                showlegend=False,
            ))
        fig_pos.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1,
                          line=dict(color="#555", dash="dot", width=1))
        fig_pos.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5,
                          line=dict(color="#555", dash="dot", width=1))
        fig_pos.update_layout(
            xaxis=dict(title="← Dedicated new infrastructure    |    Existing client-owned sensors →",
                       range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title="Detection / monitoring  ↕  Response / suppression",
                       range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(14, 18, 34, 0.50)",
            paper_bgcolor="rgba(14, 18, 34, 0.50)",
            font=dict(color="#cccccc"),
            height=420,
            margin=dict(l=60, r=20, t=20, b=60),
        )
        st.plotly_chart(fig_pos, use_container_width=True)
        st.markdown(
            "PyroFinder sits in the **low-friction detection** quadrant — camera-based early detection "
            "without requiring new infrastructure, distinct from tower solutions (Pano AI), "
            "acoustic networks (FIREWAVE), drone operations (CANDO), satellite platforms (OroraTech), "
            "and suppression systems (FireDome)."
        )

        # ── 4. Design insights ────────────────────────────────────────────────
        st.subheader("4. Design Insights")

        with st.container():
            st.markdown("**What to adopt**")
            st.markdown(
                "- **Map-first interface** — show every camera, detection, confidence score, and nearby assets\n"
                "- **Alert workflow** — suspected detection → AI confidence → user confirmation → notification\n"
                "- **Evidence view** — show the frame / clip that triggered the alert\n"
                "- **Risk context** — combine camera detection with wind, temperature, humidity, and time of day\n"
                "- **False-alarm handling** — collect user feedback to improve the model\n"
                "- **Multi-camera verification** — two cameras seeing the same smoke direction increases confidence"
            )

        st.markdown("**What to replace**")
        replace_data = {
            "Common competitor pattern": [
                "Buy and install new detection towers",
                "Deploy a new acoustic sensor network",
                "Depend only on satellite refresh cycles",
                "Offer detection without customer workflow",
            ],
            "PyroFinder approach": [
                "Connect existing security cameras",
                "Use already available visual streams first",
                "Use continuous local camera streams",
                "Provide alert review, escalation, history, and exportable incident reports",
            ],
        }
        st.dataframe(_pd_market.DataFrame(replace_data), use_container_width=True, hide_index=True)


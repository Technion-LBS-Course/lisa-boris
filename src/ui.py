"""
PyroFinder — shared UI helpers and Plotly chart theme.
Call apply_chart_theme(fig) on every Plotly figure before st.plotly_chart().
"""
import streamlit as st
from pathlib import Path

PYRO_UI_COLORS = {
    "twilight_sky":  "#1E2336",
    "deep_fjord":    "#2B3248",
    "stone_surface": "#3E445E",
    "dusk_rose":     "#E07A8A",
    "nordic_lilac":  "#8F8CC7",
    "pine_shadow":   "#264036",
    "morning_mist":  "#D6D7E6",
    "frost_white":   "#F3F4F8",
    "raven_gray":    "#A5A8B8",
    "ember_glow":    "#E4573D",
    "hud_cyan":      "#8CE9FF",
}

PYRO_GLASS = {
    "main_panel":         "rgba(30, 35, 54, 0.58)",
    "sidebar":            "rgba(30, 35, 54, 0.88)",
    "card":               "rgba(62, 68, 94, 0.64)",
    "card_hover":         "rgba(62, 68, 94, 0.78)",
    "soft_border":        "rgba(214, 215, 230, 0.16)",
    "strong_border":      "rgba(214, 215, 230, 0.30)",
    "background_overlay": "rgba(14, 18, 34, 0.64)",
}

PYRO_COLORS = {
    "fire":           "#E4573D",   # ember glow — fire detections
    "smoke":          "#A5A8B8",   # raven gray — smoke detections
    "fire_and_smoke": "#C44B30",   # deep ember — combined category
    "background":     "#3E445E",   # stone surface — background/negative class
    "primary":        "#E4573D",   # ember glow — UI accent
    "bg":             "#1E2336",   # twilight sky — chart plot area
    "card_bg":        "#2B3248",   # deep fjord — chart paper/card
    "text":           "#F3F4F8",   # frost white — all text
    "grid":           "#3E445E",   # stone surface — grid lines
    "train":          "#8F8CC7",   # nordic lilac — train split
    "test":           "#E07A8A",   # dusk rose — test split
}

CAT_COLORS = {
    "fire_only":      PYRO_COLORS["fire"],
    "smoke_only":     PYRO_COLORS["smoke"],
    "fire_and_smoke": PYRO_COLORS["fire_and_smoke"],
    "background":     PYRO_COLORS["background"],
}

SPLIT_COLORS = {
    "train": PYRO_COLORS["train"],
    "test":  PYRO_COLORS["test"],
}

CLASS_COLORS = {
    "fire":  PYRO_COLORS["fire"],
    "smoke": PYRO_COLORS["smoke"],
}


@st.cache_data(show_spinner=False)
def _load_video_base64(path_str: str, mtime_ns: int) -> str:
    import base64
    from pathlib import Path
    return base64.b64encode(Path(path_str).read_bytes()).decode("utf-8")


def inject_pyrofinder_theme(
    background_video_path=Path("design_images") / "Nordic_Forest_LowPolymp_.mp4",
    use_video_background: bool = True,
) -> None:
    """
    Injects the Röki-inspired Nordic visual theme into the Streamlit app.

    Layering strategy (NEGATIVE z-index model):
      * The <video> is injected via st.markdown, so it lives INSIDE
        .block-container. To act as a true viewport-fixed background it must
        NOT be trapped by any ancestor stacking/containing context. Therefore
        backdrop-filter / transform / z-index are kept OFF every ancestor of
        the video (.block-container, stMain, stAppViewContainer, stApp) and the
        glass-blur is applied only to LEAF cards (metrics, expanders, tabs,
        dataframes) which are not ancestors of the video.
      * Video sits at z-index:-2, a dark overlay at z-index:-1, all real
        content paints above them in normal flow.
      * position:fixed + 100vw/100vh + object-fit:cover keeps the video pinned
        to the viewport and full-bleed from the very top while content scrolls.

    If the video file is missing, falls back to a fixed full-screen gradient.
    Never crashes — always degrades gracefully.
    """
    from pathlib import Path
    import streamlit as st

    video_path = Path(background_video_path)
    bg_layers_html = ""

    if use_video_background and video_path.exists():
        try:
            mtime_ns = int(video_path.stat().st_mtime_ns)
            b64 = _load_video_base64(str(video_path), mtime_ns)
            bg_layers_html = f"""
<video autoplay muted loop playsinline id="pyrofinder-bg-video">
    <source src="data:video/mp4;base64,{b64}" type="video/mp4">
</video>
<div id="pyrofinder-bg-overlay"></div>
"""
        except Exception:
            bg_layers_html = ""

    # Fixed full-screen gradient layer when no video is available.
    if not bg_layers_html:
        bg_layers_html = '<div id="pyrofinder-bg-fallback"></div>'

    css = f"""
<style>
/* ── Page scrolls normally; the fixed background layers never move ── */
html, body {{
    background: transparent !important;
    overflow-x: hidden !important;
}}

/* ── Strip backgrounds from EVERY ancestor of the injected video so it shows
      through. None of these may use backdrop-filter / transform / z-index,
      or the fixed video would be trapped inside their box. ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
[data-testid="stMainBlockContainer"],
.main .block-container,
.block-container,
[data-testid="stVerticalBlock"] {{
    background: transparent !important;
}}

/* ── Background video: fixed to the viewport, behind all content ── */
#pyrofinder-bg-video {{
    position: fixed !important;
    top: 0;
    left: 0;
    width: 100vw !important;
    height: 100vh !important;
    object-fit: cover !important;
    object-position: center center !important;
    z-index: -2 !important;
    pointer-events: none !important;
}}

/* ── Dark gradient overlay sits just above the video, still behind content ── */
#pyrofinder-bg-overlay {{
    position: fixed !important;
    top: 0;
    left: 0;
    width: 100vw !important;
    height: 100vh !important;
    z-index: -1 !important;
    pointer-events: none !important;
    background: linear-gradient(
        180deg,
        rgba(14, 18, 34, 0.35) 0%,
        rgba(14, 18, 34, 0.45) 45%,
        rgba(14, 18, 34, 0.55) 100%
    );
}}

/* ── Fallback gradient layer (used only when the video file is missing) ── */
#pyrofinder-bg-fallback {{
    position: fixed !important;
    top: 0;
    left: 0;
    width: 100vw !important;
    height: 100vh !important;
    z-index: -2 !important;
    pointer-events: none !important;
    background: linear-gradient(180deg, #1E2336 0%, #2B3248 55%, #264036 100%);
}}

/* ── Neutralise Streamlit chrome that would otherwise paint opaque strips ── */
[data-testid="stHeader"],
[data-testid="stToolbar"] {{
    background: transparent !important;
}}
[data-testid="stDecoration"] {{
    display: none !important;
}}

/* ── Rounded corners on every Plotly chart box ── */
[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] > div {{
    border-radius: 12px !important;
    overflow: hidden !important;
}}

/* ── Sidebar: glass panel — no border, lighter opacity ── */
[data-testid="stSidebar"] {{
    background: rgba(30, 35, 54, 0.70) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-right: none !important;
    box-shadow: none !important;
}}
[data-testid="stSidebar"] > div {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

/* ── Main content: fully transparent, flush edges so the video fills every corner ── */
.main .block-container,
.block-container {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding: 1.5rem 2rem 2rem 2rem;
}}

/* ── Leaf cards: very faint glass surface, no border ── */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: rgba(14, 18, 34, 0.10) !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 12px !important;
}}
div[data-testid="stMetric"] {{
    background: rgba(14, 18, 34, 0.12) !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem;
}}
div[data-testid="stExpander"] {{
    background: rgba(14, 18, 34, 0.10) !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 12px !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(30, 35, 54, 0.56) !important;
    border-radius: 16px;
    padding: 4px;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}}
.stTabs [data-baseweb="tab"] {{
    color: #A5A8B8;
    border-radius: 6px;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(228, 87, 61, 0.20);
    color: #F3F4F8;
}}
.stTabs [data-baseweb="tab-panel"] {{ background: transparent !important; }}
.stExpander {{
    border-radius: 18px;
}}
button[kind="primary"] {{
    background: #E4573D;
    border: none;
    color: #F3F4F8;
}}
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {{
    background: rgba(30, 35, 54, 0.72) !important;
    border-radius: 14px !important;
}}

/* ── Remove default frames/borders from st.image elements ── */
[data-testid="stImage"],
[data-testid="stImage"] > div,
[data-testid="stImage"] img {{
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}}
[data-testid="stImage"] {{
    padding: 0 !important;
}}
[data-testid="stImage"] img {{
    border-radius: 10px;
}}
</style>
{bg_layers_html}
"""
    st.markdown(css, unsafe_allow_html=True)


def apply_chart_theme(fig, title_font_size: int = 14) -> object:
    """Apply PyroFinder visual theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(30, 35, 54, 0.50)",   # twilight_sky @ 50%
        plot_bgcolor="rgba(43, 50, 72, 0.50)",    # deep_fjord @ 50%
        font=dict(color=PYRO_UI_COLORS["frost_white"], size=12, family="sans-serif"),
        title_font=dict(size=title_font_size, color=PYRO_UI_COLORS["frost_white"]),
        xaxis=dict(
            gridcolor=PYRO_COLORS["grid"],
            linecolor=PYRO_COLORS["grid"],
            tickcolor=PYRO_UI_COLORS["raven_gray"],
        ),
        yaxis=dict(
            gridcolor=PYRO_COLORS["grid"],
            linecolor=PYRO_COLORS["grid"],
            tickcolor=PYRO_UI_COLORS["raven_gray"],
        ),
        legend=dict(
            bgcolor="rgba(30, 35, 54, 0.50)",
            bordercolor=PYRO_COLORS["grid"],
            borderwidth=1,
        ),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_traces(
        selector=dict(type="bar"),
        marker_line_width=0,
        marker_line_color="rgba(43, 50, 72, 0.50)",
    )
    return fig

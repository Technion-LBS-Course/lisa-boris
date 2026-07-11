"""PyroFinder — Live Ops (default landing page).

Streamlit multipage entry, added alongside the existing app with no changes to the
other dashboards. Applies the light "tablet" theme (design variant 1b) and renders
the Live Ops dashboard. Heavy ML stays lazy inside the renderer.
"""
import streamlit as st

from src.dashboards import live_ops  # module import only — emits no Streamlit calls

st.set_page_config(page_title="PyroFinder", page_icon="🔥", layout="wide")

# Light "tablet" theme (variant 1b), scoped to this page. The app-wide dark video
# theme (app.py / src.ui) is NOT injected on this page — Streamlit runs each page as
# its own script — so this gives the light look without touching global config.
_LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root { --pf-accent:#d9481f; --pf-accent-hover:#c23e18; }

/* ── Light surfaces + Public Sans typography ── */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
section.main, .block-container, [data-testid="stVerticalBlock"] {
    background:#faf9f7 !important; color:#1c1b18 !important;
    font-family:'Public Sans', system-ui, -apple-system, sans-serif !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { background:transparent !important; }
[data-testid="stSidebar"] { background:#ffffff !important; border-right:1px solid rgba(0,0,0,.06); }
h1,h2,h3,h4,h5,p,label,li,span { color:#1c1b18; }
.block-container { padding:1.1rem 2rem 2rem; max-width:1400px; }
[data-testid="stCaptionContainer"], .stCaption, small { color:rgba(0,0,0,.5) !important; }

/* ── Cards / panels: white, rounded, hairline border ── */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stExpander"],
div[data-testid="stTable"], div[data-testid="stDataFrame"],
div[data-testid="stForm"] {
    background:#ffffff !important; border:1px solid rgba(0,0,0,.08) !important;
    border-radius:14px !important; box-shadow:none !important;
}
div[data-testid="stMetric"] {
    background:#ffffff !important; border:1px solid rgba(0,0,0,.08) !important;
    border-radius:14px !important; padding:.6rem .9rem;
}

/* ── Rounded map + image + video containers (all iframes/images on the page) ── */
iframe { border-radius:14px !important; }
[data-testid="stImage"] img { border-radius:14px !important; }
[data-testid="stImage"] { overflow:hidden; }

/* ── Buttons: clean light default, red primary, rounded (override dark theme) ── */
.stApp button[data-testid^="stBaseButton"] {
    border-radius:10px !important; font-weight:600 !important;
    background:#ffffff !important; color:#1c1b18 !important;
    border:1px solid rgba(0,0,0,.15) !important;
}
.stApp button[data-testid^="stBaseButton"]:hover { border-color:var(--pf-accent) !important; color:var(--pf-accent) !important; }
/* Primary + primary form-submit = flame red */
.stApp button[data-testid="stBaseButton-primary"],
.stApp button[data-testid="stBaseButton-primaryFormSubmit"] {
    background:var(--pf-accent) !important; color:#ffffff !important; border:none !important; font-weight:700 !important;
}
.stApp button[data-testid="stBaseButton-primary"]:hover,
.stApp button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
    background:var(--pf-accent-hover) !important; color:#ffffff !important;
}

/* ── Pill-style segmented navigation (top + step nav) ── */
.stApp [data-testid="stSegmentedControl"] { background:#f0ede8 !important; border-radius:12px !important; padding:3px !important; }
.stApp [data-testid="stSegmentedControl"] button,
.stApp button[data-testid="stBaseButton-segmented_control"] {
    border-radius:9px !important; border:none !important; font-weight:600 !important;
    color:rgba(0,0,0,.6) !important; background:transparent !important;
}
.stApp [data-testid="stSegmentedControl"] button[aria-checked="true"],
.stApp button[data-testid="stBaseButton-segmented_controlActive"] {
    background:var(--pf-accent) !important; color:#ffffff !important;
}

/* ── Chat bubbles: soft rounded ── */
[data-testid="stChatMessage"] { background:#f0ede8 !important; border-radius:12px; }

/* Monospace for code-ish captions to echo the mockup's data labels */
[data-testid="stMetricValue"] { font-family:'IBM Plex Mono', monospace; }
</style>
"""
st.markdown(_LIGHT_CSS, unsafe_allow_html=True)

live_ops.render()

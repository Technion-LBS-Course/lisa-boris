"""
PyroFinder — shared UI helpers and Plotly chart theme.
Call apply_chart_theme(fig) on every Plotly figure before st.plotly_chart().
"""

PYRO_COLORS = {
    "fire":           "#e05c00",   # ember orange — fire detections
    "smoke":          "#9aab89",   # sage green-grey — smoke detections
    "fire_and_smoke": "#c44b00",   # deep ember — combined category
    "background":     "#6b7c5a",   # moss green — background/negative class
    "primary":        "#e07b39",   # amber — UI accent
    "bg":             "#1a1f14",   # deep forest — chart plot area
    "card_bg":        "#252e1c",   # darker moss — chart paper/card
    "text":           "#dde8d0",   # light sage — all text
    "grid":           "#2e3828",   # subtle forest grid
    "train":          "#e07b39",   # amber — train split
    "test":           "#9aab89",   # sage — test split
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

def apply_chart_theme(fig, title_font_size: int = 14) -> object:
    """Apply PyroFinder visual theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=PYRO_COLORS["card_bg"],
        plot_bgcolor=PYRO_COLORS["bg"],
        font=dict(color=PYRO_COLORS["text"], size=12, family="sans-serif"),
        title_font=dict(size=title_font_size, color=PYRO_COLORS["text"]),
        xaxis=dict(
            gridcolor=PYRO_COLORS["grid"],
            linecolor=PYRO_COLORS["grid"],
            tickcolor=PYRO_COLORS["text"],
        ),
        yaxis=dict(
            gridcolor=PYRO_COLORS["grid"],
            linecolor=PYRO_COLORS["grid"],
            tickcolor=PYRO_COLORS["text"],
        ),
        legend=dict(
            bgcolor=PYRO_COLORS["card_bg"],
            bordercolor=PYRO_COLORS["grid"],
            borderwidth=1,
        ),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_traces(
        selector=dict(type="bar"),
        marker_line_width=0,
        marker_line_color=PYRO_COLORS["bg"],
    )
    return fig

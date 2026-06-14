"""M2 Dashboard — thin orchestrator.

Builds the four M2 story tabs and delegates each to its own module in
``src.dashboards.m2``:

  1. Problem Understanding
  2. Literature Review
  3. Market Review
  4. Dataset & EDA

This mirrors the ``src.dashboards.m3`` layout. The split is a behavior-preserving
code move: tab content is unchanged. Heavy ML libraries stay lazy.
"""
import streamlit as st

from src.dashboards.m2 import (
    problem_understanding_tab,
    literature_review_tab,
    market_review_tab,
    dataset_eda_tab,
)


def render():
    st.subheader("M2 Dashboard")

    tab_problem, tab_lit, tab_market, tab_eda_story = st.tabs([
        "1. Problem Understanding",
        "2. Literature Review",
        "3. Market Review",
        "4. Dataset & EDA",
    ])

    with tab_problem:
        problem_understanding_tab.render()
    with tab_lit:
        literature_review_tab.render()
    with tab_market:
        market_review_tab.render()
    with tab_eda_story:
        dataset_eda_tab.render()

"""M3 Dashboard — Model comparison (KPI) tab.

Splits the comparison into three clearly labeled sub-tabs that are never mixed:
KPI / Operational alert metrics, Classification, and Object-detection. All values
come from the measured result files in ``results/`` via the shared ``model_helpers``.
"""
import streamlit as st

from src.dashboards import model_helpers as mh


def render():
    st.header("Model comparison (KPI)")
    st.markdown(
        "The model is an object detector, the metric is F2-score, because it combines recall "
        "and precision while giving more weight to recall, since missing a real fire or smoke "
        "event is more costly than a false alarm, but too many false alerts reduce customer trust."
    )
    st.caption(
        "All values come from the measured result files in `results/`."
    )
    results_data = mh.load_model_results()
    if not results_data:
        st.warning("No model result files found in `results/`.")
        return

    sub_kpi, sub_clf, sub_det = st.tabs([
        "KPI / Operational alert metrics",
        "Classification",
        "Object-detection",
    ])
    with sub_kpi:
        mh.render_operational_alert_metrics(results_data)
    with sub_clf:
        mh.render_classification_comparison(results_data)
    with sub_det:
        mh.render_object_detection_comparison(results_data)

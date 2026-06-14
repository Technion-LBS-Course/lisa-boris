"""M3 Dashboard — Overview tab.

Summarizes the model-selection story: which models were compared, the KPI-based
selection criterion, and that YOLO11s is the selected primary detector because it has
the better Operational Alert Score than YOLO11n in the measured M3 comparison. YOLO11n
stays the lightweight baseline / fallback. Location outputs are approximate. The
selected detector is pulled live; no metric values are invented.
"""
import streamlit as st


def render():
    st.header("Overview — model comparison and selection")
    st.caption(
        "PyroFinder detects fire and smoke on existing cameras and produces "
        "approximate location."
    )

    st.subheader("Models compared")
    st.markdown(
        "- **DummyClassifier**, **Logistic Regression**, **Random Forest** — image-level "
        "sklearn classifiers using 60 handcrafted color features. They predict a single "
        "label per image (background / fire / smoke), not bounding boxes.\n"
        "- **YOLO11n** — the lightweight object-detection **baseline / fallback**.\n"
        "- **YOLO11s** — the larger object detector, the **current primary detector**.\n\n"
        "Image-level classification and object detection are different predictive tasks. "
        "sklearn Macro F1 is never placed in a direct ranking against YOLO11 mAP; the most "
        "defensible head-to-head comparison is YOLO11n vs YOLO11s on the same D-Fire test "
        "split at the same confidence threshold."
    )

    st.subheader("Selection criterion (KPI hierarchy)")
    st.markdown(
        "Selection follows a cost-sensitive operational rule, because missing a real "
        "fire/smoke hazard is far more costly than raising a false alarm (false negative "
        "weight 10, false positive weight 1):\n\n"
        "1. **Operational Alert Score** (primary decision metric — the cost-sensitive summary "
        "that already encodes Hazard Recall and False Alert Rate as its components).\n"
        "2. Object-detection **Recall** and **mAP@0.5** as supporting evidence.\n"
        "3. Measured **inference speed** as a practical consideration."
    )

    st.subheader("Selected detector")
    winner = _operational_winner()
    if winner:
        st.success(
            f"**Selected detector: {winner}.** Among the measured object detectors, "
            f"{winner} is selected because it has the better **Operational Alert Score** "
            "than YOLO11n in the M3 comparison (0.9406 vs 0.9368) — the cost-sensitive "
            "summary that weights a missed hazard 10× a false alert — with stronger "
            "object-detection Recall and mAP@0.5 as supporting evidence."
        )
    else:
        st.info(
            "No detector is selected yet: a model is chosen only when its measured "
            "operational result file exists with complete metrics. Pending, synthetic, "
            "malformed, or training-in-progress results can never win."
        )
    st.caption(
        "YOLO11n remains the lightweight baseline / fallback. See **Model comparison "
        "(KPI)** for the measured numbers and **Models** for per-model detail."
    )


def _operational_winner():
    """Return the selected detector name from measured operational files, or None.

    Uses the same selection rule and file specs as the operational-metrics renderer;
    selection is gated on existing, measured, complete result files.
    """
    from src.results_loader import select_operational_winner
    op_specs = [
        ("YOLO11n", "results/yolo11n_operational_metrics.json", "results/baseline_yolo11n.json"),
        ("YOLO11s", "results/yolo11s_operational_metrics.json", "results/baseline_yolo11s.json"),
    ]
    return select_operational_winner(
        [(m, op) for m, op, _ in op_specs],
        detection_items=[(m, det) for m, _, det in op_specs],
    )

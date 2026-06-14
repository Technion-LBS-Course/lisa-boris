"""M3 Dashboard — Overview tab.

Summarizes the model-selection story: which models were compared, the KPI-based
selection criterion, and that YOLO11s is selected only because measured result
files exist and it wins the documented operational hierarchy. YOLO11n stays the
lightweight baseline / fallback. Location outputs are approximate, never precise
geolocation. The selected detector is pulled live; no metric values are invented.
"""
import streamlit as st


def render():
    st.header("Overview — model comparison and selection")
    st.caption(
        "How PyroFinder chose its detector for M3. PyroFinder detects fire and smoke "
        "on existing cameras and produces approximate, image-space location estimates "
        "only — never precise geolocation."
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
        "1. **Hazard Recall** (primary) — higher is better.\n"
        "2. **False Alert Rate** (secondary) — lower is better.\n"
        "3. **Operational Alert Score** (ranking summary).\n"
        "4. Object-detection **Recall** and **mAP@0.5** as supporting evidence.\n"
        "5. Measured **inference speed** as a practical consideration."
    )

    st.subheader("Selected detector")
    winner = _operational_winner()
    if winner:
        st.success(
            f"**Selected detector: {winner}.** YOLO11s is selected **only because** its "
            "measured detection and operational result files exist and it wins the "
            "documented operational rule — higher Hazard Recall, lower False Alert Rate, "
            "then higher Operational Alert Score, with stronger supporting detection "
            "Recall / mAP@0.5. Without those measured files it would not be selectable."
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

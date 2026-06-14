"""M2 Dashboard — Literature Review tab.

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
        st.header("Literature Review")
        st.caption("All content traced to docs/Literature_review.md.")

        # ── Section 1: Research Question ─────────────────────────────────────
        st.subheader("Research Question")
        st.markdown(
            f"""<div style="border-left: 4px solid {PYRO_COLORS['primary']}; \
padding: 12px 20px; background-color: {PYRO_COLORS['card_bg']}; \
border-radius: 4px; margin-bottom: 12px;">
<em>"How can deep learning-based object detection be used to achieve accurate and fast \
real-time fire and wildfire detection from ordinary RGB civilian/security-camera \
video and images?"</em></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            "PyroFinder addresses this question directly by applying NN object detection model fire/smoke object "
            "detection to cameras already installed at the customer site, combining real-time "
            "inference with multi-frame confirmation and approximate map-based alerting — the "
            "product approach that the literature identifies as most suitable for ordinary RGB "
            "civilian surveillance."
        )

        st.divider()

        # ── Section 2: Comparison Table ──────────────────────────────────────
        st.subheader("Paper Comparison Table")
        _lit_table = pd.DataFrame([
            {
                "Paper": "Bahhar et al. (2023)",
                "Model": "Two-stage: ensemble CNN + YOLOv5s/YOLOv5l",
                "Key Result": "Classification accuracy 0.99, F1 0.95; mAP@0.5 0.85 (smoke), 0.76 (combined)",
                "Relation to PyroFinder": (
                    "Shows a staged pipeline can reduce unnecessary detection work and improve "
                    "robustness; supports a future staged MVP option."
                ),
                "Lesson for PyroFinder": (
                    "Data imbalance and smoke detection quality strongly affect performance; "
                    "track class balance, smoke-specific metrics, and false alarms."
                ),
            },
            {
                "Paper": "Wicaksono et al. (2024)",
                "Model": "YOLOv8",
                "Key Result": "mAP 0.63, precision 0.70, recall 0.57",
                "Relation to PyroFinder": (
                    "Demonstrates a modern YOLO detector can identify fire and smoke from "
                    "ordinary images; supports PyroFinder's object-detection direction."
                ),
                "Lesson for PyroFinder": (
                    "Small dataset and no real-world testing limit reliability; validate beyond "
                    "the training dataset and report limitations clearly."
                ),
            },
            {
                "Paper": "Cheng et al. (2024)",
                "Model": "Survey of deep learning methods including YOLOv8 and improved variants",
                "Key Result": (
                    "YOLO-style detectors are fast; attention and multiscale fusion improve "
                    "accuracy and reduce false alarms"
                ),
                "Relation to PyroFinder": (
                    "Strongest theoretical support for PyroFinder's two-class object-detection "
                    "formulation: detect and localise fire and smoke instead of only classifying a frame."
                ),
                "Lesson for PyroFinder": (
                    "Use detection metrics, not only accuracy: mAP, precision, recall, "
                    "false alarm rate, and speed."
                ),
            },
            {
                "Paper": "Saleh et al. (2024)",
                "Model": "Various CNN and YOLO-based detectors",
                "Key Result": (
                    "Many studies report accuracy above 90%; YOLO-based methods are strong for "
                    "real-time surveillance"
                ),
                "Relation to PyroFinder": (
                    "Supports PyroFinder's move from passive camera viewing to automated "
                    "detection using deep learning and a Streamlit monitoring dashboard."
                ),
                "Lesson for PyroFinder": (
                    "Smoke can be small, distant, and visually similar to clouds/fog; include "
                    "background negatives, augmentation, and false-positive review."
                ),
            },
            {
                "Paper": "Das et al. (2026)",
                "Model": "YOLOv8 variants, hybrid CNN-Transformer models, lightweight detectors",
                "Key Result": (
                    "Highlights tradeoff between accuracy, latency, and energy; improved "
                    "YOLOv8 variants for small smoke and edge deployment"
                ),
                "Relation to PyroFinder": (
                    "Influences PyroFinder's evaluation plan by making inference speed and "
                    "deployability part of model selection, not only detection accuracy."
                ),
                "Lesson for PyroFinder": (
                    "Benchmark NN object detection model against YOLO11n and document whether the main model is "
                    "fast enough for near-real-time sampled-frame monitoring."
                ),
            },
        ])
        st.table(_lit_table.set_index("Paper"))

        st.divider()

        # ── Section 3: Paper Summaries ────────────────────────────────────────
        st.subheader("Paper Summaries")

        with st.expander("Bahhar et al. (2023) — Staged YOLO + Ensemble CNN"):
            st.markdown("#### Wildfire and Smoke Detection Using Staged YOLO Model and Ensemble CNN (2023)")
            st.markdown(
                "**Citation:** Bahhar, C., Ksibi, A., Ayadi, M., Jamjoom, M. M., Ullah, Z., "
                "Soufiene, B. O., & Sakli, H. (2023). *Wildfire and Smoke Detection Using Staged "
                "YOLO Model and Ensemble CNN*. Electronics, 12(1), 228."
            )
            st.markdown(
                "[https://doi.org/10.3390/electronics12010228](https://doi.org/10.3390/electronics12010228)"
            )
            st.markdown(
                "Bahhar et al. propose a two-stage pipeline that first classifies a frame with an "
                "ensemble CNN and then uses YOLO to localize fire or smoke, reducing unnecessary "
                "detection work and improving robustness in complex scenes. They report accuracy "
                "of 0.99 and F1 of 0.95 for classification, and mAP@0.5 of 0.85 for smoke "
                "detection. However, the authors note that data quality is a major limitation, "
                "especially the lack of real-world UAV fire imagery, and that models trained on "
                "limited datasets struggle to generalize to new camera views and conditions."
            )
            st.markdown(
                "**Lesson for PyroFinder:** Data imbalance and smoke detection quality strongly "
                "affect performance; PyroFinder must track class balance, smoke-specific metrics, "
                "and false alarms."
            )

        with st.expander("Wicaksono et al. (2024) — YOLOv8 for Wildfire Detection"):
            st.markdown("#### Deep Learning Wildfire Detection to Increase Fire Safety with YOLOv8 (2024)")
            st.markdown(
                "**Citation:** Wicaksono, P., Yunanda, R., Arisaputra, P., & Izdihar, Z. N. (2024). "
                "[*Deep Learning Wildfire Detection to Increase Fire Safety with YOLOv8*](https://www.ijisae.org/index.php/IJISAE/article/view/6190). "
                "International Journal of Intelligent Systems and Applications in Engineering, "
                "12(3), 4383–4387."
            )
            st.markdown(
                "Wicaksono et al. train YOLOv8 on 3,104 annotated fire and smoke images sourced "
                "from Roboflow Universe and the web, achieving mAP of 0.63, precision of 0.70, "
                "and recall of 0.57. The results demonstrate that a modern YOLO detector can "
                "produce usable real-time predictions from ordinary image data. However, the "
                "limited dataset size and absence of real-world camera testing constrain the "
                "reliability of these results for operational surveillance use."
            )
            st.markdown(
                "**Lesson for PyroFinder:** A small dataset and no real-world testing limit "
                "reliability; PyroFinder must validate beyond the training dataset and report "
                "mAP, precision, recall, and false-alarm behaviour clearly."
            )

        with st.expander("Cheng et al. (2024) — Deep Learning Fire Detection Survey"):
            st.markdown("#### Visual Fire Detection Using Deep Learning: A Survey (2024)")
            st.markdown(
                "**Citation:** Cheng, G., Chen, X., Wang, C., Li, X., Xian, B., & Yu, H. (2024). "
                "*Visual fire detection using deep learning: A survey*. Neurocomputing, 596, 127975."
            )
            st.markdown(
                "[https://doi.org/10.1016/j.neucom.2024.127975](https://doi.org/10.1016/j.neucom.2024.127975)"
            )
            st.markdown(
                "Cheng et al. survey deep learning methods for visual fire detection and argue that "
                "the field has moved from handcrafted pipelines toward models that support "
                "classification, localization, and segmentation simultaneously. The survey "
                "emphasizes that YOLO-style detectors provide a strong balance between detection "
                "speed and accuracy, making them well-suited for real-time monitoring. The authors "
                "also highlight attention modules and multiscale feature fusion as key architectural "
                "directions for reducing false alarms."
            )
            st.markdown(
                "**Lesson for PyroFinder:** Use detection metrics, not only accuracy — mAP, "
                "precision, recall, false alarm rate, and speed."
            )

        with st.expander("Saleh et al. (2024) — Forest Fire Surveillance Review"):
            st.markdown("#### Forest Fire Surveillance Systems: A Review of Deep Learning Methods (2024)")
            st.markdown(
                "**Citation:** Saleh, A., Zulkifley, M. A., Harun, H. H., Gaudreault, F., "
                "Davison, I., & Spraggon, M. (2024). *Forest fire surveillance systems: A review "
                "of deep learning methods*. Heliyon, 10(1), e23127."
            )
            st.markdown(
                "[https://doi.org/10.1016/j.heliyon.2023.e23127](https://doi.org/10.1016/j.heliyon.2023.e23127)"
            )
            st.markdown(
                "Saleh et al. review 37 deep learning papers on forest-fire surveillance from "
                "RGB, UAV, and CCTV sources, finding that many methods exceed 90% accuracy and "
                "that YOLO-based detectors are among the strongest for real-time use. However, "
                "the review highlights that smoke detection remains difficult because thin, "
                "distant smoke can be visually similar to clouds and background clutter. The "
                "authors conclude that reliable surveillance systems require small-object "
                "handling, data augmentation, and rigorous false-positive evaluation."
            )
            st.markdown(
                "**Lesson for PyroFinder:** Smoke can be small, distant, and visually similar "
                "to clouds/fog; PyroFinder needs background negatives, augmentation, and "
                "false-positive review in the dashboard."
            )

        with st.expander("Das et al. (2026) — Wildfire Detection Trends Survey"):
            st.markdown(
                "#### Emerging Trends in Wildfire Detection Through the Lens of Computer Vision "
                "and Wildfire Emission Quantification: A Comprehensive Survey (2026)"
            )
            st.markdown(
                "**Citation:** Das, K., Poovvancheri, J., Flesca, S., Roberta Calidonna, C., & "
                "Chen, D. (2026). *Emerging Trends in Wildfire Detection Through the Lens of "
                "Computer Vision and Wildfire Emission Quantification: A Comprehensive Survey*. "
                "IEEE Access, 14, 20201–20228."
            )
            st.markdown(
                "[https://doi.org/10.1109/ACCESS.2026.3660843](https://doi.org/10.1109/ACCESS.2026.3660843)"
            )
            st.markdown(
                "Das et al. survey YOLOv8-based wildfire detectors across UAV, satellite, and "
                "terrestrial RGB sources, focusing on architectural improvements such as attention "
                "modules, lightweight necks, and edge-oriented optimization. The survey emphasizes "
                "that real-time wildfire detection now requires balancing accuracy with inference "
                "latency and energy use, particularly for edge-device deployment. The authors "
                "identify small-smoke detection and lightweight model design as key open challenges "
                "for civilian surveillance systems."
            )
            st.markdown(
                "**Lesson for PyroFinder:** Benchmark NN object detection model against YOLO11n and document "
                "whether the main model is fast enough for near-real-time sampled-frame monitoring."
            )

        st.divider()

        # ── Section 4: Research Gap ───────────────────────────────────────────
        st.subheader("Research Gap")
        st.markdown(
            f"""<div style="border-left: 4px solid {PYRO_COLORS['primary']}; \
padding: 12px 20px; background-color: {PYRO_COLORS['card_bg']}; \
border-radius: 4px; margin-bottom: 12px;">
<strong>Existing models are benchmark-trained, not camera-ready.</strong><br/>
The literature shows strong detection on curated datasets but limited real-world testing. PyroFinder turns fire/smoke detection from a lab model into a practical alerting system using ordinary surveillance camera feeds.</div>""",
            unsafe_allow_html=True,
        )



"""M2 Dashboard — Dataset & EDA (story version) tab.

Moved verbatim from m2_dashboard.py during the Phase 3b split. Content unchanged.
Owns the METADATA_PATH / GENERATE_CMD constants (only this tab uses them).
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

METADATA_PATH = "data/dfire_metadata.csv"
GENERATE_CMD = (
    "python scripts/build_dfire_metadata.py "
    '--raw-root "<path-to-D-Fire-root>" '
    "--output data/dfire_metadata.csv"
)


def render():
        st.header("Dataset & EDA — D-Fire")

        with st.container():
            st.markdown(
                "**Name:** D-Fire Dataset &nbsp;·&nbsp; "
                "**License:** CC0 1.0 Universal &nbsp;·&nbsp; "
                "**Last Update:** 01/01/2026"
            )
            st.markdown(
                "**Link:** [https://github.com/gaia-solutions-on-demand/DFireDataset/tree/master]"
                "(https://github.com/gaia-solutions-on-demand/DFireDataset/tree/master)"
            )
            _ds_col1, _ds_col2 = st.columns(2)
            with _ds_col1:
                st.markdown("**Raw dataset**")
                st.markdown(
                    "- Total size: 2.90 GB\n"
                    "- 21,527 images with YOLO-format label files\n"
                    "- Object classes: `smoke` (class 0) and `fire` (class 1)\n"
                    "- Categories: fire-only, smoke-only, fire-and-smoke, background"
                )
            with _ds_col2:
                st.markdown("**Processed metadata**")
                st.markdown(
                    "- One row per image\n"
                    "- Image categories: `background`, `fire_only`, `smoke_only`, `fire_and_smoke`\n"
                    "- Target flags: `has_fire`, `has_smoke`\n"
                    "- Box counts and bbox coverage features\n"
                    "- Pixel-derived: brightness, dark pixel ratio, color spread\n"
                    "- Spatial: fire/smoke centres and thirds-grid locations"
                )

        st.caption(
            "Key EDA charts from the processed D-Fire metadata, "
            "with findings and implications for NN object detection model training and evaluation."
        )

        try:
            _m2_raw = load_dfire_metadata(METADATA_PATH)
            _m2_df = clean_dfire_metadata(_m2_raw)
        except FileNotFoundError:
            st.warning(
                "Metadata CSV not found. Generate it by running:\n\n"
                f"```\n{GENERATE_CMD}\n```"
            )
        else:
            # ── D-Fire metrics header ───────────────────────────────────────
            _m2_info = get_primary_dataset_info()
            _m2_metrics = compute_summary_metrics(_m2_df)
            _mc1, _mc2, _mc3, _mc4, _mc5, _mc6 = st.columns(6)
            _mc1.metric("Total images", f"{_m2_info['num_images']:,}")
            _mc2.metric(
                "Fire images",
                f"{_m2_info['breakdown']['fire_only'] + _m2_info['breakdown']['fire_and_smoke']:,}",
            )
            _mc3.metric(
                "Smoke images",
                f"{_m2_info['breakdown']['smoke_only'] + _m2_info['breakdown']['fire_and_smoke']:,}",
            )
            _mc4.metric("Background", f"{_m2_info['breakdown']['background']:,}")
            _mc5.metric("Mean boxes/img", f"{_m2_metrics['mean_boxes_per_image']:.2f}")
            _mc6.metric("Median boxes/img", f"{_m2_metrics['median_boxes_per_image']:.2f}")

            st.divider()

            # ── Row 1: stacked bbox histogram  |  dark pixel ratio ───────────
            _eda_r1c1, _eda_r1c2 = st.columns(2)

            with _eda_r1c1:
                st.subheader("Bounding boxes per image — by category")
                if "total_boxes" in _m2_df.columns and "image_category" in _m2_df.columns:
                    _m2_bbox_cat = (
                        _m2_df.groupby(["total_boxes", "image_category"])
                        .size()
                        .reset_index(name="count")
                    )
                    # Build one explicit go.Bar trace per category so barmode="stack"
                    # works unambiguously — all categories at the same integer x value
                    # are stacked into one bar.
                    _m2_box_x = sorted(_m2_bbox_cat["total_boxes"].unique())
                    _m2_fig_bbox_cat = go.Figure()
                    for _cat in ["background", "fire_only", "smoke_only", "fire_and_smoke"]:
                        _cat_data = _m2_bbox_cat[_m2_bbox_cat["image_category"] == _cat]
                        _cat_map = dict(zip(_cat_data["total_boxes"], _cat_data["count"]))
                        _m2_fig_bbox_cat.add_trace(go.Bar(
                            x=_m2_box_x,
                            y=[_cat_map.get(v, 0) for v in _m2_box_x],
                            name=_cat,
                            marker_color=CAT_COLORS.get(_cat),
                        ))
                    _m2_fig_bbox_cat.update_layout(
                        barmode="stack",
                        bargap=0.25,
                        bargroupgap=0.1,
                        title="Bounding boxes per image — by category",
                        xaxis_title="Bounding boxes per image",
                        yaxis_title="Image count",
                        legend_title="Category",
                        height=420,
                    )
                    apply_chart_theme(_m2_fig_bbox_cat)
                    st.plotly_chart(_m2_fig_bbox_cat, use_container_width=True)
                    st.markdown("<div style='background:rgba(15,25,50,0.7);border-left:4px solid rgba(28,131,164,0.9);border-radius:4px;padding:10px 16px;margin:6px 0;'>Nearly half the dataset is background — fire+smoke scenes are rare but the most training-valuable.</div>", unsafe_allow_html=True)
                else:
                    st.warning("Required columns for stacked histogram not found in metadata.")

            with _eda_r1c2:
                st.subheader("Scene lighting — dark pixel ratio by category")
                if "dark_pixel_ratio" in _m2_df.columns:
                    _m2_dk_df = compute_pixel_stats_by_category(_m2_df, "dark_pixel_ratio")
                    _m2_fig_dk = px.box(
                        _m2_dk_df, x="image_category", y="dark_pixel_ratio",
                        color="image_category",
                        color_discrete_map=CAT_COLORS,
                        points="outliers",
                        labels={
                            "dark_pixel_ratio": "Dark pixel ratio (brightness < 0.118)",
                            "image_category": "",
                        },
                        title="Dark pixel ratio by category",
                    )
                    _m2_fig_dk.update_layout(showlegend=False, height=420)
                    apply_chart_theme(_m2_fig_dk)
                    st.plotly_chart(_m2_fig_dk, use_container_width=True)
                    st.markdown("<div style='background:rgba(15,25,50,0.7);border-left:4px solid rgba(28,131,164,0.9);border-radius:4px;padding:10px 16px;margin:6px 0;'>Lighting differs sharply by class — fire is mainly night, smoke is mainly daytime. The model must handle both to avoid missing one class under the wrong lighting.</div>", unsafe_allow_html=True)
                else:
                    st.info("Pixel stat columns not found. Re-run scripts/build_dfire_metadata.py.")

            st.divider()

            # ── Row 2: bbox area  |  spatial 2×2 ────────────────────────────
            _eda_r2c1, _eda_r2c2 = st.columns(2)

            with _eda_r2c1:
                st.subheader("Bounding box size — fire vs smoke")
                _m2_area_df = compute_class_bbox_areas(_m2_df)
                if not _m2_area_df.empty:
                    _m2_fig_area = px.box(
                        _m2_area_df, x="class", y="mean_bbox_area",
                        color="class",
                        color_discrete_map=CLASS_COLORS,
                        points="outliers",
                        labels={
                            "mean_bbox_area": "Mean normalised area (w×h)",
                            "class": "Class",
                        },
                        title="Per-image mean bbox area — fire vs smoke",
                    )
                    _m2_fig_area.update_layout(height=420)
                    apply_chart_theme(_m2_fig_area)
                    st.plotly_chart(_m2_fig_area, use_container_width=True)
                    st.markdown("<div style='background:rgba(15,25,50,0.7);border-left:4px solid rgba(28,131,164,0.9);border-radius:4px;padding:10px 16px;margin:6px 0;'>Smoke boxes are ~7× larger than fire boxes — the model must handle both large plumes and small flames in the same scene.</div>", unsafe_allow_html=True)
                else:
                    st.info("Per-class bbox area columns not found. Re-run scripts/build_dfire_metadata.py.")

            with _eda_r2c2:
                st.subheader("Spatial analysis — fire and smoke frame position")
                _m2_spatial_ready = all(
                    c in _m2_df.columns for c in [
                        "fire_mean_x_center", "smoke_mean_x_center",
                        "fire_thirds_col", "smoke_thirds_col",
                    ]
                )
                if not _m2_spatial_ready:
                    st.warning("Spatial columns not found. Re-run scripts/build_dfire_metadata.py.")
                else:
                    _m2_fire_centers = compute_spatial_centers(_m2_df, "fire")
                    _m2_smoke_centers = compute_spatial_centers(_m2_df, "smoke")
                    _m2_fire_grid = compute_grid_distribution(_m2_df, "fire")
                    _m2_smoke_grid = compute_grid_distribution(_m2_df, "smoke")

                    _m2_fig_spatial = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=[
                            "Fire — density",
                            "Smoke — density",
                            "Fire — thirds grid",
                            "Smoke — thirds grid",
                        ],
                    )

                    if not _m2_fire_centers.empty:
                        _m2_fig_spatial.add_trace(
                            go.Histogram2d(
                                x=_m2_fire_centers["x_center"],
                                y=_m2_fire_centers["y_center"],
                                nbinsx=20, nbinsy=20,
                                colorscale="YlOrRd",
                                showscale=False,
                                name="Fire density",
                            ),
                            row=1, col=1,
                        )

                    if not _m2_smoke_centers.empty:
                        _m2_fig_spatial.add_trace(
                            go.Histogram2d(
                                x=_m2_smoke_centers["x_center"],
                                y=_m2_smoke_centers["y_center"],
                                nbinsx=20, nbinsy=20,
                                colorscale="Blues",
                                showscale=False,
                                name="Smoke density",
                            ),
                            row=1, col=2,
                        )

                    if not _m2_fire_grid.empty:
                        _m2_fig_spatial.add_trace(
                            go.Heatmap(
                                z=_m2_fire_grid.values,
                                x=[str(c) for c in _m2_fire_grid.columns],
                                y=[str(r) for r in _m2_fire_grid.index],
                                colorscale="YlOrRd",
                                showscale=False,
                                text=_m2_fire_grid.values,
                                texttemplate="%{text}",
                                name="Fire grid",
                            ),
                            row=2, col=1,
                        )

                    if not _m2_smoke_grid.empty:
                        _m2_fig_spatial.add_trace(
                            go.Heatmap(
                                z=_m2_smoke_grid.values,
                                x=[str(c) for c in _m2_smoke_grid.columns],
                                y=[str(r) for r in _m2_smoke_grid.index],
                                colorscale="Blues",
                                showscale=False,
                                text=_m2_smoke_grid.values,
                                texttemplate="%{text}",
                                name="Smoke grid",
                            ),
                            row=2, col=2,
                        )

                    # Reverse y-axis on all 4 subplots so image-space coords are
                    # correct: y=0 at top (top of frame), y=1 at bottom.
                    for _r, _c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                        _m2_fig_spatial.update_yaxes(autorange="reversed", row=_r, col=_c)

                    _m2_fig_spatial.update_layout(height=420)
                    apply_chart_theme(_m2_fig_spatial)
                    st.plotly_chart(_m2_fig_spatial, use_container_width=True)
                    st.markdown("<div style='background:rgba(15,25,50,0.7);border-left:4px solid rgba(28,131,164,0.9);border-radius:4px;padding:10px 16px;margin:6px 0;'>Fire centres mid-frame; smoke appears higher — plumes rise above the source and dominate the upper half of the image.</div>", unsafe_allow_html=True)

            st.divider()

            # ── Row 3: Pearson correlation | category balance per split ──────
            _eda_r3c1, _eda_r3c2 = st.columns(2)

            with _eda_r3c1:
                st.subheader("Pearson correlation matrix")
                _M2_CORR_COLS = [
                    "total_boxes",
                    "fire_bbox_coverage",
                    "smoke_bbox_coverage",
                    "mean_brightness",
                    "dark_pixel_ratio",
                    "color_std_mean",
                    "fire_mean_x_center",
                    "fire_mean_y_center",
                    "smoke_mean_x_center",
                    "smoke_mean_y_center",
                    "smoke_dy_vs_fire",
                    "fire_smoke_mean_iou",
                ]
                _m2_corr_avail = [c for c in _M2_CORR_COLS if c in _m2_df.columns]
                if len(_m2_corr_avail) < 2:
                    st.warning(
                        "Insufficient columns for correlation matrix. "
                        "Re-run scripts/build_dfire_metadata.py."
                    )
                else:
                    _m2_corr_mat = _m2_df[_m2_corr_avail].dropna().corr(method="pearson")
                    _m2_fig_corr = px.imshow(
                        _m2_corr_mat,
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        text_auto=".2f",
                        title="Pearson correlation matrix",
                    )
                    _m2_fig_corr.update_layout(height=420)
                    apply_chart_theme(_m2_fig_corr)
                    st.plotly_chart(_m2_fig_corr, use_container_width=True)
                    st.markdown("<div style='background:rgba(15,25,50,0.7);border-left:4px solid rgba(28,131,164,0.9);border-radius:4px;padding:10px 16px;margin:6px 0;'>Brightness features are redundant (r=0.90) — treat as one group. Fire and smoke share similar horizontal position (r≈0.55).</div>", unsafe_allow_html=True)

            with _eda_r3c2:
                st.subheader("Category balance per split")
                _m2_ct = compute_split_category_crosstab(_m2_df)
                if not _m2_ct.empty:
                    _m2_fig_ct = px.bar(
                        _m2_ct,
                        x="split",
                        y="count",
                        color="image_category",
                        barmode="stack",
                        color_discrete_map=CAT_COLORS,
                        category_orders={
                            "image_category": ["background", "fire_and_smoke", "fire_only", "smoke_only"]
                        },
                        labels={
                            "split": "Split",
                            "count": "Images",
                            "image_category": "Category",
                        },
                        title="Category distribution within each split",
                    )
                    _m2_fig_ct.update_layout(bargap=0.25, bargroupgap=0.1, height=420)
                    apply_chart_theme(_m2_fig_ct)
                    st.plotly_chart(_m2_fig_ct, use_container_width=True)
                    st.markdown("<div style='background:rgba(15,25,50,0.7);border-left:4px solid rgba(28,131,164,0.9);border-radius:4px;padding:10px 16px;margin:6px 0;'>The 80/20 split keeps category proportions consistent — test results should be representative of real distribution.</div>", unsafe_allow_html=True)
                else:
                    st.warning("Split/category balance data is not available.")

            st.divider()

            # ── Schema reference ──────────────────────────────────────────────
            with st.expander("Metadata schema summary", expanded=False):
                st.markdown(
                    "The processed metadata contains one row per image and combines "
                    "image-level labels, YOLO bounding-box statistics, pixel-derived features, "
                    "and spatial features."
                )
                st.table(pd.DataFrame([
                    {
                        "Group": "Identity",
                        "Example columns": "image_id, image_path, label_path",
                        "Meaning": "Unique image reference and file paths",
                    },
                    {
                        "Group": "Split",
                        "Example columns": "split",
                        "Meaning": "train or test partition",
                    },
                    {
                        "Group": "Target summary",
                        "Example columns": "image_category, has_fire, has_smoke",
                        "Meaning": "High-level label assigned to the image",
                    },
                    {
                        "Group": "Box counts",
                        "Example columns": "total_boxes, num_fire_boxes, num_smoke_boxes",
                        "Meaning": "Number of YOLO annotations per class",
                    },
                    {
                        "Group": "Box coverage",
                        "Example columns": "fire_bbox_coverage, smoke_bbox_coverage",
                        "Meaning": "Summed normalised area covered by each class",
                    },
                    {
                        "Group": "Image properties",
                        "Example columns": "image_width, image_height",
                        "Meaning": "Pixel dimensions of the source image",
                    },
                    {
                        "Group": "Pixel features",
                        "Example columns": "mean_brightness, dark_pixel_ratio, color_std_mean",
                        "Meaning": "Derived from 64×64 thumbnail; not raw sensor data",
                    },
                    {
                        "Group": "Spatial features",
                        "Example columns": "fire_mean_x_center, smoke_thirds_col",
                        "Meaning": "Where in the frame fire or smoke bboxes appear",
                    },
                    {
                        "Group": "Relative fire/smoke position",
                        "Example columns": "smoke_dx_vs_fire, smoke_dy_vs_fire",
                        "Meaning": "Smoke centre offset from fire centre (fire+smoke images only)",
                    },
                    {
                        "Group": "Correlation-ready numeric",
                        "Example columns": "fire_smoke_mean_iou, smoke_dy_vs_fire",
                        "Meaning": "Numeric features used in the Pearson correlation analysis",
                    },
                ]))

            # ── EDA Summary ───────────────────────────────────────────────────
            st.divider()
            st.subheader("EDA Summary")
            st.markdown(
                "<div style='background:rgba(28,131,164,0.1);border-left:4px solid rgba(28,131,164,0.7);"
                "border-radius:4px;padding:10px 16px;opacity:0.7;margin:6px 0;'>"
                "<strong>D-Fire is class-imbalanced</strong> (46% background) — recall must be the primary metric, not accuracy.<br/>"
                "Key training challenges: extreme lighting difference between fire and smoke scenes, "
                "7× bbox scale gap between classes, and fire concentrating mid-frame while smoke rises higher."
                "</div>",
                unsafe_allow_html=True,
            )


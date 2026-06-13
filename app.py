import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from src.data import get_primary_dataset_info, load_dfire_metadata, clean_dfire_metadata
from src.model import get_model_plan, get_metrics_plan
import pandas as pd

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
from src.ui import inject_pyrofinder_theme, apply_chart_theme, CAT_COLORS, PYRO_COLORS, SPLIT_COLORS, CLASS_COLORS


@st.cache_resource(show_spinner=False)
def load_detector_cached(model_name: str):
    """Load and cache a fine-tuned YOLO detector by name.

    Heavy ML libraries are imported lazily inside src.inference, so models are
    only loaded when inference is explicitly requested — never at import time.
    Cached per model name so each checkpoint loads once per session.
    """
    from src.inference import load_detector
    return load_detector(model_name)


st.set_page_config(page_title="PyroFinder", layout="wide")
inject_pyrofinder_theme(
    background_video_path=Path("design_images") / "Nordic_Forest_LowPolymp_.mp4",
    use_video_background=True,
)

METADATA_PATH = "data/dfire_metadata.csv"
SAMPLES_DIR = Path("data/samples/dfire/images")
GENERATE_CMD = (
    "python scripts/build_dfire_metadata.py "
    '--raw-root "<path-to-D-Fire-root>" '
    "--output data/dfire_metadata.csv"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("PyroFinder")
    st.caption("Fire detection and monitoring — existing cameras only")

    st.divider()

    mode = st.selectbox(
        "Dashboard mode",
        [
            "M2 Course Dashboard",
            "Operations & Learning Dashboard",
            "Central Control Dashboard",
        ],
    )

    st.divider()
    model_plan = get_model_plan()
    st.caption(f"Primary model: **{model_plan['primary_model']}**")
    st.caption(f"Baseline: {model_plan['baseline_model']}")
    st.caption(f"Classes: {', '.join(model_plan['classes'])}")

    if mode == "Operations & Learning Dashboard":
        st.divider()
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        confirmation_frames = st.number_input(
            "Confirmation frames (N)",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Alert triggered only after fire/smoke detected in N consecutive frames.",
        )

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("## 🔥 PyroFinder")
st.caption("Real-time fire and smoke detection using existing cameras.")

if mode == "Operations & Learning Dashboard":
    tab_overview, tab_eda, tab_models, tab_inference, tab_mapping, tab_alerts = st.tabs([
        "Overview",
        "Dataset & EDA",
        "Models",
        "Inference Demo",
        "Mapping Setup",
        "Alert Log",
    ])

    # ── Overview ────────────────────────────────────────────────────────────
    with tab_overview:
        st.header("System Overview")
        st.caption("System plan — model, metrics, and dataset summary")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model plan")
            _mp = get_model_plan()
            st.markdown(f"**Primary model:** {_mp['primary_model']} (`{_mp['primary_weights']}`)")
            st.caption(_mp["primary_reason"])
            st.markdown(f"**Baseline:** {_mp['baseline_model']} (`{_mp['baseline_weights']}`)")
            st.caption(_mp["baseline_reason"])
            st.markdown(f"**Task:** {_mp['task']}")
            st.markdown(f"**Classes:** {', '.join(_mp['classes'])}")
            st.markdown(
                f"**Framework:** {_mp['framework']} &nbsp;|&nbsp; "
                f"Image size: {_mp['image_size']} px &nbsp;|&nbsp; "
                f"Fine-tuned on: {_mp['fine_tuned_on']}"
            )

        with col2:
            st.subheader("Evaluation metrics")
            for m in get_metrics_plan():
                st.markdown(f"- {m}")

        st.divider()
        st.subheader("Primary dataset")
        _di = get_primary_dataset_info()
        _dc1, _dc2, _dc3, _dc4 = st.columns(4)
        _dc1.metric("Total images", f"{_di['num_images']:,}")
        _dc2.metric("Fire boxes", f"{_di['bounding_boxes']['fire']:,}")
        _dc3.metric("Smoke boxes", f"{_di['bounding_boxes']['smoke']:,}")
        _dc4.metric("License", _di["license"])

        _da, _db = st.columns(2)
        with _da:
            st.markdown(f"**Name:** {_di['name']}")
            st.markdown(f"**Role:** {_di['role']}")
            st.markdown(f"**Annotation format:** {_di['annotation_format']}")
            st.markdown(f"**Classes:** {', '.join(_di['classes'])}")
            st.markdown("**Image breakdown:**")
            st.dataframe(
                pd.DataFrame(
                    [(k.replace("_", " ").title(), f"{v:,}") for k, v in _di["breakdown"].items()],
                    columns=["Split", "Count"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        with _db:
            if _di.get("known_gaps"):
                st.markdown(
                    "**Known gaps:**\n" + "\n".join(f"- {_g}" for _g in _di["known_gaps"])
                )
            if _di.get("possible_biases"):
                st.markdown(
                    "**Possible biases:**\n" + "\n".join(f"- {_b}" for _b in _di["possible_biases"])
                )

    # ── Dataset & EDA ───────────────────────────────────────────────────────
    with tab_eda:
        st.header("Dataset & EDA — D-Fire")
        st.caption("D-Fire · 21,527 images · class 0 = smoke · class 1 = fire")

        # Load metadata
        try:
            raw_df = load_dfire_metadata(METADATA_PATH)
            df_all = clean_dfire_metadata(raw_df)
        except FileNotFoundError:
            st.warning(
                "Metadata CSV not found. Generate it by running:\n\n"
                f"```\n{GENERATE_CMD}\n```"
            )
            st.stop()

        # ── Sidebar EDA filters ─────────────────────────────────────────────
        with st.sidebar:
            st.divider()
            st.subheader("EDA Filters")

            all_categories = sorted(df_all["image_category"].unique().tolist())
            selected_categories = st.multiselect(
                "Image category",
                options=all_categories,
                default=all_categories,
            )

            all_splits = sorted(df_all["split"].unique().tolist())
            selected_splits = st.multiselect(
                "Split",
                options=all_splits,
                default=all_splits,
            )

            fire_filter = st.selectbox("Has fire", ["All", "Yes", "No"])
            smoke_filter = st.selectbox("Has smoke", ["All", "Yes", "No"])

        fire_bool = True if fire_filter == "Yes" else (False if fire_filter == "No" else None)
        smoke_bool = True if smoke_filter == "Yes" else (False if smoke_filter == "No" else None)

        df = filter_metadata(
            df_all,
            categories=selected_categories or None,
            splits=selected_splits or None,
            has_fire=fire_bool,
            has_smoke=smoke_bool,
        )

        # ── Metrics row ─────────────────────────────────────────────────────
        metrics = compute_summary_metrics(df)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total images", f"{metrics['total_images']:,}")
        c2.metric("Fire images", f"{metrics['fire_images']:,}")
        c3.metric("Smoke images", f"{metrics['smoke_images']:,}")
        c4.metric("Background", f"{metrics['background_images']:,}")
        c5.metric("Mean boxes/img", f"{metrics['mean_boxes_per_image']:.2f}")
        c6.metric("Median boxes/img", f"{metrics['median_boxes_per_image']:.1f}")

        st.divider()

        # ── Charts ──────────────────────────────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Image count by category")
            cat_df = compute_category_counts(df)
            fig_cat = px.bar(
                cat_df,
                x="category",
                y="count",
                color="category",
                color_discrete_map=CAT_COLORS,
                labels={"category": "Category", "count": "Images"},
                title="D-Fire: Image distribution by category",
            )
            fig_cat.update_layout(showlegend=False, bargap=0.25, bargroupgap=0.1)
            apply_chart_theme(fig_cat)
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_b:
            st.subheader("Bounding boxes per image")
            fig_hist = px.histogram(
                df[df["total_boxes"] > 0],
                x="total_boxes",
                nbins=30,
                color_discrete_sequence=[PYRO_COLORS["fire"]],
                labels={"total_boxes": "Total boxes", "count": "Images"},
                title="Distribution of bounding box count (images with detections only)",
            )
            apply_chart_theme(fig_hist)
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Row 2: split distribution + split × category balance ────────────
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Split distribution")
            split_df = compute_split_counts(df)
            fig_split = px.bar(
                split_df,
                x="split",
                y="count",
                color="split",
                color_discrete_map=SPLIT_COLORS,
                labels={"split": "Split", "count": "Images"},
                title="D-Fire: Images per split",
            )
            fig_split.update_layout(showlegend=False, bargap=0.25, bargroupgap=0.1)
            apply_chart_theme(fig_split)
            st.plotly_chart(fig_split, use_container_width=True)

        with col_d:
            st.subheader("Category balance per split")
            ct = compute_split_category_crosstab(df)
            if not ct.empty:
                fig_ct = px.bar(
                    ct,
                    x="split",
                    y="count",
                    color="image_category",
                    barmode="stack",
                    color_discrete_map=CAT_COLORS,
                    labels={"split": "Split", "count": "Images", "image_category": "Category"},
                    title="Category distribution within each split",
                )
                fig_ct.update_layout(bargap=0.25, bargroupgap=0.1)
                apply_chart_theme(fig_ct)
                st.plotly_chart(fig_ct, use_container_width=True)

        st.divider()

        # ── Row 3: resolution scatter + fire vs smoke box counts ─────────────
        col_e, col_f = st.columns(2)

        with col_e:
            st.subheader("Image resolution by category")
            fig_res = px.scatter(
                df.sample(min(len(df), 5000), random_state=42) if len(df) > 5000 else df,
                x="image_width",
                y="image_height",
                color="image_category",
                color_discrete_map=CAT_COLORS,
                opacity=0.35,
                size_max=4,
                labels={"image_width": "Width (px)", "image_height": "Height (px)"},
                title="Width × Height (up to 5 000 points shown)",
            )
            fig_res.update_traces(marker={"size": 3})
            apply_chart_theme(fig_res)
            st.plotly_chart(fig_res, use_container_width=True)

        with col_f:
            st.subheader("Fire vs smoke box counts")
            boxes_long = pd.melt(
                df[df["total_boxes"] > 0][["num_fire_boxes", "num_smoke_boxes"]],
                var_name="class",
                value_name="boxes",
            )
            boxes_long["class"] = boxes_long["class"].map(
                {"num_fire_boxes": "fire", "num_smoke_boxes": "smoke"}
            )
            boxes_long = boxes_long[boxes_long["boxes"] > 0]
            fig_box_cnt = px.histogram(
                boxes_long,
                x="boxes",
                color="class",
                barmode="overlay",
                nbins=20,
                opacity=0.7,
                color_discrete_map=CLASS_COLORS,
                labels={"boxes": "Boxes per image", "count": "Images"},
                title="Box count distribution — fire vs smoke",
            )
            apply_chart_theme(fig_box_cnt)
            st.plotly_chart(fig_box_cnt, use_container_width=True)

        st.divider()

        # ── Coverage & class bbox area ───────────────────────────────────────
        with st.expander("Coverage analysis — fire vs smoke bbox area"):
            col_g, col_h = st.columns(2)

            with col_g:
                st.subheader("Mean bbox area by class")
                area_df = compute_class_bbox_areas(df)
                if not area_df.empty:
                    fig_area = px.violin(
                        area_df,
                        x="class",
                        y="mean_bbox_area",
                        color="class",
                        box=True,
                        color_discrete_map=CLASS_COLORS,
                        labels={"mean_bbox_area": "Mean normalized area (w×h)", "class": "Class"},
                        title="Per-image mean bbox area — fire vs smoke",
                    )
                    apply_chart_theme(fig_area)
                    st.plotly_chart(fig_area, use_container_width=True)
                else:
                    st.info("Per-class bbox area columns not found. Re-run the metadata script.")

            with col_h:
                st.subheader("Total bbox coverage fraction")
                if "fire_bbox_coverage" in df.columns and "smoke_bbox_coverage" in df.columns:
                    cov_long = pd.melt(
                        df[df["total_boxes"] > 0][
                            ["image_category", "fire_bbox_coverage", "smoke_bbox_coverage"]
                        ],
                        id_vars="image_category",
                        var_name="class",
                        value_name="coverage",
                    )
                    cov_long["class"] = cov_long["class"].str.replace("_bbox_coverage", "")
                    cov_long = cov_long[cov_long["coverage"] > 0]
                    fig_cov = px.box(
                        cov_long,
                        x="class",
                        y="coverage",
                        color="class",
                        color_discrete_map=CLASS_COLORS,
                        labels={"coverage": "Summed normalized area", "class": "Class"},
                        title="Total class coverage per image (labeled images only)",
                    )
                    apply_chart_theme(fig_cov)
                    st.plotly_chart(fig_cov, use_container_width=True)
                else:
                    st.info("Coverage columns not found. Re-run the metadata script.")

        # ── Pixel statistics (derived features) ─────────────────────────────
        _pixel_cols_present = all(
            c in df.columns for c in ["mean_brightness", "dark_pixel_ratio", "color_std_mean"]
        )
        with st.expander("Image pixel statistics (derived — not raw sensor data)"):
            st.caption(
                "Computed from stored JPEG/PNG files via a 64×64 thumbnail using ITU-R BT.601 "
                "luma weighting. JPEG compression, gamma correction, and any pre-processing by "
                "the original dataset creators affect these values. Do not treat them as raw "
                "sensor measurements."
            )
            if not _pixel_cols_present:
                st.info("Pixel stat columns not found. Re-run the metadata script.")
            else:
                col_i, col_j, col_k = st.columns(3)

                with col_i:
                    br_df = compute_pixel_stats_by_category(df, "mean_brightness")
                    fig_br = px.violin(
                        br_df,
                        x="image_category",
                        y="mean_brightness",
                        color="image_category",
                        box=True,
                        color_discrete_map=CAT_COLORS,
                        labels={"mean_brightness": "Mean brightness (0–1)", "image_category": ""},
                        title="Mean brightness by category",
                    )
                    fig_br.update_layout(showlegend=False)
                    apply_chart_theme(fig_br)
                    st.plotly_chart(fig_br, use_container_width=True)

                with col_j:
                    dk_df = compute_pixel_stats_by_category(df, "dark_pixel_ratio")
                    fig_dk = px.violin(
                        dk_df,
                        x="image_category",
                        y="dark_pixel_ratio",
                        color="image_category",
                        box=True,
                        color_discrete_map=CAT_COLORS,
                        labels={"dark_pixel_ratio": "Dark pixel ratio (brightness < 0.118)", "image_category": ""},
                        title="Dark pixel ratio by category",
                    )
                    fig_dk.update_layout(showlegend=False)
                    apply_chart_theme(fig_dk)
                    st.plotly_chart(fig_dk, use_container_width=True)

                with col_k:
                    cs_df = compute_pixel_stats_by_category(df, "color_std_mean")
                    fig_cs = px.violin(
                        cs_df,
                        x="image_category",
                        y="color_std_mean",
                        color="image_category",
                        box=True,
                        color_discrete_map=CAT_COLORS,
                        labels={"color_std_mean": "Color spread (mean channel std)", "image_category": ""},
                        title="Color spread (diversity) by category",
                    )
                    fig_cs.update_layout(showlegend=False)
                    apply_chart_theme(fig_cs)
                    st.plotly_chart(fig_cs, use_container_width=True)

        st.divider()

        # ── Spatial analysis ─────────────────────────────────────────────────
        _spatial_ready = all(
            c in df.columns for c in [
                "fire_mean_x_center", "smoke_mean_x_center",
                "fire_thirds_col", "smoke_thirds_col",
            ]
        )
        with st.expander("Spatial analysis — where in the frame do fire and smoke appear?"):
            if not _spatial_ready:
                st.info("Spatial columns not found. Re-run the metadata script.")
            else:
                col_sp1, col_sp2 = st.columns(2)
                with col_sp1:
                    fire_centers = compute_spatial_centers(df, "fire")
                    if not fire_centers.empty:
                        fig_fire_sp = px.density_heatmap(
                            fire_centers, x="x_center", y="y_center",
                            nbinsx=20, nbinsy=20,
                            color_continuous_scale="YlOrRd",
                            title="Fire bbox centres — image frame density",
                            labels={"x_center": "X (left→right)", "y_center": "Y (top→bottom)"},
                        )
                        fig_fire_sp.update_yaxes(autorange="reversed")
                        apply_chart_theme(fig_fire_sp)
                        st.plotly_chart(fig_fire_sp, use_container_width=True)

                with col_sp2:
                    smoke_centers = compute_spatial_centers(df, "smoke")
                    if not smoke_centers.empty:
                        fig_smoke_sp = px.density_heatmap(
                            smoke_centers, x="x_center", y="y_center",
                            nbinsx=20, nbinsy=20,
                            color_continuous_scale="Blues",
                            title="Smoke bbox centres — image frame density",
                            labels={"x_center": "X (left→right)", "y_center": "Y (top→bottom)"},
                        )
                        fig_smoke_sp.update_yaxes(autorange="reversed")
                        apply_chart_theme(fig_smoke_sp)
                        st.plotly_chart(fig_smoke_sp, use_container_width=True)

                col_sp3, col_sp4 = st.columns(2)
                with col_sp3:
                    fire_grid = compute_grid_distribution(df, "fire")
                    if not fire_grid.empty:
                        fig_fg = px.imshow(
                            fire_grid,
                            text_auto=True,
                            color_continuous_scale="YlOrRd",
                            title="Fire — thirds grid (row=top/mid/bot, col=left/ctr/right)",
                            labels={"x": "Column third", "y": "Row third"},
                        )
                        apply_chart_theme(fig_fg)
                        st.plotly_chart(fig_fg, use_container_width=True)

                with col_sp4:
                    smoke_grid = compute_grid_distribution(df, "smoke")
                    if not smoke_grid.empty:
                        fig_sg = px.imshow(
                            smoke_grid,
                            text_auto=True,
                            color_continuous_scale="Blues",
                            title="Smoke — thirds grid (row=top/mid/bot, col=left/ctr/right)",
                            labels={"x": "Column third", "y": "Row third"},
                        )
                        apply_chart_theme(fig_sg)
                        st.plotly_chart(fig_sg, use_container_width=True)

                # Relative smoke vs fire position (fire_and_smoke images only)
                if "smoke_dy_vs_fire" in df.columns:
                    fs_df = df[df["image_category"] == "fire_and_smoke"].dropna(
                        subset=["smoke_dx_vs_fire", "smoke_dy_vs_fire"]
                    )
                    if not fs_df.empty:
                        st.subheader("Smoke position relative to fire centre (fire+smoke images only)")
                        fig_rel = px.scatter(
                            fs_df,
                            x="smoke_dx_vs_fire",
                            y="smoke_dy_vs_fire",
                            color="fire_smoke_mean_iou",
                            color_continuous_scale="RdYlGn",
                            opacity=0.5,
                            labels={
                                "smoke_dx_vs_fire": "Smoke X − Fire X  (right = smoke is right of fire)",
                                "smoke_dy_vs_fire": "Smoke Y − Fire Y  (negative = smoke ABOVE fire)",
                                "fire_smoke_mean_iou": "Mean IoU",
                            },
                            title="Smoke offset from fire centre — coloured by IoU",
                        )
                        fig_rel.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.4)
                        fig_rel.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.4)
                        apply_chart_theme(fig_rel)
                        st.plotly_chart(fig_rel, use_container_width=True)
                        pct_above = (fs_df["smoke_dy_vs_fire"] < 0).mean()
                        st.caption(
                            f"Smoke is above the fire centre in {pct_above:.0%} of fire+smoke images "
                            f"(negative Smoke Y − Fire Y, since y=0 is the top of the image)."
                        )

        # ── Correlation explorer ─────────────────────────────────────────────
        with st.expander("Correlation explorer"):
            num_cols = get_numeric_cols(df)
            selected_corr = st.multiselect(
                "Columns to include in correlation matrix",
                options=num_cols,
                default=[c for c in [
                    "total_boxes", "fire_bbox_coverage", "smoke_bbox_coverage",
                    "mean_brightness", "dark_pixel_ratio", "color_std_mean",
                    "fire_mean_x_center", "fire_mean_y_center",
                    "smoke_mean_x_center", "smoke_mean_y_center",
                    "smoke_dy_vs_fire", "fire_smoke_mean_iou",
                ] if c in num_cols],
            )
            if len(selected_corr) >= 2:
                corr_mat = compute_correlation_matrix(df, selected_corr)
                if not corr_mat.empty:
                    fig_corr = px.imshow(
                        corr_mat,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        title="Pearson correlation matrix",
                    )
                    fig_corr.update_layout(height=500)
                    apply_chart_theme(fig_corr)
                    st.plotly_chart(fig_corr, use_container_width=True)

            st.divider()
            st.subheader("Scatter explorer")
            c_x, c_y = st.columns(2)
            with c_x:
                x_col = st.selectbox("X axis", num_cols, index=num_cols.index("mean_brightness") if "mean_brightness" in num_cols else 0, key="corr_x")
            with c_y:
                y_col = st.selectbox("Y axis", num_cols, index=num_cols.index("dark_pixel_ratio") if "dark_pixel_ratio" in num_cols else 1, key="corr_y")

            scatter_df = df[[x_col, y_col, "image_category"]].dropna()
            if len(scatter_df) > 5000:
                scatter_df = scatter_df.sample(5000, random_state=42)
            if not scatter_df.empty:
                fig_scatter = px.scatter(
                    scatter_df, x=x_col, y=y_col,
                    color="image_category",
                    color_discrete_map=CAT_COLORS,
                    opacity=0.4,
                    title=f"{x_col} vs {y_col} (up to 5 000 points)",
                )
                fig_scatter.update_traces(marker={"size": 4})
                apply_chart_theme(fig_scatter)
                st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()

        # ── EDA insight ─────────────────────────────────────────────────────
        st.subheader("EDA Insight")
        st.info(get_primary_eda_insight(df_all))

        st.divider()

        # ── Table preview (stratified sample) ───────────────────────────────
        st.subheader("Data preview")
        preview_cols = [
            "image_id", "split", "image_category",
            "has_fire", "has_smoke", "num_fire_boxes", "num_smoke_boxes",
            "total_boxes", "fire_bbox_coverage", "smoke_bbox_coverage",
            "mean_brightness", "dark_pixel_ratio",
            "fire_mean_x_center", "fire_mean_y_center",
            "image_width", "image_height",
        ]
        preview_cols = [c for c in preview_cols if c in df.columns]

        # Stratified sample: up to 5 rows from each category for representative diversity
        _strat_parts = [
            grp.sample(min(len(grp), 5), random_state=42)
            for _, grp in df.groupby("image_category")
        ]
        preview_df = (
            pd.concat(_strat_parts).sample(frac=1, random_state=42).reset_index(drop=True)
            if _strat_parts else df.head(20)
        )
        st.dataframe(preview_df[preview_cols], use_container_width=True)
        st.caption(
            f"Showing {len(preview_df)} stratified rows (up to 5 per category) "
            f"from {len(df):,} filtered images."
        )

        # Download button
        _csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download filtered data ({len(df):,} rows, {len(df.columns)} columns)",
            data=_csv_bytes,
            file_name=f"dfire_filtered_{len(df)}_images.csv",
            mime="text/csv",
        )

        st.divider()

        # ── Sample images — matching the preview table rows ───────────────────
        with st.expander("Sample images (matching table rows above)"):
            _showable = preview_df[preview_df["image_path"].apply(
                lambda p: Path(str(p)).exists()
            )] if "image_path" in preview_df.columns else pd.DataFrame()

            if not _showable.empty:
                img_cols = st.columns(4)
                for i, (_, row_data) in enumerate(_showable.iterrows()):
                    if i >= 20:
                        break
                    lp = row_data.get("label_path", "")
                    try:
                        png = draw_yolo_boxes(row_data["image_path"], lp if lp else None)
                        img_cols[i % 4].image(
                            png,
                            caption=f"{row_data['image_id']} [{row_data['image_category']}]",
                            use_container_width=True,
                        )
                    except Exception:
                        pass
            else:
                # Fall back to committed sample images when local raw paths are unavailable
                _sample_imgs = sorted(SAMPLES_DIR.glob("*.jpg")) + sorted(SAMPLES_DIR.glob("*.png"))
                if _sample_imgs:
                    _sample_labels = SAMPLES_DIR.parent / "labels"
                    st.caption(
                        f"Showing {min(len(_sample_imgs), 20)} committed sample images "
                        "(local raw D-Fire paths not available on this machine)"
                    )
                    img_cols = st.columns(4)
                    for i, img_p in enumerate(_sample_imgs[:20]):
                        lp = _sample_labels / (img_p.stem + ".txt")
                        try:
                            png = draw_yolo_boxes(img_p, lp if lp.exists() else None)
                            img_cols[i % 4].image(
                                png,
                                caption=img_p.stem,
                                use_container_width=True,
                            )
                        except Exception:
                            pass
                else:
                    st.info("Raw image paths from the CSV are not accessible on this machine.")

    # ── Models ───────────────────────────────────────────────────────────────
    with tab_models:
        import json as _json

        st.header("Models")
        st.caption(
            "Sklearn image-level classifiers (color features), the YOLO11n object-detection "
            "baseline / fallback, and the YOLO11s current primary detector · "
            "D-Fire dataset · fire / smoke / background"
        )

        _results_dir = Path("results")
        _result_files = sorted(_results_dir.glob("*.json"))

        if not _result_files:
            st.warning(
                "No model result files found in `results/`. "
                "Run `python scripts/dummy_try.py` to generate baseline results."
            )
        else:
            # ── Load all result files ────────────────────────────────────────
            _results_data = {}
            for _rf in _result_files:
                try:
                    _d = _json.loads(_rf.read_text(encoding="utf-8"))
                    # Skip operational-metrics JSONs (e.g. yolo11n_operational_metrics.json).
                    # These are loaded only inside _render_operational_alert_metrics(),
                    # not as standalone model baselines — otherwise they would show up as a
                    # second object-detection row with missing mAP/precision/recall.
                    if _d.get("evaluation_type") == "operational_alert_metrics":
                        continue
                    _results_data[_d.get("model_name", _rf.stem)] = _d
                except Exception:
                    pass

            # ── Helper: short display label ──────────────────────────────────
            def _short_model_label(name):
                if "Dummy" in name:
                    return "Dummy"
                if "Logistic" in name:
                    return "Logistic Regression"
                if "Random Forest" in name:
                    return "Random Forest"
                if "yolo11n" in name.lower():
                    return "YOLO11n"
                if "yolo11s" in name.lower():
                    return "YOLO11s"
                return name

            # ── Helper: sort order ───────────────────────────────────────────
            def _model_sort_key(name):
                if "Dummy" in name:
                    return (0, name)
                if "Logistic" in name:
                    return (1, name)
                if "Random Forest" in name:
                    return (2, name)
                if "yolo11n" in name.lower():
                    return (3, name)
                if "yolo11s" in name.lower():
                    return (4, name)
                return (5, name)

            # ── Helper: result type detection ─────────────────────────────────
            def _is_object_detection_result(result_dict):
                return result_dict.get("model_family") == "object_detection"

            def _is_sklearn_result(result_dict):
                return not _is_object_detection_result(result_dict)

            # ── Helper: per-model summary info text ──────────────────────────
            def _model_summary_text(name):
                if "Dummy" in name:
                    return (
                        "The dummy baseline achieved 47% accuracy by always predicting background, "
                        "but it completely failed to detect fire and smoke. "
                        "This proves that accuracy alone is not enough for PyroFinder. "
                        "Any real model must improve Macro F1 and, most importantly, "
                        "achieve meaningful recall for fire and smoke."
                    )
                if "Logistic" in name:
                    return (
                        "Logistic Regression is the first real learning baseline. It uses the 60 handcrafted "
                        "color features and improves Macro F1 to about 0.62, with real recall for both fire "
                        "and smoke. This proves that simple color information contains useful signal, but the "
                        "model still creates many false alarms by confusing background images with fire or smoke."
                    )
                if "Random Forest" in name:
                    return (
                        "Random Forest is the strongest classical ML baseline so far. It improves Macro F1 to "
                        "about 0.85 and gives balanced recall for background, fire, and smoke. This suggests "
                        "that the relationship between color features and fire/smoke labels is non-linear, and "
                        "that tree-based models capture these patterns much better than a linear classifier."
                    )
                return (
                    "This baseline is an image-level sklearn classifier using handcrafted color features. "
                    "It should be compared using Macro F1 and fire/smoke recall, not accuracy alone."
                )

            # ── Helper: per-model detailed analysis text ─────────────────────
            def _model_detailed_analysis(name):
                if "Dummy" in name:
                    return """
This result is not a real fire-detection model. It is a DummyClassifier with the most_frequent strategy, meaning it always predicts the most common class: background.

---

#### What the result means

The model reaches about 47% accuracy only because background is the largest class in the test set. It gets background recall of 1.00, but fire recall and smoke recall are both 0.00. This means it misses every real fire and every smoke case.

---

#### What it tells us about the data

The dataset is somewhat imbalanced, but not broken. Background is the largest class, while fire and smoke are still well represented. The result mainly proves that accuracy alone is misleading.

---

#### What it tells us about the model

The model does not use visual meaning and does not learn fire or smoke patterns. It only gives the minimum bar that every real model must beat.

---

#### Main conclusion

The DummyClassifier is useful only as a minimum comparison point: PyroFinder must beat Macro F1 = 0.21 and must achieve recall above 0 for both fire and smoke.
"""
                if "Logistic" in name:
                    return """
Logistic Regression is a simple linear learning baseline. Unlike the dummy model, it actually uses the 60 color features extracted from each image.

---

#### What the result means

The model improves strongly over the dummy baseline. It reaches about 61% accuracy and Macro F1 around 0.62. Most importantly, it detects both danger classes, with meaningful recall for fire and smoke.

---

#### What it tells us about the data

The color features contain useful signal. Fire and smoke images are not random in feature space: simple RGB/HSV statistics and histograms already help separate them from background.

---

#### What it tells us about the model

The model is still limited because it is linear. It catches many fire and smoke cases, but it also misclassifies many background images as fire or smoke. This means it is too aggressive and would create too many false alarms in an operational system.

---

#### Main conclusion

Logistic Regression proves that handcrafted color features are useful, but it is not strong enough as an operational baseline because background handling and false-alarm behavior are weak.
"""
                if "Random Forest" in name:
                    return """
Random Forest is a non-linear classical ML baseline using the same 60 handcrafted color features as Logistic Regression.

---

#### What the result means

Random Forest performs much better than the other sklearn baselines. It reaches about 86% accuracy and Macro F1 around 0.85, with strong recall for background, fire, and smoke.

---

#### What it tells us about the data

The dataset contains strong visual color patterns that can separate the three image-level classes. However, because this is based on simple color features, the model may also learn dataset-specific patterns such as lighting, background style, or scene color distribution.

---

#### What it tells us about the model

The strong improvement over Logistic Regression suggests that the relationship between features and labels is non-linear. Random Forest captures these interactions better and reduces false alarms much more effectively.

---

#### Main conclusion

Random Forest is the strongest classical ML baseline. It should be used as the main simple sklearn baseline, but it is still only an image-level classifier. It cannot replace YOLO11s, because PyroFinder needs object detection, bounding boxes, and approximate location support.
"""
                return """
This is an image-level sklearn classifier using handcrafted color features.

---

#### What the result means

Evaluate using Macro F1 and fire/smoke recall rather than accuracy alone, since background class imbalance makes accuracy misleading.

---

#### What it tells us about the data

Performance relative to other baselines indicates how much useful signal the 60 color features contain for this class.

---

#### What it tells us about the model

Compare against DummyClassifier (Macro F1 = 0.21) as the minimum bar and against Logistic Regression and Random Forest to see if it adds value.

---

#### Main conclusion

Any sklearn baseline is an image-level classifier. It cannot replace YOLO11s object detection, which is needed for bounding boxes, confidence scores, and approximate location support.
"""

            # ── Helper: render one model tab ─────────────────────────────────
            def _render_single_baseline_model(model_name, result_dict):
                _r = result_dict
                _metrics = _r.get("metrics", {})
                _clf_report = _metrics.get("classification_report", {})
                _dataset = _r.get("dataset", {})
                _features = _r.get("features", {})
                _classes_ordered = ["background", "fire", "smoke"]
                _classes = [c for c in _classes_ordered if c in _clf_report]
                _slug = _short_model_label(model_name).lower().replace(" ", "_")

                # Key metric cards
                _km1, _km2, _km3, _km4, _km5 = st.columns(5)
                _km1.metric("Accuracy",    f"{_metrics.get('accuracy', 0):.2f}")
                _km2.metric("F1 macro",    f"{_metrics.get('macro_avg', {}).get('f1', 0):.2f}")
                _km3.metric("F1 weighted", f"{_metrics.get('weighted_avg', {}).get('f1', 0):.2f}")
                _km4.metric("Fire recall",  f"{_clf_report.get('fire', {}).get('recall', 0):.2f}")
                _km5.metric("Smoke recall", f"{_clf_report.get('smoke', {}).get('recall', 0):.2f}")

                st.info(_model_summary_text(model_name))
                st.caption(
                    f"Run date: {_r.get('run_date', '—')}  ·  "
                    f"Dataset: {_dataset.get('name', '—')}  ·  "
                    f"Train: {_dataset.get('train_size', '—'):,}  ·  "
                    f"Test: {_dataset.get('test_size', '—'):,}"
                )

                st.divider()

                # Row 1: Per-class bar chart | Dataset distribution
                _row1_l, _row1_r = st.columns(2)

                with _row1_l:
                    st.subheader("Precision / Recall / F1 per class")
                    _prf_rows = []
                    for _cls in _classes:
                        for _mn, _ml in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]:
                            _prf_rows.append({
                                "class": _cls,
                                "metric": _ml,
                                "value": _clf_report[_cls].get(_mn, 0),
                            })
                    if _prf_rows:
                        _fig_prf = px.bar(
                            pd.DataFrame(_prf_rows),
                            x="class", y="value", color="metric",
                            barmode="group",
                            color_discrete_map={
                                "Precision": "#4fc3f7",
                                "Recall":    "#e07b39",
                                "F1":        "#81c784",
                            },
                            labels={"value": "Score (0–1)", "class": "Class", "metric": ""},
                            title=f"Per-class metrics — {_short_model_label(model_name)}",
                        )
                        _fig_prf.update_layout(yaxis_range=[0, 1], bargap=0.2, height=360)
                        apply_chart_theme(_fig_prf)
                        st.plotly_chart(_fig_prf, use_container_width=True, key=f"baseline_prf_{_slug}")

                with _row1_r:
                    st.subheader("Class distribution — train vs test")
                    _dist = _dataset.get("class_distribution", {})
                    _dist_rows = []
                    for _split_name, _counts in _dist.items():
                        for _cls, _n in _counts.items():
                            _dist_rows.append({"split": _split_name, "class": _cls, "count": _n})
                    if _dist_rows:
                        _fig_dist = px.bar(
                            pd.DataFrame(_dist_rows),
                            x="class", y="count", color="split",
                            barmode="group",
                            color_discrete_map=SPLIT_COLORS,
                            labels={"count": "Images", "class": "Class", "split": "Split"},
                            title="Images per class — train vs test",
                        )
                        _fig_dist.update_layout(bargap=0.2, height=360)
                        apply_chart_theme(_fig_dist)
                        st.plotly_chart(_fig_dist, use_container_width=True, key=f"baseline_dist_{_slug}")

                st.divider()

                # Row 2: Radar chart | Full metrics table
                _row2_l, _row2_r = st.columns(2)

                with _row2_l:
                    st.subheader("Macro average radar")
                    _macro = _metrics.get("macro_avg", {})
                    _radar_cats = ["Precision", "Recall", "F1", "Accuracy"]
                    _radar_vals = [
                        _macro.get("precision", 0),
                        _macro.get("recall",    0),
                        _macro.get("f1",        0),
                        _metrics.get("accuracy", 0),
                    ]
                    _fig_radar = go.Figure(go.Scatterpolar(
                        r=_radar_vals + [_radar_vals[0]],
                        theta=_radar_cats + [_radar_cats[0]],
                        fill="toself",
                        fillcolor="rgba(224,123,57,0.18)",
                        line=dict(color=PYRO_COLORS["primary"], width=2),
                        name=_short_model_label(model_name),
                    ))
                    _fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(range=[0, 1], tickfont=dict(size=10)),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cccccc"),
                        height=340,
                        margin=dict(l=50, r=50, t=40, b=40),
                    )
                    st.plotly_chart(_fig_radar, use_container_width=True, key=f"baseline_radar_{_slug}")

                with _row2_r:
                    st.subheader("Full metrics table")
                    _tbl_rows = []
                    for _cls in _classes:
                        _row_data = _clf_report[_cls]
                        _tbl_rows.append({
                            "Class":     _cls,
                            "Precision": round(_row_data.get("precision", 0), 2),
                            "Recall":    round(_row_data.get("recall", 0), 2),
                            "F1":        round(_row_data.get("f1", 0), 2),
                            "Support":   int(_row_data.get("support", 0)),
                        })
                    for _avg_key, _avg_label in [("macro_avg", "macro avg"), ("weighted_avg", "weighted avg")]:
                        _avg_data = _metrics.get(_avg_key, {})
                        if _avg_data:
                            _tbl_rows.append({
                                "Class":     _avg_label,
                                "Precision": round(_avg_data.get("precision", 0), 2),
                                "Recall":    round(_avg_data.get("recall", 0), 2),
                                "F1":        round(_avg_data.get("f1", 0), 2),
                                "Support":   "—",
                            })
                    st.dataframe(pd.DataFrame(_tbl_rows), use_container_width=True, hide_index=True)
                    st.caption(
                        f"Accuracy: **{_metrics.get('accuracy', 0):.2f}**  ·  "
                        f"Macro F1: **{_metrics.get('macro_avg', {}).get('f1', 0):.2f}**"
                    )

                st.divider()

                # Feature details
                with st.expander("Feature extraction details"):
                    _fc1, _fc2 = st.columns(2)
                    with _fc1:
                        st.markdown(f"**Description:** {_features.get('description', '—')}")
                        st.markdown(f"**Vector length:** {_features.get('vector_length', '—')}")
                        st.markdown(f"**Image resize:** {_features.get('image_resize', '—')}")
                        st.markdown(f"**Normalisation:** {_features.get('normalization', '—')}")
                    with _fc2:
                        _comps = _features.get("components", [])
                        if _comps:
                            st.dataframe(pd.DataFrame(_comps), use_container_width=True, hide_index=True)

                # Detailed analysis
                with st.expander("Detailed analysis — what this baseline tells us", expanded=False):
                    st.markdown(_model_detailed_analysis(model_name))

            # ── Helper: render a YOLO object-detection model tab ─────────────
            # One generic renderer shared by YOLO11n and YOLO11s so their tabs
            # stay identical. It shows object-detection metrics and training
            # curves only — never sklearn classification metrics. Real values are
            # read from ``result_dict``; when it is missing/empty a clear pending
            # state is shown and no metric values are invented.
            def _render_yolo_detection_model(
                model_key, display_name, role_label, result_dict,
                training_csv_path, fallback_csv_path,
            ):
                _r = result_dict or {}
                _m = _r.get("metrics", {})
                _slug = model_key.lower()
                _ran = _r.get("run_date") is not None and _m.get("map50") is not None

                st.subheader(f"{display_name} Object-Detection — {role_label}")
                st.caption(
                    f"{display_name} detector · fire / smoke · "
                    "bounding boxes + confidence · D-Fire test split"
                )

                if not _ran:
                    st.warning(
                        f"{display_name} measured result file not found. "
                        "No metric values are shown until a measured result file exists "
                        f"(`results/baseline_{_slug}.json`). No placeholder values are invented."
                    )

                # Metric cards
                def _fmt(v):
                    return f"{v:.4f}" if v is not None else "—"

                _yk1, _yk2, _yk3, _yk4, _yk5 = st.columns(5)
                _yk1.metric("mAP@0.5",       _fmt(_m.get("map50")))
                _yk2.metric("mAP@0.5:0.95",  _fmt(_m.get("map50_95")))
                _yk3.metric("Precision",      _fmt(_m.get("precision")))
                _yk4.metric("Recall",         _fmt(_m.get("recall")))
                _yk5.metric("F1",             _fmt(_m.get("f1")))

                if model_key == "YOLO11s":
                    st.info(
                        "YOLO11s is the **current primary detector** for PyroFinder. "
                        "Like YOLO11n it is an object detector — it predicts bounding boxes, "
                        "class labels, and confidence scores for fire and smoke — but it is the "
                        "larger model that delivers the best detection quality. "
                        "YOLO11n remains the lightweight speed baseline / fallback; YOLO11s is "
                        "compared against it using detection metrics (mAP, precision, recall), "
                        "never against the sklearn image-level classifiers."
                    )
                else:
                    st.info(
                        "YOLO11n is the lightweight object-detection **baseline / fallback** for PyroFinder. "
                        "Unlike the sklearn baselines, it does not classify the whole image only — "
                        "it predicts bounding boxes, class labels, and confidence scores for fire and smoke. "
                        "This makes it the correct baseline for the YOLO11s current primary detector, because "
                        "PyroFinder needs localization for approximate map-based alerts. "
                        "YOLO11n is a speed-oriented baseline, not an equal parallel model to YOLO11s."
                    )

                if _r.get("run_date"):
                    _ds = _r.get("dataset", {})
                    st.caption(
                        f"Run date: {_r['run_date']}  ·  "
                        f"Dataset: {_ds.get('name', '—')}  ·  "
                        f"Train: {_ds.get('train_size', '—'):,}  ·  "
                        f"Test: {_ds.get('test_size', '—'):,}"
                    )

                st.divider()

                # Metrics table
                st.subheader("Detection metrics")
                _det_rows = [
                    {"Metric": "mAP@0.5",      "Value": _fmt(_m.get("map50")),    "Meaning": "Detection quality at IoU threshold 0.5"},
                    {"Metric": "mAP@0.5:0.95", "Value": _fmt(_m.get("map50_95")), "Meaning": "Stricter detection quality across IoU thresholds 0.5 to 0.95"},
                    {"Metric": "Precision",     "Value": _fmt(_m.get("precision")),"Meaning": "How many predicted detections are correct"},
                    {"Metric": "Recall",        "Value": _fmt(_m.get("recall")),   "Meaning": "How many real fire/smoke objects are found"},
                    {"Metric": "F1",            "Value": _fmt(_m.get("f1")),       "Meaning": "Balance between precision and recall"},
                ]
                st.dataframe(pd.DataFrame(_det_rows), use_container_width=True, hide_index=True)

                # Per-class table
                _per_class = _m.get("per_class")
                if _per_class:
                    st.subheader("Per-class metrics")
                    _pc_rows = []
                    for _cls_name in ["smoke", "fire"]:
                        _cls_d = _per_class.get(_cls_name, {})
                        _pc_rows.append({
                            "Class":      _cls_name,
                            "mAP@0.5":    _fmt(_cls_d.get("map50")),
                            "mAP@0.5:0.95": _fmt(_cls_d.get("map50_95")),
                        })
                    st.dataframe(pd.DataFrame(_pc_rows), use_container_width=True, hide_index=True)

                # ── Run metadata ──────────────────────────────────────────────
                if _ran:
                    with st.expander("Run metadata", expanded=False):
                        _ds = _r.get("dataset", {})
                        _mp = _r.get("model_params", {})

                        def _meta_int(v):
                            return f"{v:,}" if isinstance(v, int) else "—"

                        _sel_epoch = _mp.get("selected_epoch_or_row")
                        if _sel_epoch is None:
                            _sel_epoch = _m.get("selected_epoch_or_row")
                        _batch = _mp.get("batch", _mp.get("batch_requested"))
                        _meta_rows = [
                            ("Run date", _r.get("run_date", "—")),
                            ("Dataset", _ds.get("name", "—")),
                            ("Train size", _meta_int(_ds.get("train_size"))),
                            ("Test size", _meta_int(_ds.get("test_size"))),
                            ("Image size", _mp.get("imgsz", "—")),
                            ("Epochs requested", _mp.get("epochs_requested", "—")),
                            ("Selected epoch / row", _sel_epoch if _sel_epoch is not None else "—"),
                            ("Batch", _batch if _batch is not None else "—"),
                        ]
                        if _mp.get("run_dir"):
                            _meta_rows.append(("Run directory", _mp.get("run_dir")))
                        st.dataframe(
                            pd.DataFrame(_meta_rows, columns=["Field", "Value"]).astype(str),
                            use_container_width=True, hide_index=True,
                        )

                st.divider()

                # ── Training curves ───────────────────────────────────────────
                _results_csv_primary = Path(training_csv_path)
                _runs_csv_fallback = Path(fallback_csv_path)
                _runs_csv = _results_csv_primary if _results_csv_primary.exists() else _runs_csv_fallback
                if _runs_csv.exists():
                    try:
                        _tc_df = pd.read_csv(_runs_csv)
                        _tc_df.columns = [c.strip() for c in _tc_df.columns]

                        _loss_cols = {
                            "train/box_loss": "Train box loss",
                            "train/cls_loss": "Train cls loss",
                            "train/dfl_loss": "Train dfl loss",
                            "val/box_loss":   "Val box loss",
                            "val/cls_loss":   "Val cls loss",
                        }
                        _map_cols = {
                            "metrics/mAP50(B)":     "mAP@0.5",
                            "metrics/mAP50-95(B)":  "mAP@0.5:0.95",
                            "metrics/precision(B)": "Precision",
                            "metrics/recall(B)":    "Recall",
                        }

                        _epoch_col = "epoch" if "epoch" in _tc_df.columns else None
                        if _epoch_col:
                            _tc_l, _tc_r = st.columns(2)

                            with _tc_l:
                                st.subheader("Training loss vs. epoch")
                                _loss_fig = go.Figure()
                                _loss_colors = {
                                    "Train box loss": "#e07b39",
                                    "Train cls loss": "#4fc3f7",
                                    "Train dfl loss": "#81c784",
                                    "Val box loss":   "#e07b39",
                                    "Val cls loss":   "#4fc3f7",
                                }
                                _loss_dash = {
                                    "Train box loss": "solid",
                                    "Train cls loss": "solid",
                                    "Train dfl loss": "solid",
                                    "Val box loss":   "dash",
                                    "Val cls loss":   "dash",
                                }
                                for _col, _label in _loss_cols.items():
                                    if _col in _tc_df.columns:
                                        _loss_fig.add_trace(go.Scatter(
                                            x=_tc_df[_epoch_col],
                                            y=_tc_df[_col],
                                            mode="lines",
                                            name=_label,
                                            line=dict(
                                                color=_loss_colors.get(_label, "#cccccc"),
                                                dash=_loss_dash.get(_label, "solid"),
                                                width=2,
                                            ),
                                        ))
                                _loss_fig.update_layout(
                                    xaxis_title="Epoch",
                                    yaxis_title="Loss",
                                    legend=dict(orientation="h", yanchor="top", y=-0.25),
                                    height=340,
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#cccccc"),
                                    margin=dict(l=50, r=20, t=20, b=60),
                                )
                                apply_chart_theme(_loss_fig)
                                st.plotly_chart(_loss_fig, use_container_width=True, key=f"yolo_loss_curve_{_slug}")

                            with _tc_r:
                                st.subheader("Validation metrics vs. epoch")
                                _map_fig = go.Figure()
                                _map_colors = ["#e07b39", "#4fc3f7", "#81c784", "#ffb74d"]
                                for _i, (_col, _label) in enumerate(_map_cols.items()):
                                    if _col in _tc_df.columns:
                                        _map_fig.add_trace(go.Scatter(
                                            x=_tc_df[_epoch_col],
                                            y=_tc_df[_col],
                                            mode="lines",
                                            name=_label,
                                            line=dict(color=_map_colors[_i % len(_map_colors)], width=2),
                                        ))
                                _map_fig.update_layout(
                                    xaxis_title="Epoch",
                                    yaxis_title="Score (0–1)",
                                    yaxis_range=[0, 1],
                                    legend=dict(orientation="h", yanchor="top", y=-0.25),
                                    height=340,
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#cccccc"),
                                    margin=dict(l=50, r=20, t=20, b=60),
                                )
                                apply_chart_theme(_map_fig)
                                st.plotly_chart(_map_fig, use_container_width=True, key=f"yolo_map_curve_{_slug}")

                            _n_epochs_done = int(_tc_df[_epoch_col].max()) + 1
                            st.caption(
                                f"Showing {_n_epochs_done} epoch(s) completed so far · "
                                f"solid lines = train · dashed lines = val · "
                                f"source: `{_runs_csv}`"
                            )
                    except Exception as _e:
                        st.info(f"Could not load training curves: {_e}")
                else:
                    st.info(
                        "Training curves not found. "
                        f"Expected at: `{training_csv_path}` "
                        f"(fallback: `{fallback_csv_path}`)."
                    )

                st.divider()

                with st.expander(f"Detailed analysis — what {display_name} tells us", expanded=False):
                    if model_key == "YOLO11s":
                        st.markdown("""
YOLO11s is the current primary detector for PyroFinder. Like YOLO11n it performs object detection — it predicts **where** fire or smoke appears in the frame — but it is the larger model that delivers the best detection quality.

---

#### What the result means

The key metrics are mAP, precision, and recall, not accuracy. mAP measures whether the predicted bounding boxes overlap the real fire/smoke boxes. Recall is especially important because missing real fire or smoke is more dangerous than creating a false alert.

---

#### What it tells us about the data

This evaluation tests whether D-Fire annotations let the larger YOLO11s model learn stronger localization patterns for fire and smoke than the lightweight YOLO11n baseline.

---

#### What it tells us about the model

YOLO11s is heavier than YOLO11n and is expected to be more accurate. It should be compared against YOLO11n using detection metrics (mAP@0.5, recall), never against the sklearn image-level classifiers.

---

#### Main conclusion

YOLO11s is selected as the main detector only if it improves detection quality — especially mAP@0.5 and recall — and the operational alert metrics over YOLO11n, while keeping acceptable inference speed. YOLO11n remains the lightweight speed baseline / fallback.
""")
                    else:
                        st.markdown("""
YOLO11n is different from the sklearn baselines. The sklearn models classify an entire image as background, fire, or smoke using handcrafted color features. YOLO11n performs object detection: it predicts **where** fire or smoke appears in the frame.

---

#### What the result means

The key metrics are mAP, precision, and recall, not accuracy. mAP measures whether the predicted bounding boxes overlap the real fire/smoke boxes. Recall is especially important because missing real fire or smoke is more dangerous than creating a false alert.

---

#### What it tells us about the data

This evaluation tests whether D-Fire annotations are useful for object detection, not only for image-level classification. It also shows whether the model can learn separate localization patterns for fire and smoke.

---

#### What it tells us about the model

YOLO11n is lightweight and fast, but it may be less accurate than YOLO11s. It is useful as a speed-oriented baseline and fallback model.

---

#### Main conclusion

YOLO11n is the correct object-detection baseline for PyroFinder. YOLO11s should be selected as the main model only if it improves detection quality, especially mAP@0.5 and recall, while keeping acceptable inference speed.
""")

            # ── Helper: build comparison dataframe ───────────────────────────
            def _build_comparison_df(results_data):
                _cmp_rows = []
                for _n, _d in results_data.items():
                    _m = _d.get("metrics", {})
                    _cr = _m.get("classification_report", {})
                    _ma = _m.get("macro_avg", {})
                    _cmp_rows.append({
                        "Model":             _short_model_label(_n),
                        "Accuracy":          round(_m.get("accuracy", 0), 2),
                        "Macro Precision":   round(_ma.get("precision", 0), 2),
                        "Macro Recall":      round(_ma.get("recall", 0), 2),
                        "Macro F1":          round(_ma.get("f1", 0), 2),
                        "Fire Recall":       round(_cr.get("fire", {}).get("recall", 0), 2),
                        "Smoke Recall":      round(_cr.get("smoke", {}).get("recall", 0), 2),
                        "Background Recall": round(_cr.get("background", {}).get("recall", 0), 2),
                        "Run date":          _d.get("run_date", "—"),
                    })
                return pd.DataFrame(_cmp_rows)

            # ── Helper: render comparison tab ────────────────────────────────
            def _render_model_comparison(results_data):
                _sklearn_data = {n: d for n, d in results_data.items() if _is_sklearn_result(d)}
                _yolo_data    = {n: d for n, d in results_data.items() if _is_object_detection_result(d)}

                 # ── A. Conclusion text ────────────────────────────────────────
                st.info(
                    "**Comparison conclusion:** "
                    "The sklearn baselines test whether simple image-level color features can separate "
                    "background, fire, and smoke. DummyClassifier is only a minimum bar, Logistic Regression "
                    "proves that color features contain signal, and Random Forest is the strongest classical "
                    "image-level baseline. "
                    "YOLO11n is different: it is the first object-detection baseline. It should be judged by "
                    "mAP, precision, and recall because it predicts bounding boxes, not just image labels. "
                    "The final YOLO11s model should be compared mainly against YOLO11n, not against the sklearn "
                    "models, because both YOLO models solve the real PyroFinder task: detecting and localizing "
                    "fire/smoke."
                )

                st.divider()


                # ── B. Sklearn classification baselines ──────────────────────
                st.subheader("Image-level sklearn classification baselines")
                st.caption(
                    "These baselines classify the whole image using 60 handcrafted color features. "
                    "They are evaluated with image-level classification metrics (accuracy, Macro F1, recall)."
                )

                if _sklearn_data:
                    _cmp_df = _build_comparison_df(_sklearn_data)

                    st.dataframe(_cmp_df, use_container_width=True, hide_index=True)

                    st.divider()

                    # Macro F1 bar chart — sklearn only
                    st.subheader("Macro F1 comparison — sklearn baselines")
                    _fig_cmp = px.bar(
                        _cmp_df,
                        x="Model", y="Macro F1",
                        color="Model",
                        text="Macro F1",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        title="Macro F1 — sklearn image-level baselines",
                        labels={"Macro F1": "F1 macro (higher is better)"},
                    )
                    _fig_cmp.update_layout(yaxis_range=[0, 1], showlegend=False)
                    apply_chart_theme(_fig_cmp)
                    st.plotly_chart(_fig_cmp, use_container_width=True, key="baseline_cmp_f1_bar")

                    st.divider()

                    # Radar chart — sklearn only
                    st.subheader("Macro average radar — sklearn baselines")
                    _radar_cats = ["Macro Precision", "Macro Recall", "Macro F1", "Accuracy"]
                    _radar_colors = px.colors.qualitative.Set2
                    _radar_fig = go.Figure()
                    for _i, (_, _row) in enumerate(_cmp_df.iterrows()):
                        _vals = [
                            _row["Macro Precision"],
                            _row["Macro Recall"],
                            _row["Macro F1"],
                            _row["Accuracy"],
                        ]
                        _radar_fig.add_trace(go.Scatterpolar(
                            r=_vals + [_vals[0]],
                            theta=_radar_cats + [_radar_cats[0]],
                            fill="toself",
                            name=_row["Model"],
                            line=dict(color=_radar_colors[_i % len(_radar_colors)], width=2),
                        ))
                    _radar_fig.update_layout(
                        polar=dict(
                            radialaxis=dict(range=[0, 1], tickfont=dict(size=10)),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cccccc"),
                        height=400,
                        margin=dict(l=60, r=60, t=50, b=40),
                        showlegend=True,
                    )
                    st.plotly_chart(_radar_fig, use_container_width=True, key="baseline_cmp_radar")
                else:
                    st.info("No sklearn baseline results found in `results/`.")

                st.divider()

                # ── C. Object-detection baselines (YOLO11n + YOLO11s) ─────────
                from src.results_loader import (
                    load_detection_result as _load_det,
                    status_label as _status_label,
                    STATUS_OK as _STATUS_OK,
                )

                st.subheader("Object-detection comparison")
                st.caption(
                    "YOLO11n (baseline / fallback) and YOLO11s (current primary detector) "
                    "are evaluated with detection metrics (mAP, precision, recall, F1). "
                    "These are not classification accuracy and must not be compared to the "
                    "sklearn baselines. A detector row shows a missing-file status until its "
                    "measured result file exists — no values are invented."
                )

                # Expected detection result files. Each detector is loaded the same way;
                # if a measured file is absent the loader reports a missing-file status
                # instead of fabricating metrics.
                _det_specs = [
                    ("YOLO11n", "results/baseline_yolo11n.json"),
                    ("YOLO11s", "results/baseline_yolo11s.json"),
                ]

                def _fv(v):
                    return round(v, 4) if v is not None else "—"

                _det_rows = []
                _det_loaded = {}
                for _dname, _dpath in _det_specs:
                    _loaded = _load_det(_dpath)
                    _det_loaded[_dname] = _loaded
                    _dm = (_loaded["data"] or {}).get("metrics", {}) if _loaded["status"] == _STATUS_OK else {}
                    _det_rows.append({
                        "Model":        _dname,
                        "mAP@0.5":      _fv(_dm.get("map50")),
                        "mAP@0.5:0.95": _fv(_dm.get("map50_95")),
                        "Precision":    _fv(_dm.get("precision")),
                        "Recall":       _fv(_dm.get("recall")),
                        "F1":           _fv(_dm.get("f1")),
                        "Status":       _status_label(_loaded["status"]),
                    })
                st.dataframe(
                    pd.DataFrame(_det_rows)[
                        ["Model", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1", "Status"]
                    ],
                    use_container_width=True, hide_index=True,
                )

                # Radar chart for any measured detector.
                _radar_colors = {"YOLO11n": "#e07b39", "YOLO11s": "#4fc3f7"}
                _det_radar_fig = go.Figure()
                _radar_cats = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]
                _any_measured = False
                for _dname, _loaded in _det_loaded.items():
                    if _loaded["status"] != _STATUS_OK:
                        continue
                    _dm = (_loaded["data"] or {}).get("metrics", {})
                    if not all(_dm.get(k) is not None for k in ["map50", "map50_95", "precision", "recall", "f1"]):
                        continue
                    _any_measured = True
                    _vals = [_dm["map50"], _dm["map50_95"], _dm["precision"], _dm["recall"], _dm["f1"]]
                    _det_radar_fig.add_trace(go.Scatterpolar(
                        r=_vals + [_vals[0]],
                        theta=_radar_cats + [_radar_cats[0]],
                        fill="toself",
                        line=dict(color=_radar_colors.get(_dname, "#81c784"), width=2),
                        name=_dname,
                    ))
                if _any_measured:
                    st.subheader("Detection radar — measured detectors")
                    _det_radar_fig.update_layout(
                        polar=dict(
                            radialaxis=dict(range=[0, 1], tickfont=dict(size=10)),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cccccc"),
                        height=380,
                        margin=dict(l=60, r=60, t=50, b=40),
                    )
                    st.plotly_chart(_det_radar_fig, use_container_width=True, key="baseline_yolo_radar")

                st.caption(
                    "YOLO11s is the current primary detector. A detector is selected as the main "
                    "model only when its measured detection and operational result files exist and "
                    "it wins the operational selection rule — never from missing or incomplete files."
                )

                st.divider()

               
            # ── Helper: Operational Alert Metrics (cost-sensitive comparison) ─
            def _render_operational_alert_metrics(results_data):
                from src.evaluation import (
                    operational_alert_metrics_from_confusion_matrix as _op_from_cm,
                )
                from src.results_loader import (
                    load_operational_result as _load_op,
                    is_selectable_operational as _is_selectable,
                    select_operational_winner as _select_winner,
                    status_label as _status_label,
                    STATUS_OK as _STATUS_OK,
                )

                st.subheader("Operational Alert Metrics")
                st.caption(
                    "Cost-sensitive comparison at the alert level: fire/smoke = hazard, "
                    "background = no hazard. A missed hazard (false negative) is weighted "
                    "**10×** a false alert (false positive). "
                    "**Primary decision metric: Hazard Recall.** Secondary: False Alert Rate. "
                    "Final ranking: Operational Alert Score (FN weight 10, FP weight 1)."
                )

                def _op_for(d):
                    # Prefer embedded operational_metrics; else derive from a stored
                    # confusion matrix (same predictions, no re-run).
                    om = d.get("operational_metrics")
                    if om:
                        return om
                    cm = d.get("metrics", {}).get("confusion_matrix")
                    if cm and cm.get("matrix") and cm.get("labels"):
                        return _op_from_cm(cm["matrix"], cm["labels"])
                    return None

                def _fmt(v):
                    return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"

                # Expected YOLO operational result files. A detector reports a
                # missing-file status until its measured file exists.
                _op_specs = [
                    ("YOLO11n", "results/yolo11n_operational_metrics.json",
                     "results/baseline_yolo11n.json"),
                    ("YOLO11s", "results/yolo11s_operational_metrics.json",
                     "results/baseline_yolo11s.json"),
                ]

                rows, chart_data = [], []

                # ── sklearn rows (image-level classifiers — no localization) ──
                _sk = {n: d for n, d in results_data.items()
                       if not _is_object_detection_result(d)}
                for _n in sorted(_sk, key=_model_sort_key):
                    _om = _op_for(_sk[_n])
                    if not _om:
                        continue
                    _label = _short_model_label(_n)
                    rows.append({
                        "Model": _label,
                        "Hazard Recall": _fmt(_om.get("hazard_recall")),
                        "False Alert Rate": _fmt(_om.get("false_alert_rate")),
                        "Alert Precision": _fmt(_om.get("alert_precision")),
                        "Alert F1": _fmt(_om.get("alert_f1")),
                        "Operational Alert Score": _fmt(_om.get("operational_alert_score")),
                        "Location Coverage": "N/A",
                        "Mean Location Error": "N/A",
                        "3x3 Grid Hit Rate": "N/A",
                        "Status": "Measured (image-level)",
                    })
                    chart_data.append({"Model": _label, "Metric": "Hazard Recall",
                                       "Value": _om.get("hazard_recall", 0)})
                    chart_data.append({"Model": _label, "Metric": "Operational Alert Score",
                                       "Value": _om.get("operational_alert_score", 0)})

                # ── YOLO operational rows (dedicated operational JSON per model) ──
                _missing_yolo_op = []
                for _mname, _op_path, _det_path in _op_specs:
                    _loaded = _load_op(_op_path)
                    if _loaded["status"] != _STATUS_OK:
                        # No measured file yet → explicit status row, no invented values.
                        _missing_yolo_op.append((_mname, _op_path, _loaded["status"]))
                        rows.append({
                            "Model": _mname,
                            "Hazard Recall": "—",
                            "False Alert Rate": "—",
                            "Alert Precision": "—",
                            "Alert F1": "—",
                            "Operational Alert Score": "—",
                            "Location Coverage": "—",
                            "Mean Location Error": "—",
                            "3x3 Grid Hit Rate": "—",
                            "Status": _status_label(_loaded["status"]),
                        })
                        continue
                    _data = _loaded["data"] or {}
                    _om = _data.get("operational_metrics", {})
                    _lm = _data.get("location_metrics", {}) or {}
                    _label = _data.get("model_name", _mname)
                    rows.append({
                        "Model": _label,
                        "Hazard Recall": _fmt(_om.get("hazard_recall")),
                        "False Alert Rate": _fmt(_om.get("false_alert_rate")),
                        "Alert Precision": _fmt(_om.get("alert_precision")),
                        "Alert F1": _fmt(_om.get("alert_f1")),
                        "Operational Alert Score": _fmt(_om.get("operational_alert_score")),
                        "Location Coverage": _fmt(_lm.get("location_coverage_rate")),
                        "Mean Location Error": _fmt(_lm.get("fire_location_error_mean")),
                        "3x3 Grid Hit Rate": _fmt(_lm.get("fire_location_grid_hit_rate")),
                        "Status": "Measured" if _is_selectable(_loaded) else "Measured (incomplete)",
                    })
                    chart_data.append({"Model": _label, "Metric": "Hazard Recall",
                                       "Value": _om.get("hazard_recall", 0)})
                    chart_data.append({"Model": _label, "Metric": "Operational Alert Score",
                                       "Value": _om.get("operational_alert_score", 0)})

                _table_cols = [
                    "Model", "Hazard Recall", "False Alert Rate",
                    "Alert Precision", "Alert F1", "Operational Alert Score",
                    "Location Coverage", "Mean Location Error", "3x3 Grid Hit Rate",
                    "Status",
                ]
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows)[_table_cols],
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("No operational alert metrics available in `results/` yet.")

                # ── Winner selection (YOLO detectors only; pending never wins) ──
                _winner = _select_winner(
                    [(m, op) for m, op, _ in _op_specs],
                    detection_items=[(m, det) for m, _, det in _op_specs],
                )
                if _winner:
                    st.success(
                        f"**Selected detector: {_winner}** — chosen by Hazard Recall, then "
                        "False Alert Rate, then Operational Alert Score (object-detection "
                        "recall / mAP@0.5 as supporting metrics). Selection uses only models "
                        "with measured, complete result files."
                    )
                else:
                    st.info(
                        "No detector selected yet: a model is chosen only when its measured "
                        "operational result file exists with complete metrics. A detector with "
                        "missing or incomplete result files cannot be selected."
                    )

                # ── M3 failure-analysis summary (how YOLO11s wins + open weakness) ──
                st.info(
                    "YOLO11s is selected, but the operational gain over YOLO11n is modest and "
                    "consistent, not large. The detection-quality gain is clearer, especially "
                    "mAP@0.5. Smoke-only images remain the dominant failure mode for both "
                    "detectors."
                )
                with st.expander("M3 failure-analysis notes", expanded=False):
                    st.markdown(
                        "- 9 fewer missed hazards (FN 145 vs 154).\n"
                        "- 5 fewer false alerts (FP 37 vs 42).\n"
                        "- +0.0038 Operational Alert Score (0.9406 vs 0.9368).\n"
                        "- +1.98 pp mAP@0.5 (0.7668 vs 0.7470), with consistent gains on "
                        "mAP@0.5:0.95, Precision, Recall, and F1.\n"
                        "- Smoke-only imagery is the main bottleneck (most misses for both "
                        "detectors); the paired comparison shows partial complementarity — "
                        "YOLO11s does not beat YOLO11n on every individual image.\n"
                        "- Approximate fire-location metrics are practically tied; YOLO11s has "
                        "slightly better coverage and 3x3 grid-hit rate. Location outputs are "
                        "approximate image-space estimates only, never precise geolocation.\n\n"
                        "Full analysis: `docs/M3_RESULTS_SUMMARY.md`."
                    )

                # Compact chart: Hazard Recall + Operational Alert Score per model
                if chart_data:
                    _fig_op = px.bar(
                        pd.DataFrame(chart_data),
                        x="Model", y="Value", color="Metric", barmode="group",
                        color_discrete_map={
                            "Hazard Recall": "#e07b39",
                            "Operational Alert Score": "#4fc3f7",
                        },
                        labels={"Value": "Score (0–1, higher is better)"},
                        title="Hazard Recall vs Operational Alert Score (FN weight 10, FP weight 1)",
                    )
                    _fig_op.update_layout(yaxis_range=[0, 1], bargap=0.25, height=360)
                    apply_chart_theme(_fig_op)
                    st.plotly_chart(_fig_op, use_container_width=True, key="baseline_operational_bar")

                # ── Missing YOLO operational metrics → show generation command ──
                for _mname, _op_path, _ in _missing_yolo_op:
                    _weights = f"models/{_mname.lower()}_dfire_best.pt"
                    _csv = _op_path.replace("operational_metrics.json", "test_predictions.csv")
                    st.info(
                        f"{_mname} operational alert metrics not found (`{_op_path}`). "
                        "Generate them (evaluation only, no training) with:\n\n"
                        "```\n"
                        "py scripts/evaluate_yolo_alert_metrics.py "
                        "--raw-root \"<path-to-D-Fire-root>\" "
                        f"--weights \"{_weights}\" "
                        f"--model-name \"{_mname}\" --conf 0.25 "
                        f"--output-json \"{_op_path}\" "
                        f"--output-csv \"{_csv}\"\n"
                        "```"
                    )

                st.caption(
                    "Location metrics apply only to object detectors (YOLO11n / YOLO11s). "
                    "The sklearn models are **image-level baselines only** and cannot replace "
                    "YOLO11s, the current primary detector. YOLO11n is the lightweight "
                    "object-detection baseline / fallback. Both YOLO models solve the real "
                    "detection + localization task PyroFinder needs; the sklearn classifiers "
                    "produce no bounding boxes."
                )

            # ── Separate sklearn vs object-detection results ──────────────────
            _sklearn_names = sorted(
                [n for n, d in _results_data.items() if _is_sklearn_result(d)],
                key=_model_sort_key,
            )

            # Look up each YOLO detection result by model key. A missing file is
            # simply absent from _results_data, so this returns None and the
            # renderer shows a pending / missing-file state — never invented values.
            def _find_yolo_result(key_substr):
                for _n, _d in _results_data.items():
                    if _is_object_detection_result(_d) and key_substr in _n.lower():
                        return _d
                return None

            _yolo11n_result = _find_yolo_result("yolo11n")
            _yolo11s_result = _find_yolo_result("yolo11s")

            # ── Build tabs: Dummy → LR → RF → YOLO11n → YOLO11s → Model comparison
            _sklearn_labels = [_short_model_label(n) for n in _sklearn_names]
            _tab_labels = _sklearn_labels + ["YOLO11n", "YOLO11s", "Model comparison"]
            _model_tabs = st.tabs(_tab_labels)

            # ── Render sklearn tabs ───────────────────────────────────────────
            for _tab, _mname in zip(_model_tabs[:len(_sklearn_names)], _sklearn_names):
                with _tab:
                    _render_single_baseline_model(_mname, _results_data[_mname])

            # ── YOLO11n tab (lightweight baseline / fallback) ─────────────────
            with _model_tabs[len(_sklearn_names)]:
                _render_yolo_detection_model(
                    "YOLO11n", "YOLO11n", "lightweight baseline / fallback",
                    _yolo11n_result,
                    "results/results_yolo11n.csv",
                    "runs/detect/yolo11n_dfire_baseline/results.csv",
                )

            # ── YOLO11s tab (current primary detector) ────────────────────────
            with _model_tabs[len(_sklearn_names) + 1]:
                _render_yolo_detection_model(
                    "YOLO11s", "YOLO11s", "current primary detector",
                    _yolo11s_result,
                    "results/results_yolo11s.csv",
                    "runs/detect/yolo11s_dfire_baseline/results.csv",
                )

            # ── Model comparison tab (incl. Operational Alert Metrics) ────────
            # Operational Alert Metrics live ONLY here so they are not repeated
            # under every model sub-tab.
            with _model_tabs[-1]:
                _render_model_comparison(_results_data)
                st.divider()
                _render_operational_alert_metrics(_results_data)

    # ── Inference Demo ───────────────────────────────────────────────────────
    with tab_inference:
        from src import inference as _inf

        st.header("Inference Demo — YOLO11n / YOLO11s")
        st.caption(
            "Upload one image and run the fine-tuned D-Fire detectors. Models load only "
            "when you click **Run inference** — never at import time — and only the "
            "fine-tuned `fire`/`smoke` checkpoints are used. Pretrained weights are never "
            "substituted. Inference time is measured during the run; no value is estimated."
        )

        _n_available = _inf.checkpoint_exists("YOLO11n")
        _s_available = _inf.checkpoint_exists("YOLO11s")

        # YOLO11s is the current primary detector; until its checkpoint exists the
        # side-by-side / YOLO11s options are simply not offered.
        if not _s_available:
            st.warning(_inf.MISSING_YOLO11S_MESSAGE)
        if not _n_available:
            st.warning(
                "YOLO11n checkpoint not found. Add `models/yolo11n_dfire_best.pt` "
                "(e.g. `python scripts/YOLO11n_baseline.py --train`) to enable YOLO11n inference."
            )

        _options = []
        if _n_available:
            _options.append("YOLO11n only")
        if _s_available:
            _options.append("YOLO11s only")
        if _n_available and _s_available:
            _options.append("Side-by-side comparison")

        if not _options:
            st.info(
                "No fine-tuned checkpoints available yet. Add at least one of "
                "`models/yolo11n_dfire_best.pt` or `models/yolo11s_dfire_best.pt` to run inference."
            )
        else:
            _choice = st.radio("Detector", _options, horizontal=True)
            _uploaded = st.file_uploader("Upload one image", type=["jpg", "jpeg", "png"])
            _run = st.button(
                "Run inference", type="primary", disabled=_uploaded is None
            )

            if _uploaded is not None and _run:
                from PIL import Image as _PILImage

                _image = _PILImage.open(_uploaded).convert("RGB")
                if _choice == "Side-by-side comparison":
                    _models_to_run = ["YOLO11n", "YOLO11s"]
                elif _choice == "YOLO11s only":
                    _models_to_run = ["YOLO11s"]
                else:
                    _models_to_run = ["YOLO11n"]

                _cols = st.columns(len(_models_to_run))
                for _col, _mname in zip(_cols, _models_to_run):
                    with _col:
                        st.subheader(_mname)
                        try:
                            with st.spinner(f"Loading {_mname} and running inference..."):
                                _model = load_detector_cached(_mname)
                                _res = _inf.run_detection(
                                    _model, _image, conf=confidence_threshold
                                )
                            st.image(
                                _res["annotated_png"],
                                caption=f"{_mname} — {_res['total_detections']} detection(s)",
                                use_container_width=True,
                            )
                            _ic1, _ic2 = st.columns(2)
                            _ic1.metric("Fire detections", _res["fire_count"])
                            _ic2.metric("Smoke detections", _res["smoke_count"])
                            _ic3, _ic4 = st.columns(2)
                            _hc = _res["max_confidence"]
                            _ic3.metric(
                                "Highest confidence",
                                f"{_hc:.2f}" if _hc is not None else "—",
                            )
                            _ic4.metric("Inference time", f"{_res['inference_ms']:.0f} ms")
                        except FileNotFoundError:
                            st.warning(
                                _inf.MISSING_YOLO11S_MESSAGE if _mname == "YOLO11s"
                                else f"{_mname} checkpoint not found. Add its fine-tuned D-Fire checkpoint."
                            )
                        except ValueError as _ve:
                            st.error(str(_ve))
                        except Exception as _exc:
                            st.error(f"{_mname} inference failed: {_exc}")

        st.caption(
            f"Active confidence threshold: `{confidence_threshold}` · "
            f"Confirmation frames (N) for alert logic: `{confirmation_frames}` · "
            f"Classes: fire, smoke only."
        )

    # ── Mapping Setup ────────────────────────────────────────────────────────
    with tab_mapping:
        st.info(
            "**Coming in M3:** "
            "Draw named polygons on camera images, link them to map areas, "
            "and configure camera GPS and metadata for approximate location output."
        )
        st.progress(0, text="M3 progress: 0%")
        from src.mapping import get_mapping_modes
        st.markdown("**Supported mapping modes:**")
        for i, mode_name in enumerate(get_mapping_modes(), 1):
            st.markdown(f"{i}. {mode_name.capitalize()}")

        st.caption("All location outputs will be marked as approximate.")

    # ── Alert Log ────────────────────────────────────────────────────────────
    with tab_alerts:
        st.info(
            "**Coming in M3:** "
            "Confirmed fire/smoke alerts will appear here with camera ID, timestamp, "
            "apparent direction, approximate location, and status."
        )
        st.progress(0, text="M3 progress: 0%")
        st.markdown("""
**Alert statuses:**
- `active` — detection confirmed, awaiting review
- `confirmed` — verified as real event
- `rejected` — dismissed by operator
- `false_alarm` — marked as false positive
        """)

elif mode == "Central Control Dashboard":
    st.header("Central Control Dashboard")
    st.info(
        "**Basic version — coming in M3:** "
        "Map view of all cameras, camera metadata table, alert history, "
        "and manual image polygon setup."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Camera metadata (placeholder)")
        sample_cameras = pd.DataFrame({
            "camera_id": ["cam_001", "cam_002"],
            "site": ["north_boundary", "parking_area"],
            "status": ["active", "active"],
            "lat": [None, None],
            "lon": [None, None],
            "height_m": [None, None],
            "azimuth_deg": [None, None],
        })
        st.dataframe(sample_cameras, use_container_width=True)

    with col2:
        st.subheader("Map (placeholder)")
        st.caption("Map will display camera markers. Configure coordinates in camera metadata.")
        try:
            import folium
            from streamlit_folium import st_folium
            m = folium.Map(location=[32.0853, 34.7818], zoom_start=10)
            st_folium(m, height=300, use_container_width=True)
        except ImportError:
            st.info(
                "Map requires `folium` and `streamlit-folium`. "
                "Install with: `pip install folium streamlit-folium`"
            )

# ── M2 Course Dashboard ────────────────────────────────────────────────────────
elif mode == "M2 Course Dashboard":
    st.subheader("M2 Course Dashboard — Story Tabs")

    tab_problem, tab_lit, tab_market, tab_eda_story = st.tabs([
        "1. Problem Understanding",
        "2. Literature Review",
        "3. Market Review",
        "4. Dataset & EDA",
    ])

    # ── Tab 1: Problem Understanding ─────────────────────────────────────────
    with tab_problem:
        st.header("Problem Understanding")

        # One-sentence problem + value proposition
        with st.container():
            st.markdown(
                "**Problem:** Property owners in fire-prone areas cannot monitor every camera "
                "feed at once — a small fire becomes a crisis before anyone notices."
            )
            st.markdown(
                "**Value proposition:** PyroFinder watches your existing cameras automatically "
                "and sends a confirmed alert within seconds of detecting fire or smoke."
            )

        st.divider()

        # KPI
        st.subheader("KPI")
        st.markdown(
            "This is a two-class object-detection task. The primary operational KPI is "
            "**Hazard Recall** — missing a real fire/smoke hazard is far more costly than a "
            "false alarm. **False Alert Rate** is the secondary metric and **Operational Alert "
            "Score** (FN weight 10, FP weight 1) is the ranking summary. Detection Recall and "
            "mAP@0.5 are supporting object-detection quality metrics, not the same as Hazard Recall."
        )

        st.divider()

        # Stakeholder map
        st.subheader("Stakeholder Map")
        _sh_gap1, _sh_mid, _sh_gap2 = st.columns([1, 3, 1])
        with _sh_mid:
            st.components.v1.html("""
<!DOCTYPE html><html><head>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  body { margin:0; background:transparent; }
  .mermaid { background:transparent; opacity:0.7; }
  .mermaid svg { width:100% !important; max-width:100%; height:auto; }
</style>
</head><body>
<div class="mermaid">
%%{init:{"theme":"base","themeVariables":{"primaryColor":"#E4573D","primaryTextColor":"#F3F4F8","primaryBorderColor":"#F3F4F8","lineColor":"#D6D7E6","secondaryColor":"#3E445E","tertiaryColor":"#264036","edgeLabelBackground":"#2B3248","fontFamily":"Inter, Source Sans Pro, sans-serif","fontSize":"15px"}}}%%
quadrantChart
    title Stakeholder Map — Interest vs Influence
    x-axis Low Influence --> High Influence
    y-axis Low Interest --> High Interest
    quadrant-1 Manage Closely
    quadrant-2 Keep Informed
    quadrant-3 Monitor
    quadrant-4 Keep Satisfied
    Property Owner / Dani: [0.80, 0.88]
    Dev / ML Team: [0.72, 0.78]
    Operator / Admin: [0.88, 0.70]
    Farm Workers / Residents: [0.22, 0.72]
    Emergency Services: [0.78, 0.32]
    Camera Vendor / Integrator: [0.65, 0.18]
    Dataset / Research Sources: [0.20, 0.14]
</div>
<script>mermaid.initialize({startOnLoad:true,securityLevel:"loose"});</script>
</body></html>
""", height=500)

        _sh_data = [
            ("Property Owner / Dani",     "Manage Closely",  "Primary user — directly affected by alerts and fire risk",                "Weekly demos, usability feedback, alert UX review"),
            ("Dev / ML Team",             "Manage Closely",  "Builds and improves the detection model and dashboard",                   "Sprint planning, model performance reviews"),
            ("Operator / Admin",          "Manage Closely",  "Runs the system, manages cameras and alert configuration",                "Ops documentation, alert tuning, incident log review"),
            ("Farm Workers / Residents",  "Keep Informed",   "Affected by fire risk but do not control the system",                     "Clear alert language, evacuation guidance"),
            ("Emergency Services",        "Keep Satisfied",  "High authority in fire response; PyroFinder does not auto-dispatch",      "Share detection reports on request; future viewer dashboard"),
            ("Camera Vendor / Integrator","Keep Satisfied",  "Provides hardware PyroFinder depends on; limited day-to-day interest",    "Integration specs, compatibility requirements"),
            ("Dataset / Research Sources","Monitor",         "Enables model training; no active role in operations",                    "Citation, license compliance, periodic dataset updates"),
        ]
        _sh_tabs = st.tabs([row[0] for row in _sh_data])
        for _tab, (_name, _quadrant, _reason, _strategy) in zip(_sh_tabs, _sh_data):
            with _tab:
                st.markdown(f"**Quadrant:** {_quadrant}")
                st.markdown(f"**Reason:** {_reason}")
                st.markdown(f"**Communication strategy:** {_strategy}")

        st.divider()

        # Persona card
        st.subheader("Primary Persona")
        col_img, col_bio = st.columns([1, 3])
        with col_img:
            _persona_path = Path("design_images") / "DANI_PERSONA.png"
            if _persona_path.exists():
                import base64 as _b64mod
                _persona_b64 = _b64mod.b64encode(_persona_path.read_bytes()).decode()
                st.markdown(
                    f"<div style='width:150px;height:150px;border-radius:50%;overflow:hidden;"
                    f"margin:0 auto;box-shadow:0 0 0 3px rgba(228,87,61,0.5);'>"
                    f"<img src='data:image/png;base64,{_persona_b64}' "
                    f"style='width:100%;height:100%;object-fit:cover;display:block;' /></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='background:linear-gradient(135deg,#d4691e,#8b4513);"
                    "width:150px;height:150px;border-radius:50%;"
                    "display:flex;align-items:center;justify-content:center;"
                    "font-size:52px;margin:0 auto;'>🧑‍🌾</div>",
                    unsafe_allow_html=True,
                )
        with col_bio:
            with st.container():
                st.markdown("**Dani Cohen** — Farm Owner, Avivim, Israel")
                st.markdown("- 120-dunam agricultural farm with fixed outdoor cameras at boundary points")
                st.markdown("- Cannot continuously watch every feed during the dry summer months")
                st.markdown("- Risk: fires from neighbouring fields or agricultural machinery")
                st.markdown("- Goal: be alerted within minutes of any confirmed fire or smoke")

        st.divider()

        # Before / After journey
        st.subheader("User Journey")
        journey_before, journey_after = st.tabs(["Before PyroFinder", "After PyroFinder"])

        with journey_before:
            _jb_text, _jb_img = st.columns([1, 1], gap="small")
            with _jb_text:
                for _step_title, _step_body in [
                    ("Fire starts", "A spark from neighbouring machinery ignites dry brush at the property edge."),
                    ("No alert", "Dani's cameras capture smoke — but nobody is watching the screens."),
                    ("Late discovery", "Dani notices smoke from a window or gets a call from a neighbour — 15–30 minutes later."),
                    ("Crisis", "By the time emergency services arrive, the fire has already spread."),
                ]:
                    with st.container():
                        st.markdown(f"**{_step_title}**")
                        st.write(_step_body)
            with _jb_img:
                st.image("design_images/User_Journey_before.png", use_container_width=True)

        with journey_after:
            _ja_text, _ja_img = st.columns([1, 1], gap="small")
            with _ja_text:
                for _step_title, _step_body in [
                    ("Fire starts", "Same spark at the property edge."),
                    ("Detected", "NN object detection model detects smoke in the camera frame within seconds."),
                    ("Confirmed", "Detection confirmed across N consecutive frames — single-frame noise filtered out."),
                    ("Alert sent", "Dani receives an alert: camera ID, timestamp, approximate location, direction."),
                    ("Fast response", "Dani contacts emergency services within minutes. Fire is contained early."),
                ]:
                    with st.container():
                        st.markdown(f"**{_step_title}**")
                        st.write(_step_body)
            with _ja_img:
                st.image("design_images/User_Journey_after.png", use_container_width=True)

        st.divider()

        # Detection flow
        st.subheader("Detection Flow")
        st.caption(
            "Signal-to-alert pipeline — from existing RGB camera feed to reviewable alert record."
        )
        st.components.v1.html("""
<!DOCTYPE html>
<html>
<head>
<style>
  :root {
    --pf-card: rgba(18, 28, 31, 0.88);
    --pf-card-soft: rgba(23, 33, 37, 0.74);
    --pf-border: rgba(116, 151, 158, 0.55);
    --pf-border-soft: rgba(116, 151, 158, 0.28);
    --pf-text: #E8E3D8;
    --pf-muted: #9AA3A0;
    --pf-ember: #C8643F;
    --pf-cyan: #83C5BE;
    --pf-line: rgba(157, 177, 179, 0.70);
  }

  body {
    margin: 0;
    background: transparent;
    font-family: Inter, "Source Sans Pro", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--pf-text);
  }

  .pf-flow {
    padding: 16px 18px 18px;
    border: 1px solid var(--pf-border-soft);
    border-radius: 18px;
    background:
      radial-gradient(circle at 12% 0%, rgba(200, 100, 63, 0.16), transparent 28%),
      linear-gradient(180deg, rgba(12, 18, 22, 0.70), rgba(8, 12, 15, 0.44));
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
  }

  .pf-header {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: baseline;
    margin-bottom: 12px;
  }

  .pf-kicker {
    color: var(--pf-cyan);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
  }

  .pf-note {
    color: var(--pf-muted);
    font-size: 13px;
  }

  .pf-row {
    display: flex;
    align-items: stretch;
    gap: 8px;
    width: 100%;
  }

  .pf-row + .pf-row {
    margin-top: 10px;
  }

  .pf-step,
  .pf-reject {
    min-width: 0;
    flex: 1 1 0;
    padding: 12px 12px 11px;
    border-radius: 14px;
  }

  .pf-step {
    border: 1px solid var(--pf-border);
    background: linear-gradient(180deg, var(--pf-card), var(--pf-card-soft));
    position: relative;
  }

  .pf-step.core {
    border-color: rgba(200, 100, 63, 0.74);
    box-shadow: inset 0 0 0 1px rgba(200, 100, 63, 0.22);
  }

  .pf-step.output {
    border-color: rgba(131, 197, 190, 0.82);
    box-shadow: inset 0 0 0 1px rgba(131, 197, 190, 0.18);
  }

  .pf-num {
    display: inline-block;
    color: var(--pf-ember);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    margin-bottom: 7px;
  }

  .pf-step h4 {
    margin: 0 0 5px;
    font-size: 14px;
    line-height: 1.12;
    font-weight: 700;
    white-space: nowrap;
  }

  .pf-step p {
    margin: 0;
    color: var(--pf-muted);
    font-size: 12px;
    line-height: 1.3;
  }

  .pf-arrow {
    width: 22px;
    min-width: 22px;
    position: relative;
    align-self: center;
    height: 2px;
    background: var(--pf-line);
  }

  .pf-arrow::after {
    content: "";
    position: absolute;
    right: -1px;
    top: -4px;
    width: 0;
    height: 0;
    border-left: 8px solid var(--pf-line);
    border-top: 5px solid transparent;
    border-bottom: 5px solid transparent;
  }

  .pf-turn {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    margin: 5px 0 4px;
    padding-right: calc(12.5% + 11px);
    color: var(--pf-muted);
    font-size: 12px;
  }

  .pf-turn span {
    border-left: 1px solid var(--pf-line);
    border-bottom: 1px solid var(--pf-line);
    border-radius: 0 0 0 10px;
    padding: 7px 0 5px 12px;
  }

  .pf-reject {
    border: 1px dashed rgba(154, 163, 160, 0.48);
    background: rgba(10, 15, 18, 0.42);
    color: var(--pf-muted);
    font-size: 12px;
    line-height: 1.32;
  }

  .pf-reject strong {
    color: var(--pf-text);
    font-weight: 700;
  }

  @media (max-width: 900px) {
    .pf-step h4 { white-space: normal; }
    .pf-step p, .pf-reject { font-size: 11.5px; }
    .pf-arrow { width: 16px; min-width: 16px; }
  }
</style>
</head>
<body>
  <section class="pf-flow">
    <div class="pf-header">
      <div class="pf-kicker">Signal-to-alert pipeline</div>
      <div class="pf-note">No owner alert is sent before temporal confirmation.</div>
    </div>

    <div class="pf-row">
      <div class="pf-step">
        <span class="pf-num">01</span>
        <h4>Camera Input</h4>
        <p>Existing RGB security feed</p>
      </div>

      <div class="pf-arrow"></div>

      <div class="pf-step">
        <span class="pf-num">02</span>
        <h4>Frame Sampling</h4>
        <p>Periodic frame extraction</p>
      </div>

      <div class="pf-arrow"></div>

      <div class="pf-step core">
        <span class="pf-num">03</span>
        <h4>NN Detection Model</h4>
        <p>Fire / smoke + confidence</p>
      </div>

      <div class="pf-arrow"></div>

      <div class="pf-step core">
        <span class="pf-num">04</span>
        <h4>N-frame Confirmation</h4>
        <p>Filters single-frame noise</p>
      </div>
    </div>

    <div class="pf-turn">
      <span>confirmed signal continues to mapping and alerting</span>
    </div>

    <div class="pf-row">
      <div class="pf-reject">
        <strong>Not confirmed:</strong><br/>
        single-frame or non-persistent detections are ignored as operational alerts.
      </div>

      <div class="pf-step">
        <span class="pf-num">05</span>
        <h4>Mapping Layer</h4>
        <p>Approx. zone + direction</p>
      </div>

      <div class="pf-arrow"></div>

      <div class="pf-step output">
        <span class="pf-num">06</span>
        <h4>Alert Record</h4>
        <p>Camera · time · class · location</p>
      </div>

      <div class="pf-arrow"></div>

      <div class="pf-step output">
        <span class="pf-num">07</span>
        <h4>Dashboard Log</h4>
        <p>Review · confirm · reject</p>
      </div>
    </div>
  </section>
</body>
</html>
""", height=360)


    # ── Tab 2: Literature Review ──────────────────────────────────────────────
    with tab_lit:
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


    # ── Tab 3: Market Review ──────────────────────────────────────────────────
    with tab_market:
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

    # ── Tab 4: Dataset & EDA (story version) ─────────────────────────────────
    with tab_eda_story:
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

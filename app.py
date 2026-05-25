import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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

st.set_page_config(page_title="PyroFinder", layout="wide")

METADATA_PATH = "data/dfire_metadata.csv"
SAMPLES_DIR = Path("data/samples/dfire/images")
GENERATE_CMD = (
    'python scripts/build_dfire_metadata.py '
    '--raw-root "C:\\Users\\boris.azarov\\OneDrive - Technion\\Desktop\\PyroFinder\\RAW_DATA\\D-Fire" '
    "--output data/dfire_metadata.csv"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("PyroFinder")
    st.caption("Fire detection and monitoring — existing cameras only")

    st.divider()

    mode = st.selectbox(
        "Dashboard mode",
        ["Operations & Learning Dashboard", "Central Control Dashboard"],
    )

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

    st.divider()
    model_plan = get_model_plan()
    st.caption(f"Primary model: **{model_plan['primary_model']}**")
    st.caption(f"Baseline: {model_plan['baseline_model']}")
    st.caption(f"Classes: {', '.join(model_plan['classes'])}")

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("PyroFinder")
st.subheader("Real-time fire outbreak detection and monitoring using existing customer cameras.")

if mode == "Operations & Learning Dashboard":
    tab_overview, tab_eda, tab_inference, tab_mapping, tab_alerts = st.tabs([
        "Overview",
        "Dataset & EDA",
        "Inference Demo",
        "Mapping Setup",
        "Alert Log",
    ])

    # ── Overview ────────────────────────────────────────────────────────────
    with tab_overview:
        st.header("System Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Planned model")
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
            for _k, _v in _di["breakdown"].items():
                st.markdown(f"&nbsp;&nbsp;&nbsp;• {_k.replace('_', ' ')}: {_v:,}")
        with _db:
            if _di.get("known_gaps"):
                st.markdown("**Known gaps:**")
                for _g in _di["known_gaps"]:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;• {_g}")
            if _di.get("possible_biases"):
                st.markdown("**Possible biases:**")
                for _b in _di["possible_biases"]:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;• {_b}")

    # ── Dataset & EDA ───────────────────────────────────────────────────────
    with tab_eda:
        st.header("Dataset & EDA — D-Fire")

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
                color_discrete_map={
                    "fire_only": "#e05c00",
                    "smoke_only": "#7b9ccc",
                    "fire_and_smoke": "#c44b00",
                    "background": "#888888",
                },
                labels={"category": "Category", "count": "Images"},
                title="D-Fire: Image distribution by category",
            )
            fig_cat.update_layout(showlegend=False)
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_b:
            st.subheader("Bounding boxes per image")
            fig_hist = px.histogram(
                df[df["total_boxes"] > 0],
                x="total_boxes",
                nbins=30,
                color_discrete_sequence=["#e05c00"],
                labels={"total_boxes": "Total boxes", "count": "Images"},
                title="Distribution of bounding box count (images with detections only)",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Row 2: split distribution + split × category balance ────────────
        col_c, col_d = st.columns(2)

        _CAT_COLORS = {
            "fire_only": "#e05c00",
            "smoke_only": "#7b9ccc",
            "fire_and_smoke": "#c44b00",
            "background": "#888888",
        }

        with col_c:
            st.subheader("Split distribution")
            split_df = compute_split_counts(df)
            fig_split = px.bar(
                split_df,
                x="split",
                y="count",
                color="split",
                labels={"split": "Split", "count": "Images"},
                title="D-Fire: Images per split",
            )
            fig_split.update_layout(showlegend=False)
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
                    color_discrete_map=_CAT_COLORS,
                    labels={"split": "Split", "count": "Images", "image_category": "Category"},
                    title="Category distribution within each split",
                )
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
                color_discrete_map=_CAT_COLORS,
                opacity=0.35,
                size_max=4,
                labels={"image_width": "Width (px)", "image_height": "Height (px)"},
                title="Width × Height (up to 5 000 points shown)",
            )
            fig_res.update_traces(marker={"size": 3})
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
                color_discrete_map={"fire": "#e05c00", "smoke": "#7b9ccc"},
                labels={"boxes": "Boxes per image", "count": "Images"},
                title="Box count distribution — fire vs smoke",
            )
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
                        color_discrete_map={"fire": "#e05c00", "smoke": "#7b9ccc"},
                        labels={"mean_bbox_area": "Mean normalized area (w×h)", "class": "Class"},
                        title="Per-image mean bbox area — fire vs smoke",
                    )
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
                        color_discrete_map={"fire": "#e05c00", "smoke": "#7b9ccc"},
                        labels={"coverage": "Summed normalized area", "class": "Class"},
                        title="Total class coverage per image (labeled images only)",
                    )
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
                        color_discrete_map=_CAT_COLORS,
                        labels={"mean_brightness": "Mean brightness (0–1)", "image_category": ""},
                        title="Mean brightness by category",
                    )
                    fig_br.update_layout(showlegend=False)
                    st.plotly_chart(fig_br, use_container_width=True)

                with col_j:
                    dk_df = compute_pixel_stats_by_category(df, "dark_pixel_ratio")
                    fig_dk = px.violin(
                        dk_df,
                        x="image_category",
                        y="dark_pixel_ratio",
                        color="image_category",
                        box=True,
                        color_discrete_map=_CAT_COLORS,
                        labels={"dark_pixel_ratio": "Dark pixel ratio (brightness < 0.118)", "image_category": ""},
                        title="Dark pixel ratio by category",
                    )
                    fig_dk.update_layout(showlegend=False)
                    st.plotly_chart(fig_dk, use_container_width=True)

                with col_k:
                    cs_df = compute_pixel_stats_by_category(df, "color_std_mean")
                    fig_cs = px.violin(
                        cs_df,
                        x="image_category",
                        y="color_std_mean",
                        color="image_category",
                        box=True,
                        color_discrete_map=_CAT_COLORS,
                        labels={"color_std_mean": "Color spread (mean channel std)", "image_category": ""},
                        title="Color spread (diversity) by category",
                    )
                    fig_cs.update_layout(showlegend=False)
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
                    color_discrete_map=_CAT_COLORS,
                    opacity=0.4,
                    title=f"{x_col} vs {y_col} (up to 5 000 points)",
                )
                fig_scatter.update_traces(marker={"size": 4})
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
                            use_column_width=True,
                        )
                    except Exception:
                        pass
            else:
                st.info("Raw image paths from the CSV are not accessible on this machine.")

    # ── Inference Demo ───────────────────────────────────────────────────────
    with tab_inference:
        st.header("Inference Demo")
        st.info(
            "**Coming in Lecture 6–7 (M3):** "
            "Upload an image or video, run YOLO11s inference, "
            "and display bounding-box overlays with fire/smoke class labels and confidence scores."
        )
        st.markdown(f"""
**Settings active:**
- Confidence threshold: `{confidence_threshold}`
- Confirmation frames: `{confirmation_frames}`
- Primary model: `{model_plan['primary_model']}`
        """)

        uploaded = st.file_uploader(
            "Upload image or video (placeholder — inference not yet wired)",
            type=["jpg", "jpeg", "png", "mp4", "avi"],
            disabled=False,
        )
        if uploaded:
            st.warning("Model not loaded yet. Inference will be wired in M3.")

    # ── Mapping Setup ────────────────────────────────────────────────────────
    with tab_mapping:
        st.header("Mapping Setup")
        st.info(
            "**Coming in M3:** "
            "Draw named polygons on camera images, link them to map areas, "
            "and configure camera GPS and metadata for approximate location output."
        )
        from src.mapping import get_mapping_modes
        st.markdown("**Supported mapping modes:**")
        for i, mode_name in enumerate(get_mapping_modes(), 1):
            st.markdown(f"{i}. {mode_name.capitalize()}")

        st.caption("All location outputs will be marked as approximate.")

    # ── Alert Log ────────────────────────────────────────────────────────────
    with tab_alerts:
        st.header("Alert Log")
        st.info(
            "**Coming in M3:** "
            "Confirmed fire/smoke alerts will appear here with camera ID, timestamp, "
            "apparent direction, approximate location, and status."
        )
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

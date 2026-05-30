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
from src.ui import apply_chart_theme, CAT_COLORS, PYRO_COLORS, SPLIT_COLORS, CLASS_COLORS

st.set_page_config(page_title="PyroFinder", layout="wide")

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
            "Operations & Learning Dashboard",
            "Central Control Dashboard",
            "M2 Course Dashboard",
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
        st.caption("System plan — model, metrics, and dataset summary")
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

    # ── Inference Demo ───────────────────────────────────────────────────────
    with tab_inference:
        st.info(
            "**Coming in Lecture 6–7 (M3):** "
            "Upload an image or video, run YOLO11s inference, "
            "and display bounding-box overlays with fire/smoke class labels and confidence scores."
        )
        st.progress(0, text="M3 progress: 0%")
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
    st.caption("Five narrative tabs covering the M2 deliverable requirements.")

    tab_problem, tab_lit, tab_market, tab_eda_story, tab_kpi = st.tabs([
        "1. Problem Understanding",
        "2. Literature Review",
        "3. Market Review",
        "4. Dataset & EDA",
        "5. KPI & Metrics",
    ])

    # ── Tab 1: Problem Understanding ─────────────────────────────────────────
    with tab_problem:
        st.header("Problem Understanding")
        st.caption(
            "The problem space, affected users, core use case, scope boundaries, "
            "and competitive differentiation."
        )

        # ── Problem statement ───────────────────────────────────────────────
        st.subheader("The Problem")
        st.markdown("""
Private property owners in fire-prone areas rely on existing security cameras, but these cameras
are **passive**: someone must actively watch them to notice smoke or fire.

During dry seasons, fires can start at property edges, agricultural fields, forest borders,
parking areas, or neighbouring land. **Delayed awareness can turn a small incident into a
dangerous event.**

Existing wildfire monitoring solutions typically require dedicated towers, acoustic sensors,
drones, or public-sector infrastructure. **None of them are built for the private landowner who
already has cameras installed.**
        """)

        m1, m2, m3 = st.columns(3)
        m1.metric("Root cause", "Cameras are passive")
        m2.metric("Trigger", "Dry-season fire risk")
        m3.metric("Gap", "No private-property solution")

        st.divider()

        # ── Primary persona ─────────────────────────────────────────────────
        st.subheader("Primary Persona — Dani")
        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.markdown("""
**Role:** Farm owner, central Israel
**Property:** 120-dunam agricultural farm
**Cameras:** Fixed outdoor cameras at boundary points
**Season of concern:** Dry summer months
**Risk sources:** Neighbouring fields, agricultural equipment
**Current behaviour:** Cannot continuously watch every camera feed
            """)
        with col_b:
            st.info(
                "Dani needs PyroFinder to monitor camera feeds automatically and send an alert "
                "the moment fire or smoke is confirmed — without requiring Dani to watch screens "
                "all day. Time from ignition to awareness is the critical metric."
            )

        st.divider()

        # ── Use-case flow ───────────────────────────────────────────────────
        st.subheader("Main Use Case — Step by Step")
        steps = [
            ("1. Ignition", "Fire starts at a property edge or neighbouring field."),
            ("2. Detection", "PyroFinder detects smoke or fire in the camera frame using YOLO11s."),
            ("3. Confirmation", "Detection is confirmed across N consecutive frames (configurable) to filter single-frame noise."),
            ("4. Alert", "Owner receives an alert within seconds — camera ID, timestamp, approximate location, and apparent direction."),
            ("5. Response", "Owner can act immediately: evacuate, call emergency services, or mark as false alarm from the dashboard."),
        ]
        for title, body in steps:
            with st.expander(title, expanded=True):
                st.write(body)

        st.divider()

        # ── Target audience ─────────────────────────────────────────────────
        st.subheader("Target Audience")
        col_p, col_s = st.columns(2)
        with col_p:
            st.markdown("**Primary — paying customers**")
            for seg in [
                "Homeowners in fire-prone areas",
                "Farm and ranch owners",
                "Agricultural facility managers",
                "Private landowners near wildland-urban interface",
            ]:
                st.markdown(f"- {seg}")
        with col_s:
            st.markdown("**Secondary — alert recipients**")
            for seg in [
                "Municipalities",
                "Emergency response teams",
                "Forest and park authorities",
                "PyroFinder internal team (Operations & Learning Dashboard)",
            ]:
                st.markdown(f"- {seg}")

        st.divider()

        # ── Scope boundaries ────────────────────────────────────────────────
        st.subheader("Scope Boundaries — What PyroFinder Is Not")
        not_list = [
            ("Early warning system", "PyroFinder does not predict fire spread; it detects and alerts on confirmed events."),
            ("Fire spread predictor", "No true physical fire-spread simulation or prediction in the MVP."),
            ("New hardware product", "No dedicated towers, acoustic sensors, or drones required."),
            ("Emergency dispatch integration", "Alerts reach the property owner; integration with rescue services is a future feature."),
            ("Precise GPS tracker", "Location outputs are approximate, based on camera metadata. Never claimed as precise."),
            ("Pure YOLO demo", "PyroFinder is a full monitoring system built on top of detection results, not a model showcase."),
        ]
        for label, explanation in not_list:
            st.markdown(f"**{label}:** {explanation}")

        st.divider()

        # ── Competitive landscape ────────────────────────────────────────────
        st.subheader("Competitive Landscape")

        comp_data = {
            "Solution": ["PyroFinder", "Pano AI", "FIREWAVE", "CANDO"],
            "Targets private landowners": ["Yes", "No", "No", "No"],
            "Uses existing cameras": ["Yes", "No", "No", "No"],
            "Requires new dedicated hardware": ["No", "Yes (towers)", "Yes (acoustic sensors)", "Yes (drones)"],
            "Fire/smoke detection": ["Yes — YOLO11s", "Yes — panoramic AI", "No (acoustic)", "Yes — aerial"],
            "Approximate GPS alerting": ["Yes", "Yes", "No", "Yes"],
            "Operates without public-sector infra": ["Yes", "No", "No", "Partial"],
        }
        comp_df = pd.DataFrame(comp_data).set_index("Solution")
        st.dataframe(comp_df, use_container_width=True)

        st.caption(
            "Sources: Pano AI (panoramic camera towers for public land), "
            "FIREWAVE (acoustic forest-fire detection), "
            "CANDO (autonomous drone security and monitoring)."
        )

        st.divider()

        # ── Key risks ───────────────────────────────────────────────────────
        st.subheader("Main Risks")
        risks = [
            ("Dataset domain gap", "D-Fire and supplementary datasets may not match real private-property camera angles, lighting, or fire scenarios."),
            ("False alarms", "Reflections, sunsets, headlights, fog, or dust may trigger false detections."),
            ("Inference speed", "YOLO11s may be too slow for near-real-time sampled-frame inference on available MVP hardware."),
            ("Poor camera calibration", "Missing or incorrect camera height/azimuth leads to inaccurate location estimates."),
            ("Manual mapping errors", "Operators may draw incorrect polygons or enter wrong GPS coordinates during setup."),
            ("Adverse conditions", "Detection accuracy degrades at night, under heavy smoke, or in rain and fog."),
        ]
        risk_df = pd.DataFrame(risks, columns=["Risk", "Description"])
        risk_df.index = risk_df.index + 1
        st.table(risk_df)
        st.caption("Source: PROJECT_CONTEXT.md §2, §4, §5, §16")

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
            "PyroFinder addresses this question directly by applying YOLO11s fire/smoke object "
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
                    "Benchmark YOLO11s against YOLO11n and document whether the main model is "
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
                "**Lesson for PyroFinder:** Benchmark YOLO11s against YOLO11n and document "
                "whether the main model is fast enough for near-real-time sampled-frame monitoring."
            )

        st.divider()

        # ── Section 4: Research Gap ───────────────────────────────────────────
        st.subheader("Research Gap")
        st.info(
            "**Research Gap**\n\n"
            "Across these five papers, the key unresolved issue is transfer from benchmark "
            "datasets to real civilian or security-camera environments. Although deep learning "
            "and YOLO-based object detection have significantly improved fire and wildfire "
            "detection, the literature still shows limited real-world testing, dataset bias, "
            "weak generalization across scenes, and insufficient evaluation of false alarms in "
            "ordinary camera settings.\n\n"
            "This leaves room for PyroFinder: a surveillance-focused system trained and "
            "evaluated for ordinary RGB camera feeds, with explicit model metrics, multi-frame "
            "confirmation, false-alarm review, and approximate location output."
        )

        st.divider()

        # ── Section 5: Practical Implications ────────────────────────────────
        st.subheader("Practical Implications for PyroFinder")
        st.markdown("""
1. **Use object detection, not classification only.** PyroFinder needs bounding boxes because alerts require both detection and approximate location context.
2. **Keep the class schema simple.** The project should detect only `fire` and `smoke`; other objects can be used as background negatives but not as detection targets in the MVP.
3. **Measure both accuracy and operational performance.** The dashboard should report mAP@0.5, precision, recall, F1-score, false alarm rate, and inference speed.
4. **Handle false alarms explicitly.** Multi-frame confirmation, threshold tuning, and false-positive review are required because clouds, haze, glare, fog, dust, and lighting changes can resemble smoke or fire.
5. **Validate domain transfer.** D-Fire and supplementary datasets may not fully represent private-property cameras; PyroFinder should validate on additional images/videos and document known gaps.
6. **Compare YOLO11s and YOLO11n.** YOLO11s is the main model, but YOLO11n should be used as a speed baseline/fallback to understand the accuracy/speed tradeoff.
        """)

    # ── Tab 3: Market Review ──────────────────────────────────────────────────
    with tab_market:
        st.info("**Coming soon** — addressable market size, competitive pricing, and go-to-market strategy for private-property fire monitoring.")

    # ── Tab 4: Dataset & EDA (story version) ─────────────────────────────────
    with tab_eda_story:
        st.header("Dataset & EDA — D-Fire")

        with st.container(border=True):
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
            "with findings and implications for YOLO11s training and evaluation."
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
                    st.info(
                        "Background images dominate the 0-box bin — they carry no annotations "
                        "by definition. Fire-and-smoke images account for most high-box-count "
                        "images; the right tail represents complex multi-object scenes valuable "
                        "for training YOLO11s multi-object handling."
                    )
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
                    st.info(
                        "Fire-only images average ~67% dark pixels (night scenes) while "
                        "smoke-only images average ~17% (bright daytime). This lighting gap "
                        "means YOLO11s may under-perform on fire in daylight and smoke at "
                        "night; augmentation targeting both extremes is a priority for M3."
                    )
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
                    st.info(
                        "Smoke bounding boxes are ~7× larger than fire boxes in normalised "
                        "area — plumes cover far more of the frame than visible flame. "
                        "YOLO11s must handle both large smoke blobs and small fire targets; "
                        "small-object recall for fire should be verified in M3 evaluation."
                    )
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
                    st.info(
                        "Fire centroids cluster in the lower-centre of the frame — consistent "
                        "with ground-level fires captured by outdoor cameras. Smoke centroids "
                        "drift toward the upper frame as smoke rises, confirmed by the "
                        "thirds-grid bottom-centre concentration for fire."
                    )

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
                    st.info(
                        "Strongest non-trivial correlation: dark_pixel_ratio ↔ fire_bbox_coverage "
                        "(fire-only images are predominantly dark night scenes). Near-zero "
                        "correlation between fire and smoke coverage supports separate confidence "
                        "threshold calibration for YOLO11s fire and smoke heads."
                    )

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
                    st.info(
                        "This chart verifies that target-image categories are represented across "
                        "the train/test split. It helps detect split imbalance before YOLO11s "
                        "training and evaluation."
                    )
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

    # ── Tab 5: KPI & Metrics ──────────────────────────────────────────────────
    with tab_kpi:
        st.caption(
            "Formal ML metrics defined in the problem specification. "
            "Values will be populated after M3 model training on D-Fire."
        )

        _KPI_METRICS = [
            {
                "name": "mAP@0.5",
                "interpretation": (
                    "Mean Average Precision at IoU threshold 0.5 — "
                    "primary detection quality metric; measures how well the model "
                    "localises and classifies fire/smoke boxes at a lenient overlap threshold."
                ),
            },
            {
                "name": "mAP@0.5:0.95",
                "interpretation": (
                    "Mean Average Precision averaged over IoU thresholds 0.5–0.95 — "
                    "stricter localisation quality; penalises imprecise bounding boxes "
                    "even when the class is correctly identified."
                ),
            },
            {
                "name": "Precision",
                "interpretation": (
                    "Fraction of fire/smoke detections that are correct — "
                    "high precision means fewer false alarms reaching the operator."
                ),
            },
            {
                "name": "Recall",
                "interpretation": (
                    "Fraction of actual fire/smoke events that are detected — "
                    "the critical safety metric; a missed fire is more dangerous "
                    "than a false alarm."
                ),
            },
            {
                "name": "F1-Score",
                "interpretation": (
                    "Harmonic mean of Precision and Recall — "
                    "balances the trade-off between missing fires and raising false alarms."
                ),
            },
            {
                "name": "False Alarm Rate",
                "interpretation": (
                    "False positives per hour (or per 1 000 sampled frames) — "
                    "operational KPI; excessive false alarms erode operator trust "
                    "and cause alert fatigue."
                ),
            },
            {
                "name": "Inference Speed",
                "interpretation": (
                    "Frames per second (FPS) or milliseconds per frame — "
                    "determines whether YOLO11s meets near-real-time requirements "
                    "on available MVP hardware; YOLO11n is the speed baseline."
                ),
            },
        ]

        # Two-column grid — pairs of cards, last card centred if count is odd
        for row_start in range(0, len(_KPI_METRICS), 2):
            pair = _KPI_METRICS[row_start : row_start + 2]
            cols = st.columns(2)
            for col, kpi in zip(cols, pair):
                with col:
                    with st.container(border=True):
                        st.metric(
                            label=kpi["name"],
                            value="N/A — awaits M3 training",
                        )
                        st.caption(kpi["interpretation"])

        st.divider()
        st.subheader("Metric definitions (from formal ML problem)")
        st.markdown("""
| Metric | Type | Goal |
|--------|------|------|
| mAP@0.5 | Detection quality | Higher is better |
| mAP@0.5:0.95 | Detection quality (strict) | Higher is better |
| Precision | Per-class | Higher → fewer false alarms |
| Recall | Per-class | Higher → fewer missed fires |
| F1-Score | Per-class | Higher → balanced detector |
| False Alarm Rate | Operational | Lower is better |
| Inference Speed | Operational | Higher FPS → more real-time |
        """)
        st.caption(
            "Split strategy: D-Fire train/val/test split if provided by the dataset, "
            "otherwise a reproducible 70 / 15 / 15 stratified split by image category. "
            "Primary model: YOLO11s. Speed baseline: YOLO11n."
        )
        st.caption("Source: PROJECT_CONTEXT.md §8")

"""Operations & Learning Dashboard renderer.

Moved out of app.py during the M3 dashboard refactor. The dashboard body is
unchanged except that the Models tab now calls the shared rendering helpers in
``model_helpers`` instead of defining them inline. Heavy ML libraries stay lazy
(imported inside ``src.inference`` and inside the inference-demo code path),
never at module import time.
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

from src.dashboards import model_helpers as mh

METADATA_PATH = "data/dfire_metadata.csv"
SAMPLES_DIR = Path("data/samples/dfire/images")
GENERATE_CMD = (
    "python scripts/build_dfire_metadata.py "
    '--raw-root "<path-to-D-Fire-root>" '
    "--output data/dfire_metadata.csv"
)


def render(confidence_threshold, confirmation_frames):
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
        st.header("Models")
        st.caption(
            "Image-level classifiers (color features), the YOLO11n object-detection "
            "baseline / fallback, and the YOLO11s current primary detector · "
            "D-Fire dataset · fire / smoke / background"
        )
        _results_data = mh.load_model_results()
        if not _results_data:
            st.warning(
                "No model result files found in `results/`. "
                "Run `python scripts/dummy_try.py` to generate baseline results."
            )
        else:
            mh.render_models_section(_results_data, include_comparison=True)

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
                                _model = _inf.load_detector_cached(_mname)
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

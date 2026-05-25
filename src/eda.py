"""EDA helper functions for PyroFinder D-Fire metadata analysis."""

from __future__ import annotations

import pandas as pd


def compute_summary_metrics(df: pd.DataFrame) -> dict:
    """Return high-level summary metrics from the metadata DataFrame."""
    cats = df["image_category"].value_counts() if "image_category" in df.columns else pd.Series(dtype=int)
    return {
        "total_images": len(df),
        "fire_images": int(cats.get("fire_only", 0) + cats.get("fire_and_smoke", 0)),
        "smoke_images": int(cats.get("smoke_only", 0) + cats.get("fire_and_smoke", 0)),
        "background_images": int(cats.get("background", 0)),
        "fire_and_smoke_images": int(cats.get("fire_and_smoke", 0)),
        "mean_boxes_per_image": float(df["total_boxes"].mean()) if "total_boxes" in df.columns else 0.0,
        "median_boxes_per_image": float(df["total_boxes"].median()) if "total_boxes" in df.columns else 0.0,
    }


def compute_category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with image counts per category."""
    counts = (
        df["image_category"]
        .value_counts()
        .reset_index()
        .rename(columns={"image_category": "category", "count": "count"})
    )
    # Ensure consistent ordering
    order = ["fire_only", "smoke_only", "fire_and_smoke", "background"]
    counts["category"] = pd.Categorical(counts["category"], categories=order, ordered=True)
    counts = counts.sort_values("category").reset_index(drop=True)
    return counts


def compute_split_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with image counts per split."""
    return (
        df["split"]
        .value_counts()
        .reset_index()
        .rename(columns={"split": "split", "count": "count"})
    )


def compute_bbox_stats(df: pd.DataFrame) -> dict:
    """Return aggregate bounding box statistics from the metadata."""
    labeled = df[df["total_boxes"] > 0] if "total_boxes" in df.columns else df
    if labeled.empty:
        return {"mean_area": 0.0, "median_area": 0.0, "max_area": 0.0}
    return {
        "mean_area": float(labeled["mean_bbox_area"].mean()),
        "median_area": float(labeled["median_bbox_area"].median()),
        "max_area": float(labeled["max_bbox_area"].max()),
        "mean_aspect_ratio": float(labeled["mean_bbox_aspect_ratio"].mean()),
    }


def filter_metadata(
    df: pd.DataFrame,
    categories: list[str] | None = None,
    splits: list[str] | None = None,
    has_fire: bool | None = None,
    has_smoke: bool | None = None,
) -> pd.DataFrame:
    """Filter metadata DataFrame by category, split, and detection flags."""
    mask = pd.Series(True, index=df.index)
    if categories:
        mask &= df["image_category"].isin(categories)
    if splits:
        mask &= df["split"].isin(splits)
    if has_fire is not None:
        mask &= df["has_fire"] == has_fire
    if has_smoke is not None:
        mask &= df["has_smoke"] == has_smoke
    return df[mask].reset_index(drop=True)


_NUMERIC_COLS_FOR_CORR = [
    "num_fire_boxes", "num_smoke_boxes", "total_boxes",
    "mean_bbox_area", "max_bbox_area", "mean_bbox_aspect_ratio",
    "fire_mean_bbox_area", "smoke_mean_bbox_area",
    "fire_bbox_coverage", "smoke_bbox_coverage",
    "image_width", "image_height",
    "mean_brightness", "dark_pixel_ratio", "color_std_mean",
    "fire_mean_x_center", "fire_mean_y_center",
    "smoke_mean_x_center", "smoke_mean_y_center",
    "smoke_dy_vs_fire", "smoke_dx_vs_fire", "fire_smoke_mean_iou",
]


def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return the subset of correlation-candidate columns present in df."""
    return [c for c in _NUMERIC_COLS_FOR_CORR if c in df.columns]


def compute_correlation_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Return a Pearson correlation matrix for the given numeric columns."""
    if cols is None:
        cols = get_numeric_cols(df)
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame()
    return df[available].corr(method="pearson", numeric_only=True)


def compute_spatial_centers(df: pd.DataFrame, cls: str) -> pd.DataFrame:
    """Return x/y center data for fire or smoke bboxes (for 2-D density plots).

    Only rows where the class is present (x_center is not NaN) are returned.
    y=0 is the top of the image; y=1 is the bottom.
    """
    x_col = f"{cls}_mean_x_center"
    y_col = f"{cls}_mean_y_center"
    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame(columns=["x_center", "y_center", "image_category"])
    sub = df[[x_col, y_col, "image_category"]].dropna(subset=[x_col, y_col]).copy()
    return sub.rename(columns={x_col: "x_center", y_col: "y_center"})


def compute_grid_distribution(df: pd.DataFrame, cls: str) -> pd.DataFrame:
    """Return a 3×3 pivot table of bbox thirds-grid cell counts for fire or smoke.

    Index = row third (1=top, 2=middle, 3=bottom).
    Columns = column third (1=left, 2=center, 3=right).
    Rows where class is absent (thirds == 0) are excluded.
    """
    col_col = f"{cls}_thirds_col"
    row_col = f"{cls}_thirds_row"
    if col_col not in df.columns or row_col not in df.columns:
        return pd.DataFrame()
    sub = df[(df[col_col] > 0) & (df[row_col] > 0)][[row_col, col_col]]
    if sub.empty:
        return pd.DataFrame()
    pivot = (
        sub.groupby([row_col, col_col])
        .size()
        .reset_index(name="count")
        .pivot(index=row_col, columns=col_col, values="count")
        .fillna(0)
        .astype(int)
    )
    pivot.index.name = "row_third"
    pivot.columns.name = "col_third"
    return pivot


def compute_split_category_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form split × category counts for a stacked bar chart."""
    if df.empty or "split" not in df.columns or "image_category" not in df.columns:
        return pd.DataFrame(columns=["split", "image_category", "count"])
    return (
        df.groupby(["split", "image_category"])
        .size()
        .reset_index(name="count")
    )


def compute_class_bbox_areas(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-image fire/smoke mean bbox area in long form (for violin/box plots).

    Only includes rows where the class actually appears (has_fire / has_smoke == True).
    """
    parts = []
    if "fire_mean_bbox_area" in df.columns and "has_fire" in df.columns:
        sub = df[df["has_fire"]][["image_category", "fire_mean_bbox_area"]].copy()
        sub = sub.rename(columns={"fire_mean_bbox_area": "mean_bbox_area"})
        sub["class"] = "fire"
        parts.append(sub)
    if "smoke_mean_bbox_area" in df.columns and "has_smoke" in df.columns:
        sub = df[df["has_smoke"]][["image_category", "smoke_mean_bbox_area"]].copy()
        sub = sub.rename(columns={"smoke_mean_bbox_area": "mean_bbox_area"})
        sub["class"] = "smoke"
        parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=["image_category", "mean_bbox_area", "class"])
    return pd.concat(parts, ignore_index=True)


def compute_pixel_stats_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return two-column DataFrame of a pixel stat vs image_category for violin plots."""
    if col not in df.columns or "image_category" not in df.columns:
        return pd.DataFrame(columns=["image_category", col])
    return df[["image_category", col]].dropna().copy()


def get_primary_eda_insight(df: pd.DataFrame) -> str:
    """Return a data-driven EDA insight string from the metadata."""
    if df.empty:
        return "No data loaded."

    cats = df["image_category"].value_counts()
    bg = cats.get("background", 0)
    total = len(df)
    bg_pct = 100 * bg / total if total > 0 else 0

    fire_only = cats.get("fire_only", 0)
    smoke_only = cats.get("smoke_only", 0)
    fire_smoke = cats.get("fire_and_smoke", 0)

    # Split-specific note
    splits = df["split"].value_counts()
    no_label_splits = df[~df["has_label"]]["split"].value_counts()

    unlabeled_note = ""
    if not no_label_splits.empty:
        unlabeled_split = no_label_splits.index[0]
        unlabeled_count = int(no_label_splits.iloc[0])
        unlabeled_note = (
            f" Note: the '{unlabeled_split}' split ({unlabeled_count} images) "
            "has no label files in this D-Fire download; those images are counted as background."
        )

    # Pixel stat findings (if available)
    pixel_note = ""
    if "dark_pixel_ratio" in df.columns and "image_category" in df.columns:
        fire_dark = df[df["image_category"] == "fire_only"]["dark_pixel_ratio"].mean()
        smoke_dark = df[df["image_category"] == "smoke_only"]["dark_pixel_ratio"].mean()
        if not (pd.isna(fire_dark) or pd.isna(smoke_dark)):
            pixel_note = (
                f" Pixel analysis (derived from stored JPEG, not raw sensor data) reveals "
                f"a strong scene-type split: fire_only images have {fire_dark:.0%} dark pixels "
                f"on average — most are night or low-light scenes where fire is the light source. "
                f"Smoke_only images have only {smoke_dark:.0%} dark pixels — predominantly "
                f"bright daylight scenes with smoke against an open sky. "
                f"This brightness gap implies that a model fine-tuned on D-Fire may struggle "
                f"with fire in bright daytime scenes and smoke in dark conditions."
            )

    # Bbox size finding (if available)
    size_note = ""
    if "fire_mean_bbox_area" in df.columns and "smoke_mean_bbox_area" in df.columns:
        f_area = df[df["has_fire"]]["fire_mean_bbox_area"].mean()
        s_area = df[df["has_smoke"]]["smoke_mean_bbox_area"].mean()
        if not (pd.isna(f_area) or pd.isna(s_area)) and f_area > 0:
            ratio = s_area / f_area
            size_note = (
                f" Smoke bounding boxes are on average {ratio:.1f}× larger than fire bounding "
                f"boxes, reflecting that smoke plumes cover much larger image areas than visible "
                f"flame regions."
            )

    # Spatial finding (if available)
    spatial_note = ""
    if "smoke_dy_vs_fire" in df.columns:
        pct_above = (df["smoke_dy_vs_fire"] < 0).mean()
        if not pd.isna(pct_above):
            spatial_note = (
                f" In fire+smoke images, smoke appears above the fire centre "
                f"in {pct_above:.0%} of cases (smoke_dy < 0), consistent with "
                f"thermal convection carrying smoke upward."
            )

    insight = (
        f"The dataset ({total:,} images) is heavily skewed: "
        f"{bg_pct:.0f}% are background images ({bg:,}), "
        f"while fire-only={fire_only:,}, smoke-only={smoke_only:,}, "
        f"fire+smoke={fire_smoke:,}. "
        "Future model evaluation should pay close attention to fire recall and false negatives, "
        "since background dominates and a naive classifier could achieve high accuracy by "
        f"predicting background for every image.{unlabeled_note}"
        f"{pixel_note}{size_note}{spatial_note}"
    )
    return insight

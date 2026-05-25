"""Dataset loading and inspection utilities for PyroFinder.

Large dataset files live outside Git (see data/.gitkeep).
These helpers return metadata and schema information without downloading anything.
The generated metadata CSV (data/dfire_metadata.csv) is tracked by Git.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Generated metadata helpers ─────────────────────────────────────────────────

_REQUIRED_COLUMNS = {
    "image_id", "split", "image_path", "label_path", "has_label",
    "has_fire", "has_smoke", "image_category",
    "num_fire_boxes", "num_smoke_boxes", "total_boxes",
    "mean_bbox_area", "median_bbox_area", "max_bbox_area",
    "mean_bbox_aspect_ratio", "image_width", "image_height",
    "source_dataset",
}

_FLOAT_COLS = [
    "mean_bbox_area", "median_bbox_area", "max_bbox_area", "mean_bbox_aspect_ratio",
    "fire_mean_bbox_area", "smoke_mean_bbox_area", "fire_bbox_coverage", "smoke_bbox_coverage",
    "mean_brightness", "dark_pixel_ratio", "color_std_mean",
]
_INT_COLS = [
    "num_fire_boxes", "num_smoke_boxes", "total_boxes", "image_width", "image_height",
    "fire_thirds_col", "fire_thirds_row", "smoke_thirds_col", "smoke_thirds_row",
]
_BOOL_COLS = ["has_label", "has_fire", "has_smoke"]


def load_dfire_metadata(path: str | Path = "data/dfire_metadata.csv") -> pd.DataFrame:
    """Load the generated D-Fire metadata CSV.

    Raises FileNotFoundError if the CSV has not been generated yet.
    Run scripts/build_dfire_metadata.py to generate it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {path}\n"
            "Generate it with:\n"
            "  python scripts/build_dfire_metadata.py "
            '--raw-root "C:\\Users\\boris.azarov\\OneDrive - Technion\\Desktop\\PyroFinder\\RAW_DATA\\D-Fire" '
            "--output data/dfire_metadata.csv"
        )
    df = pd.read_csv(path, low_memory=False)
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")
    return df


def clean_dfire_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy of the metadata DataFrame with correct dtypes."""
    df = df.copy()

    for c in _FLOAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in _INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in _BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    if "image_category" in df.columns:
        df["image_category"] = df["image_category"].astype(str)

    if "split" in df.columns:
        df["split"] = df["split"].astype(str)

    # Normalize valid -> val
    if "split" in df.columns:
        df["split"] = df["split"].replace("valid", "val")

    # Remove duplicate image_id rows
    if "image_id" in df.columns:
        df = df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    return df


def get_dfire_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict from a cleaned metadata DataFrame."""
    return {
        "total_images": len(df),
        "splits": df["split"].value_counts().to_dict() if "split" in df.columns else {},
        "categories": df["image_category"].value_counts().to_dict() if "image_category" in df.columns else {},
        "total_fire_boxes": int(df["num_fire_boxes"].sum()) if "num_fire_boxes" in df.columns else 0,
        "total_smoke_boxes": int(df["num_smoke_boxes"].sum()) if "num_smoke_boxes" in df.columns else 0,
        "missing_labels": int((~df["has_label"]).sum()) if "has_label" in df.columns else 0,
        "mean_boxes_per_image": float(df["total_boxes"].mean()) if "total_boxes" in df.columns else 0.0,
    }




def get_primary_dataset_info() -> dict:
    """Return D-Fire dataset metadata as defined in PROJECT_CONTEXT.md."""
    return {
        "name": "D-Fire Dataset",
        "url": "https://github.com/gaia-solutions-on-demand/DFireDataset",
        "role": "primary training and held-out test evaluation",
        "num_images": 21527,
        "breakdown": {
            "fire_only": 1164,
            "smoke_only": 5867,
            "fire_and_smoke": 4658,
            "background": 9838,
        },
        "bounding_boxes": {
            "fire": 14692,
            "smoke": 11865,
        },
        "classes": ["fire", "smoke"],
        "annotation_format": "YOLO-format normalized coordinates",
        "license": "CC0 1.0 Universal",
        "known_gaps": [
            "limited night scenes",
            "limited indoor fires",
            "limited close-range agricultural fires",
        ],
        "possible_biases": [
            "skews toward outdoor wildland fires",
            "private-property camera angles may differ from dataset",
        ],
    }


def get_supplementary_datasets() -> list[dict]:
    """Return metadata for supplementary and validation datasets."""
    return [
        {
            "name": "Smart Fire System Dataset",
            "url": "https://github.com/mehmoodulhaq570/Smart-Fire-System-Yolov11n",
            "role": "supplementary training and external validation",
            "note": "Use dataset only — do not use repo code or trained model",
        },
        {
            "name": "Aerial Rescue Object Detection",
            "url": "https://www.kaggle.com/datasets/julienmeine/rescue-object-detection",
            "role": "robustness validation",
            "note": "Use Fire class for eval; Vehicle/Human as background negatives",
        },
        {
            "name": "Fire Detection in YOLO Format",
            "url": "https://www.kaggle.com/datasets/ankan1998/fire-detection-in-yolo-format",
            "role": "supplementary training after class compatibility verification",
            "note": "Small dataset — verify fire/smoke coverage before use",
        },
        {
            "name": "FURG Fire Dataset",
            "url": "https://github.com/steffensbola/furg-fire-dataset",
            "role": "video validation for temporal behavior and multi-frame tracking",
            "note": "24 videos; XML annotations; conversion to YOLO format required",
        },
    ]


def list_expected_dataset_columns() -> list[str]:
    """Return the expected fields for a dataset record in the system schema."""
    return [
        "dataset_id",
        "name",
        "source_url",
        "num_images",
        "classes",
        "split_info",
        "license",
        "role",
    ]


VALID_CLASSES = {"fire", "smoke"}

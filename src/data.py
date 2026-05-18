"""Dataset loading and inspection utilities for PyroFinder.

Large dataset files live outside Git (see data/.gitkeep).
These helpers return metadata and schema information without downloading anything.
"""

from __future__ import annotations


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

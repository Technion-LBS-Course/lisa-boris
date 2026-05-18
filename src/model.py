"""Model metadata and evaluation planning helpers for PyroFinder.

Does NOT load Ultralytics or require model weights to exist.
Model loading is implemented separately once weights are available.
"""

from __future__ import annotations


def get_model_plan() -> dict:
    """Return the planned model configuration as defined in PROJECT_CONTEXT.md."""
    return {
        "primary_model": "YOLO11s",
        "primary_weights": "yolo11s.pt",
        "primary_reason": (
            "Fast enough for near-real-time sampled-frame inference "
            "and more accurate than YOLO11n"
        ),
        "baseline_model": "YOLO11n",
        "baseline_weights": "yolo11n.pt",
        "baseline_reason": "Speed baseline and fallback — not an equal parallel model",
        "task": "two-class object detection",
        "classes": ["fire", "smoke"],
        "image_size": 640,
        "framework": "Ultralytics YOLO11",
        "fine_tuned_on": "D-Fire Dataset",
    }


def get_metrics_plan() -> list[str]:
    """Return the evaluation metrics planned for model comparison."""
    return [
        "mAP@0.5",
        "mAP@0.5:0.95",
        "Precision",
        "Recall",
        "F1-score",
        "False Alarm Rate (FP per hour or per 1,000 sampled frames)",
        "Inference speed (FPS or ms/frame)",
    ]


def get_split_strategy() -> dict:
    """Return the dataset split strategy."""
    return {
        "preferred": "Use D-Fire provided train/val/test split if available",
        "fallback": "Reproducible 70/15/15 stratified split by image category",
        "seed": 42,
    }

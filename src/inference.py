"""Lazy YOLO11n / YOLO11s inference helpers for the PyroFinder demo.

Heavy ML libraries (ultralytics, torch, numpy) are imported lazily inside the
functions that need them, never at module import time, so importing this module
is cheap and test-safe. Only the two fine-tuned D-Fire checkpoints are used:

    models/yolo11n_dfire_best.pt   — lightweight baseline / fallback
    models/yolo11s_dfire_best.pt   — current primary detector (measured; weights Git-ignored)

The demo never falls back to pretrained ``yolo11n.pt`` / ``yolo11s.pt``: the M3
inference demo must run on the fine-tuned fire/smoke detectors, so a missing
checkpoint raises ``FileNotFoundError`` instead of loading generic weights.

The detector is validated to expose only the two allowed classes, ``fire`` and
``smoke``. No model is loaded, trained, or downloaded at import time.
"""

from __future__ import annotations

from pathlib import Path

# The only two classes PyroFinder detects.
VALID_DETECTION_CLASSES = {"fire", "smoke"}

# Fine-tuned D-Fire checkpoints (local only; Git-ignored).
CHECKPOINTS: dict[str, Path] = {
    "YOLO11n": Path("models/yolo11n_dfire_best.pt"),
    "YOLO11s": Path("models/yolo11s_dfire_best.pt"),
}

# Shown when the local YOLO11s checkpoint file is absent (weights are Git-ignored,
# so a fresh clone has no checkpoint even though YOLO11s results are measured).
MISSING_YOLO11S_MESSAGE = (
    "YOLO11s checkpoint not found locally (model weights are Git-ignored). "
    "Add models/yolo11s_dfire_best.pt; if a Kaggle run is still in progress, "
    "add the checkpoint after it completes."
)


def checkpoint_path(model_name: str) -> Path:
    """Return the fine-tuned checkpoint path for a detector name.

    Raises ``KeyError`` for an unknown detector name.
    """
    try:
        return CHECKPOINTS[model_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown detector '{model_name}'. Known: {sorted(CHECKPOINTS)}"
        ) from exc


def checkpoint_exists(model_name: str) -> bool:
    """True when the fine-tuned checkpoint file for ``model_name`` is present."""
    return checkpoint_path(model_name).exists()


def available_detectors() -> list[str]:
    """Detector names whose fine-tuned checkpoint is present, in CHECKPOINTS order."""
    return [name for name in CHECKPOINTS if checkpoint_exists(name)]


def validate_detector_classes(class_names) -> bool:
    """True only when the detector exposes exactly the two PyroFinder classes.

    PyroFinder is a strict two-class object detector, so a valid fine-tuned
    checkpoint must expose exactly ``{"fire", "smoke"}`` (order-independent).
    Rejected: only ``fire``, only ``smoke``, an empty mapping, any extra class
    (e.g. ``background`` / ``person`` / ``vehicle``), and malformed class
    metadata. Accepts an Ultralytics ``names`` dict (``{0: "smoke", 1: "fire"}``)
    or any iterable of class names.
    """
    raw = class_names.values() if isinstance(class_names, dict) else class_names
    try:
        names = {str(n).strip().lower() for n in raw}
    except TypeError:
        # Non-iterable / malformed class metadata.
        return False
    return names == VALID_DETECTION_CLASSES


def _model_class_names(model) -> list[str]:
    """Extract the class-name list from a loaded Ultralytics model."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return list(names.values())
    if names is None:
        return []
    return list(names)


def load_detector(model_name: str):
    """Load a fine-tuned YOLO detector from its D-Fire checkpoint.

    Ultralytics is imported lazily here, never at module import time. Raises
    ``FileNotFoundError`` when the checkpoint is missing (never substitutes
    pretrained weights) and ``ValueError`` when the checkpoint exposes classes
    other than ``fire`` / ``smoke``.
    """
    path = checkpoint_path(model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found for {model_name}: {path}. "
            "Add the D-Fire checkpoint; pretrained weights are never substituted."
        )
    from ultralytics import YOLO  # lazy import

    model = YOLO(str(path))
    class_names = _model_class_names(model)
    if not validate_detector_classes(class_names):
        raise ValueError(
            f"{model_name} exposes classes {class_names}. "
            "Expected exactly the two PyroFinder classes: fire and smoke."
        )
    return model


def run_detection(model, pil_image, conf: float = 0.25, imgsz: int = 640) -> dict:
    """Run one-image inference and return an annotated overlay plus a summary.

    ``pil_image`` is a PIL image. Inference time is measured during this call —
    no estimated value is ever inserted.

    Returns::

        {
            "annotated_png":    PNG bytes with boxes/labels/confidence drawn,
            "fire_count":       number of fire detections,
            "smoke_count":      number of smoke detections,
            "total_detections": fire_count + smoke_count,
            "max_confidence":   highest detection confidence (float) or None,
            "inference_ms":     measured inference time in milliseconds,
        }
    """
    import io
    import time

    import numpy as np
    from PIL import Image

    rgb = pil_image.convert("RGB")
    start = time.perf_counter()
    results = model.predict(
        source=np.array(rgb), conf=conf, imgsz=imgsz, verbose=False
    )
    inference_ms = (time.perf_counter() - start) * 1000.0

    result = results[0]
    names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))

    fire_count = 0
    smoke_count = 0
    max_confidence = None
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        cls_ids = [int(c) for c in boxes.cls.tolist()]
        confidences = [float(c) for c in boxes.conf.tolist()]
        for cls_id, confidence in zip(cls_ids, confidences):
            label = str(names.get(cls_id, "")).strip().lower()
            if label == "fire":
                fire_count += 1
            elif label == "smoke":
                smoke_count += 1
            if max_confidence is None or confidence > max_confidence:
                max_confidence = confidence

    # result.plot() returns a BGR numpy array; convert to RGB for PNG output.
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    buf = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="PNG")

    return {
        "annotated_png": buf.getvalue(),
        "fire_count": fire_count,
        "smoke_count": smoke_count,
        "total_detections": fire_count + smoke_count,
        "max_confidence": max_confidence,
        "inference_ms": inference_ms,
    }

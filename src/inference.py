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

# Repository root (two levels up from this file: src/inference.py -> repo root).
# Checkpoint paths are resolved against this, not the current working directory,
# so inference works regardless of where the process is launched (e.g. Streamlit
# Community Cloud, where the CWD is not guaranteed to be the repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Fine-tuned D-Fire checkpoints. The two committed checkpoints ship in the repo
# (gitignore exceptions) so the public Streamlit app runs on a fresh clone.
CHECKPOINTS: dict[str, Path] = {
    "YOLO11n": _REPO_ROOT / "models" / "yolo11n_dfire_best.pt",
    "YOLO11s": _REPO_ROOT / "models" / "yolo11s_dfire_best.pt",
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

    from PIL import Image

    rgb = pil_image.convert("RGB")
    start = time.perf_counter()
    # Pass the PIL Image directly so Ultralytics handles BGR conversion internally.
    # Passing np.array(rgb) would give an RGB array treated as BGR, swapping R↔B.
    results = model.predict(
        source=rgb, conf=conf, imgsz=imgsz, verbose=False
    )
    inference_ms = (time.perf_counter() - start) * 1000.0

    result = results[0]
    names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))

    fire_count = 0
    smoke_count = 0
    max_confidence = None
    detections: list[dict] = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        cls_ids = [int(c) for c in boxes.cls.tolist()]
        confidences = [float(c) for c in boxes.conf.tolist()]
        # Normalized [x_center, y_center, w, h] per box, used for zone matching.
        xywhn = boxes.xywhn.tolist() if getattr(boxes, "xywhn", None) is not None else [None] * len(cls_ids)
        for cls_id, confidence, box in zip(cls_ids, confidences, xywhn):
            label = str(names.get(cls_id, "")).strip().lower()
            if label == "fire":
                fire_count += 1
            elif label == "smoke":
                smoke_count += 1
            if label in ("fire", "smoke") and box is not None:
                detections.append({
                    "class": label,
                    "confidence": confidence,
                    "bbox_norm": [float(v) for v in box],
                })
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
        "detections": detections,
    }


def top_hazard_detection(result: dict) -> dict | None:
    """Return the event-focus fire/smoke detection from a ``run_detection`` result.

    Selection priority: fire is always the event focus when present, regardless
    of confidence, since a fire is more urgent than smoke; ties within a class
    are broken by highest confidence. Smoke is only selected when no fire
    detection exists in the frame. Returns ``None`` when the result carries no
    fire/smoke detections. Pure — takes the result dict, so it is testable
    without loading a model.
    """
    detections = result.get("detections") or []
    if not detections:
        return None
    fires = [d for d in detections if d.get("class") == "fire"]
    pool = fires or detections
    return max(pool, key=lambda d: d.get("confidence", 0.0))


def select_confirmed_event_detection(window_results: list[dict | None]) -> dict | None:
    """Pick the incident-focus detection across an N-frame confirmation window.

    ``window_results`` holds per-frame ``run_detection`` results ordered oldest
    to newest, current frame last; an entry is ``None`` when that frame's
    detection was never computed. Fire always outranks smoke, extending
    :func:`top_hazard_detection`'s single-frame rule across the window: the
    current frame's own fire detection wins when present; otherwise the most
    recent fire detection from an earlier frame in the window wins; only when
    no frame in the window has fire does the current frame's smoke detection
    (or ``None``, if the current frame has no detections) win. This never
    weakens the single-frame fire-first rule in :func:`top_hazard_detection` —
    it only extends the same priority across multiple frames. Pure — testable
    without a loaded model.
    """
    if not window_results:
        return None
    current = window_results[-1]
    current_top = top_hazard_detection(current) if current else None
    if current_top is not None and current_top.get("class") == "fire":
        return current_top
    for result in reversed(window_results[:-1]):
        if result is None:
            continue
        top = top_hazard_detection(result)
        if top is not None and top.get("class") == "fire":
            return top
    return current_top


def bbox_bottom_center_norm(bbox_norm) -> tuple[float, float]:
    """Return the bottom-center anchor (x, y) in [0,1] for a normalized xywh box.

    Matches the project's approximate fire-location convention (``anchor_x =
    x_center``, ``anchor_y = y_center + height/2``), clamped to the frame.
    """
    x_center, y_center, _w, h = bbox_norm
    x = min(max(x_center, 0.0), 1.0)
    y = min(max(y_center + h / 2.0, 0.0), 1.0)
    return x, y

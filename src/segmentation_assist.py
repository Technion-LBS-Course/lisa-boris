"""Segmentation-assisted polygon refinement for Image Zone setup.

After a rough ROI box is available for a named image zone — from the Groq Vision
suggestion, a manually entered box, or the bounding box of a few clicked points —
this module refines that box into a cleaner polygon by running a *local* image
segmentation inside it.

Scope and separation of concerns:

* This is **setup tooling for Image Zones only** — it is NOT fire/smoke detection.
* The fire/smoke detector stays **YOLO11s / YOLO11n** (``src/inference.py``).
* Rough ROI boxes come from Groq Vision (``src/llm.detect_zone_boxes``) or from a
  manual box; this module only *refines* a box that already exists.
* Segmentation here **never calls Groq and never calls YOLO11s**. It uses a
  classical, box-prompted OpenCV GrabCut segmentation — no model weights, no
  download, no network, and nothing to commit.

Import safety (required):

* No heavy segmentation model is imported or loaded at module import time.
* No Streamlit dependency — this module is pure and unit-testable.
* ``cv2`` / ``numpy`` / ``PIL`` are imported lazily *inside* the functions that
  need them, so importing this module is cheap and always succeeds. The
  segmentation itself runs only when :func:`refine_box_to_mask` is called (i.e.
  when the operator clicks the button).

Schemas:

* **ROI box** — one internal dict schema ``{"x_min", "y_min", "x_max", "y_max"}``
  with normalized floats in ``[0, 1]``. :func:`box_norm_from_xyxy` adapts the
  ``[x0, y0, x1, y1]`` list schema that Groq Vision already produces.
* **Polygon** — the normalized vertex format used by the rest of Image Zones,
  ``[{"x": float, "y": float}, ...]``. :func:`polygon_to_pixel_vertices` converts
  it to the pixel ``[[x, y], ...]`` list the Central Control editor stores.
"""

from __future__ import annotations

# The classical, weight-free segmentation backend. Kept as a constant so the UI
# can describe it and tests can assert it without importing cv2.
SEG_BACKEND = "opencv-grabcut"

# A near-zero side means the box is degenerate (a point or a line) — not usable.
_MIN_BOX_SIDE = 1e-3

_BOX_KEYS = ("x_min", "y_min", "x_max", "y_max")


def _clamp01(value: float) -> float:
    """Clamp a float into the normalized ``[0, 1]`` range."""
    return min(max(float(value), 0.0), 1.0)


# ── ROI box: validation + adapters ────────────────────────────────────────────


def box_norm_from_xyxy(seq) -> dict:
    """Adapt a ``[x0, y0, x1, y1]`` sequence into the internal box dict.

    Does not validate/clamp — pass the result through :func:`validate_roi_box`
    before use. Raises ``ValueError`` if ``seq`` is not four numbers.
    """
    try:
        x0, y0, x1, y1 = (float(v) for v in seq)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected a 4-item [x0, y0, x1, y1] box.") from exc
    return {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}


def validate_roi_box(box_norm) -> dict:
    """Validate, clamp, and normalize a rough ROI box.

    Accepts either the internal ``{"x_min", "y_min", "x_max", "y_max"}`` dict or a
    ``[x0, y0, x1, y1]`` list/tuple (adapted automatically). Coordinates are
    coerced to floats, clamped to ``[0, 1]``, and reordered so ``x_min <= x_max``
    and ``y_min <= y_max``.

    Returns a fresh ``{"x_min", "y_min", "x_max", "y_max"}`` dict.

    Raises:
        ValueError: if the input is malformed or the box is degenerate
            (near-zero area). The UI guards against calling segmentation in that
            case, so this only fires on genuinely unusable input.
    """
    if isinstance(box_norm, (list, tuple)):
        if len(box_norm) != 4:
            raise ValueError("A list ROI box must have exactly 4 values [x0, y0, x1, y1].")
        box_norm = box_norm_from_xyxy(box_norm)
    if not isinstance(box_norm, dict):
        raise ValueError("ROI box must be a dict with x_min, y_min, x_max, y_max (or a 4-item list).")

    try:
        x_min = _clamp01(box_norm["x_min"])
        y_min = _clamp01(box_norm["y_min"])
        x_max = _clamp01(box_norm["x_max"])
        y_max = _clamp01(box_norm["y_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ROI box needs numeric x_min, y_min, x_max, y_max.") from exc

    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min

    if (x_max - x_min) < _MIN_BOX_SIDE or (y_max - y_min) < _MIN_BOX_SIDE:
        raise ValueError("ROI box is degenerate (near-zero area).")

    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def box_norm_to_xyxy(box_norm) -> list:
    """Return a validated box as a normalized ``[x0, y0, x1, y1]`` list."""
    box = validate_roi_box(box_norm)
    return [box["x_min"], box["y_min"], box["x_max"], box["y_max"]]


# ── Polygon helpers ───────────────────────────────────────────────────────────


def polygon_from_box_fallback(box_norm) -> list[dict]:
    """Return a four-point normalized polygon from a rough ROI box.

    Used when segmentation is unavailable, fails, or the operator prefers the
    plain rectangle. Vertices go clockwise from the top-left corner.
    """
    box = validate_roi_box(box_norm)
    return [
        {"x": box["x_min"], "y": box["y_min"]},
        {"x": box["x_max"], "y": box["y_min"]},
        {"x": box["x_max"], "y": box["y_max"]},
        {"x": box["x_min"], "y": box["y_max"]},
    ]


def polygon_to_pixel_vertices(polygon: list[dict], image_width: int, image_height: int) -> list:
    """Convert a normalized ``[{"x", "y"}, ...]`` polygon to pixel ``[[x, y], ...]``.

    This is the shape the Central Control editor stores in ``cc_pending_vertices``.
    """
    w = float(image_width)
    h = float(image_height)
    out: list = []
    for vertex in polygon or []:
        out.append([float(vertex["x"]) * w, float(vertex["y"]) * h])
    return out


def mask_to_polygon(
    mask,
    image_width: int,
    image_height: int,
    simplify_tolerance: float = 2.0,
) -> list[dict]:
    """Convert a binary mask into a simplified, normalized polygon.

    Traces the largest external contour of the mask and simplifies it with the
    Douglas–Peucker algorithm (``cv2.approxPolyDP``); ``simplify_tolerance`` is the
    epsilon in mask pixels. Returns normalized ``[{"x", "y"}, ...]`` vertices, or
    an empty list if the mask is empty or no usable contour is found.

    ``cv2`` / ``numpy`` are imported lazily so this module stays import-safe.
    """
    import numpy as np

    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return []
    binary = (arr > 0).astype("uint8")
    if int(binary.sum()) == 0:
        return []

    import cv2

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0.0:
        return []

    approx = cv2.approxPolyDP(largest, float(simplify_tolerance), True)
    points = approx.reshape(-1, 2)
    if len(points) < 3:
        # The simplification collapsed the shape — keep the raw contour instead.
        points = largest.reshape(-1, 2)
    if len(points) < 3:
        return []

    mask_h, mask_w = binary.shape[:2]
    w = float(image_width) if image_width else float(mask_w)
    h = float(image_height) if image_height else float(mask_h)
    return [{"x": _clamp01(px / w), "y": _clamp01(py / h)} for px, py in points]


# ── Backend availability + the main entry point ───────────────────────────────


def segmentation_backend_available() -> bool:
    """True if the local segmentation backend (OpenCV) can be imported.

    Uses ``importlib`` so it never actually imports ``cv2`` (and never loads a
    model). The UI calls this to show a controlled message when segmentation is
    unavailable, before offering the box-as-polygon fallback.
    """
    import importlib.util

    return importlib.util.find_spec("cv2") is not None


def _image_to_rgb_array(image):
    """Coerce a PIL image, raw image bytes, or an array into an HxWx3 uint8 array."""
    import numpy as np

    if hasattr(image, "convert"):  # PIL.Image.Image
        return np.asarray(image.convert("RGB"), dtype="uint8")
    if isinstance(image, (bytes, bytearray)):
        import io

        from PIL import Image as _Image

        pil = _Image.open(io.BytesIO(bytes(image))).convert("RGB")
        return np.asarray(pil, dtype="uint8")
    arr = np.asarray(image)
    if arr.ndim == 2:  # grayscale -> 3 channels
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("Unsupported image; expected a PIL image, bytes, or an HxWx3 array.")
    return arr[:, :, :3].astype("uint8")


def _fail(box: dict, backend: str, message: str) -> dict:
    """Build a well-formed failure result (never raises, no polygon)."""
    return {"ok": False, "backend": backend, "polygon": [], "box_norm": box, "message": message}


def refine_box_to_mask(
    image,
    box_norm,
    *,
    iter_count: int = 5,
    max_side: int = 768,
) -> dict:
    """Segment inside a rough ROI box and return a polygon candidate.

    Runs a local, box-prompted OpenCV GrabCut segmentation constrained to the
    selected box, then traces the foreground into a normalized polygon. This uses
    **no model weights, no network, no Groq, and no YOLO** — it is classical image
    segmentation and runs only when called (i.e. on an explicit button click).

    Args:
        image: A PIL image, raw image bytes, or an HxWx3 numpy array (the frame).
        box_norm: The rough ROI box (dict or ``[x0, y0, x1, y1]`` list); validated.
        iter_count: GrabCut iterations.
        max_side: The frame is downscaled so its longest side is at most this many
            pixels before segmentation (for speed); the polygon is normalized, so
            it maps back to any resolution.

    Returns:
        A result dict ``{"ok", "backend", "polygon", "box_norm", "message"}``.
        On success ``ok`` is True and ``polygon`` is a normalized
        ``[{"x", "y"}, ...]`` list. On any failure ``ok`` is False, ``polygon`` is
        empty, and ``message`` explains why (the caller offers the box fallback).

    Raises:
        ValueError: only if ``box_norm`` is malformed/degenerate. The UI validates
            the box first, so this does not fire in normal use.
    """
    box = validate_roi_box(box_norm)  # degenerate/malformed boxes are the caller's job to guard

    if not segmentation_backend_available():
        return _fail(
            box,
            "unavailable",
            "Local segmentation needs OpenCV (opencv-python-headless), which is not "
            "installed. Use the original box as a polygon instead.",
        )

    try:
        import numpy as np
        import cv2

        arr = _image_to_rgb_array(image)
        h, w = arr.shape[:2]
        if h < 2 or w < 2:
            return _fail(box, "error", "Frame is too small to segment.")

        # Downscale for speed; normalized output is resolution-independent.
        scale = min(1.0, max_side / float(max(h, w))) if max(h, w) else 1.0
        if scale < 1.0:
            small = cv2.resize(
                arr, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA
            )
        else:
            small = arr
        sh, sw = small.shape[:2]

        x0 = int(round(box["x_min"] * sw))
        y0 = int(round(box["y_min"] * sh))
        x1 = int(round(box["x_max"] * sw))
        y1 = int(round(box["y_max"] * sh))
        # Keep the rect strictly inside the frame with a positive size.
        x0 = min(max(x0, 0), sw - 2)
        y0 = min(max(y0, 0), sh - 2)
        x1 = min(max(x1, x0 + 1), sw - 1)
        y1 = min(max(y1, y0 + 1), sh - 1)
        rect = (x0, y0, x1 - x0, y1 - y0)

        mask = np.zeros((sh, sw), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, int(iter_count), cv2.GC_INIT_WITH_RECT)

        foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
        if int(foreground.sum()) == 0:
            return _fail(
                box,
                "empty-mask",
                "Segmentation found no distinct object inside the box. "
                "Use the original box as a polygon instead.",
            )

        polygon = mask_to_polygon(foreground, sw, sh, simplify_tolerance=2.0)
    except Exception as exc:  # noqa: BLE001 — controlled: never crash the UI
        return _fail(box, "error", f"Segmentation failed: {exc}")

    if len(polygon) < 3:
        return _fail(
            box,
            "empty-mask",
            "Segmentation did not produce a usable polygon. "
            "Use the original box as a polygon instead.",
        )

    return {
        "ok": True,
        "backend": SEG_BACKEND,
        "polygon": polygon,
        "box_norm": box,
        "message": "Segmentation polygon generated",
    }

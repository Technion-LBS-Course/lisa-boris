"""Cost-sensitive operational evaluation metrics for PyroFinder model comparison.

PyroFinder's most important failure is a MISSED HAZARD: ground-truth fire and/or
smoke is present, but the model raises no fire/smoke detection. At the alert level
this is a false negative, and it is treated as far more costly than a FALSE ALERT
(background ground truth, but the model predicts fire or smoke). Default cost
weights reflect this: false_negative_weight = 10, false_positive_weight = 1.

For alert-level evaluation, `fire` and `smoke` both count as "hazard":
    hazard_present  = ground truth has fire OR smoke
    hazard_detected = prediction  has fire OR smoke

Location metrics apply ONLY to object detectors (YOLO11n / YOLO11s), which produce
bounding boxes. Image-level sklearn classifiers have no boxes, so their location
metrics are N/A (None). Every location output is an APPROXIMATE image-space point,
never precise geolocation.

This module is intentionally dependency-light: it imports only the Python standard
library so it can be used from scripts, tests, and the Streamlit app without loading
torch, ultralytics, PIL, image files, or any dataset.

D-Fire class mapping (verified in CLAUDE.md): class 0 = smoke, class 1 = fire.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

# ── Alert-level hazard labels ──────────────────────────────────────────────────

HAZARD_LABELS = {"fire", "smoke"}

# Default operational cost weights: a missed hazard is 10x worse than a false alert.
DEFAULT_FN_WEIGHT = 10
DEFAULT_FP_WEIGHT = 1


def to_hazard_label(label: str) -> bool:
    """Map an image-level / detection class label to an alert-level hazard flag.

    `fire` and `smoke` are hazards; everything else (e.g. `background`, empty,
    ``None``) is not. Comparison is case-insensitive for robustness.
    """
    return str(label).strip().lower() in HAZARD_LABELS


def alert_outcome_from_flags(hazard_present: bool, hazard_detected: bool) -> str:
    """Single-case alert outcome from hazard flags: ``TP``/``FN``/``FP``/``TN``.

    ``FN`` is a missed hazard (the worst case); ``FP`` is a false alert.
    """
    if hazard_present and hazard_detected:
        return "TP"
    if hazard_present and not hazard_detected:
        return "FN"
    if (not hazard_present) and hazard_detected:
        return "FP"
    return "TN"


def alert_outcome(true_label: str, pred_label: str) -> str:
    """Alert outcome for one image from its true and predicted class labels."""
    return alert_outcome_from_flags(
        to_hazard_label(true_label), to_hazard_label(pred_label)
    )


# ── Alert-level confusion ──────────────────────────────────────────────────────

def compute_alert_confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """Confusion counts at the alert (hazard vs no-hazard) level.

    Each label is reduced to a hazard flag via :func:`to_hazard_label`:
        TP_alert: hazard present and detected
        FN_alert: hazard present but NOT detected  (missed hazard — worst case)
        FP_alert: no hazard but detected           (false alert)
        TN_alert: no hazard and not detected
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred length mismatch: {len(y_true)} vs {len(y_pred)}"
        )

    tp = fn = fp = tn = 0
    for true_label, pred_label in zip(y_true, y_pred):
        present = to_hazard_label(true_label)
        detected = to_hazard_label(pred_label)
        if present and detected:
            tp += 1
        elif present and not detected:
            fn += 1
        elif (not present) and detected:
            fp += 1
        else:
            tn += 1

    return {
        "tp_alert": tp,
        "fn_alert": fn,
        "fp_alert": fp,
        "tn_alert": tn,
        "total_hazard_cases": tp + fn,
        "total_background_cases": fp + tn,
    }


def alert_confusion_from_confusion_matrix(matrix, labels: Sequence[str]) -> dict:
    """Reduce a multiclass confusion matrix to alert-level counts.

    ``matrix[i][j]`` is the number of samples whose true label is ``labels[i]`` and
    whose predicted label is ``labels[j]`` (the layout produced by
    ``sklearn.metrics.confusion_matrix`` with ``labels=labels``). Rows and columns
    are collapsed to hazard vs background via :func:`to_hazard_label`. This lets the
    app derive operational metrics from a result JSON that already stores a
    confusion matrix, without re-running the model.
    """
    hazard_flags = [to_hazard_label(lbl) for lbl in labels]
    tp = fn = fp = tn = 0
    for i, true_is_hazard in enumerate(hazard_flags):
        for j, pred_is_hazard in enumerate(hazard_flags):
            count = int(matrix[i][j])
            if true_is_hazard and pred_is_hazard:
                tp += count
            elif true_is_hazard and not pred_is_hazard:
                fn += count
            elif (not true_is_hazard) and pred_is_hazard:
                fp += count
            else:
                tn += count

    return {
        "tp_alert": tp,
        "fn_alert": fn,
        "fp_alert": fp,
        "tn_alert": tn,
        "total_hazard_cases": tp + fn,
        "total_background_cases": fp + tn,
    }


# ── Operational metrics ────────────────────────────────────────────────────────

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` instead of raising on a zero denominator."""
    return numerator / denominator if denominator else default


def _maybe_int(value: float):
    """Return an int when ``value`` is integral, else a 4-decimal float.

    Keeps cost values clean (e.g. ``5480`` rather than ``5480.0``) while still
    supporting fractional weights.
    """
    f = float(value)
    return int(f) if f.is_integer() else round(f, 4)


def operational_alert_metrics_from_confusion(
    confusion: dict,
    fn_weight: float = DEFAULT_FN_WEIGHT,
    fp_weight: float = DEFAULT_FP_WEIGHT,
) -> dict:
    """Cost-sensitive operational metrics from alert-level confusion counts.

    Metrics:
        hazard_recall           = TP / (TP + FN)        — primary decision metric
        false_alert_rate        = FP / (FP + TN)        — secondary metric
        alert_precision         = TP / (TP + FP)
        alert_f1                = harmonic mean of alert_precision and hazard_recall
        weighted_error_cost     = fn_weight * FN + fp_weight * FP
        max_possible_cost       = fn_weight * hazard_cases + fp_weight * bg_cases
        operational_alert_score = 1 - weighted_error_cost / max_possible_cost
                                  (final ranking metric; higher is better)
    """
    tp = confusion["tp_alert"]
    fn = confusion["fn_alert"]
    fp = confusion["fp_alert"]
    tn = confusion["tn_alert"]
    total_hazard = tp + fn
    total_background = fp + tn

    hazard_recall = _safe_div(tp, tp + fn)
    false_alert_rate = _safe_div(fp, fp + tn)
    alert_precision = _safe_div(tp, tp + fp)
    alert_f1 = _safe_div(
        2 * alert_precision * hazard_recall, alert_precision + hazard_recall
    )

    weighted_error_cost = fn_weight * fn + fp_weight * fp
    max_possible_cost = fn_weight * total_hazard + fp_weight * total_background
    # No cases at all -> no possible cost -> treat as a perfect (vacuous) score.
    operational_alert_score = (
        1.0 if max_possible_cost == 0
        else 1.0 - weighted_error_cost / max_possible_cost
    )

    return {
        "tp_alert": int(tp),
        "fn_alert": int(fn),
        "fp_alert": int(fp),
        "tn_alert": int(tn),
        "total_hazard_cases": int(total_hazard),
        "total_background_cases": int(total_background),
        "hazard_recall": round(hazard_recall, 4),
        "false_alert_rate": round(false_alert_rate, 4),
        "alert_precision": round(alert_precision, 4),
        "alert_f1": round(alert_f1, 4),
        "weighted_error_cost": _maybe_int(weighted_error_cost),
        "max_possible_cost": _maybe_int(max_possible_cost),
        "operational_alert_score": round(operational_alert_score, 4),
        "fn_weight": _maybe_int(fn_weight),
        "fp_weight": _maybe_int(fp_weight),
    }


def compute_operational_alert_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    fn_weight: float = DEFAULT_FN_WEIGHT,
    fp_weight: float = DEFAULT_FP_WEIGHT,
) -> dict:
    """Alert-level operational metrics from raw true/predicted label sequences."""
    confusion = compute_alert_confusion(y_true, y_pred)
    return operational_alert_metrics_from_confusion(confusion, fn_weight, fp_weight)


def operational_alert_metrics_from_confusion_matrix(
    matrix,
    labels: Sequence[str],
    fn_weight: float = DEFAULT_FN_WEIGHT,
    fp_weight: float = DEFAULT_FP_WEIGHT,
) -> dict:
    """Operational metrics derived from a stored multiclass confusion matrix."""
    confusion = alert_confusion_from_confusion_matrix(matrix, labels)
    return operational_alert_metrics_from_confusion(confusion, fn_weight, fp_weight)


# ── Approximate fire-location helpers (object detectors only) ──────────────────

def bbox_bottom_center(
    x_center: float, y_center: float, width: float, height: float
) -> tuple[float, float]:
    """Bottom-center anchor of a bbox — the approximate ground point of a fire.

    Anchors at the bottom edge of the box (where flames meet the ground in image
    space)::

        anchor_x = x_center
        anchor_y = y_center + height / 2

    Coordinates stay in the box's own units (normalized 0-1 for YOLO labels). This
    is an APPROXIMATE image-space location, never precise geolocation.
    """
    return (x_center, y_center + height / 2.0)


def grid_cell(x: float, y: float, grid_size: int = 3) -> tuple[int, int]:
    """``(col, row)`` index of point ``(x, y)`` in a ``grid_size`` x ``grid_size`` grid.

    ``x`` and ``y`` are normalized image coordinates in [0, 1]. Indices are clamped
    into ``[0, grid_size - 1]`` so a point on the far border (``x`` or ``y`` == 1.0)
    maps to the last cell rather than overflowing.
    """
    def _index(value: float) -> int:
        cell = int(value * grid_size)
        if cell < 0:
            return 0
        if cell >= grid_size:
            return grid_size - 1
        return cell

    return (_index(x), _index(y))


def same_grid_cell(p1: tuple[float, float], p2: tuple[float, float], grid_size: int = 3) -> bool:
    """True when two points fall in the same ``grid_size`` x ``grid_size`` grid cell."""
    return grid_cell(p1[0], p1[1], grid_size) == grid_cell(p2[0], p2[1], grid_size)


def normalized_euclidean_distance(
    p1: tuple[float, float], p2: tuple[float, float]
) -> float:
    """Euclidean distance between two normalized image-space points.

    Inputs are normalized [0, 1] image coordinates, so the result is in the same
    normalized units: 0 for identical points, up to sqrt(2) for opposite corners.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def bbox_iou(
    box_a: Sequence[float], box_b: Sequence[float]
) -> float:
    """IoU of two YOLO-format boxes ``(x_center, y_center, width, height)``, normalized."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_x1, a_x2 = ax - aw / 2.0, ax + aw / 2.0
    a_y1, a_y2 = ay - ah / 2.0, ay + ah / 2.0
    b_x1, b_x2 = bx - bw / 2.0, bx + bw / 2.0
    b_y1, b_y2 = by - bh / 2.0, by + bh / 2.0

    inter_w = max(0.0, min(a_x2, b_x2) - max(a_x1, b_x1))
    inter_h = max(0.0, min(a_y2, b_y2) - max(a_y1, b_y1))
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def best_iou_fire_match(
    gt_fire_boxes: Sequence[Sequence[float]],
    pred_fire_boxes: Sequence[Sequence[float]],
) -> Optional[tuple]:
    """Highest-IoU ``(gt_box, pred_box, iou)`` fire-box pair for one image.

    Each box is ``(x_center, y_center, width, height)``. Returns ``None`` if either
    list is empty. This is a simple single-pair MVP rule, not full Hungarian
    assignment — sufficient for an approximate fire-location estimate.
    """
    if not gt_fire_boxes or not pred_fire_boxes:
        return None
    best: Optional[tuple] = None
    for gt in gt_fire_boxes:
        for pred in pred_fire_boxes:
            iou = bbox_iou(gt, pred)
            if best is None or iou > best[2]:
                best = (gt, pred, iou)
    return best


def fire_location_error(
    gt_fire_boxes: Sequence[Sequence[float]],
    pred_fire_boxes: Sequence[Sequence[float]],
    grid_size: int = 3,
) -> Optional[dict]:
    """Approximate fire-location error for one image, using bottom-center anchors.

    Matches the best-IoU GT/predicted fire-box pair, then compares their
    bottom-center anchors. Returns ``None`` when location cannot be computed
    (no GT fire boxes or no predicted fire boxes) — the caller should count this as
    a detection failure but must NOT record a numeric location error for it.

    On success returns::

        {
            "error":    normalized Euclidean distance between anchors,
            "grid_hit": bool — anchors fall in the same grid cell,
            "iou":      IoU of the matched fire boxes,
        }

    Only `fire` boxes (class 1) feed this. If an image has only smoke and no fire,
    do not call this a fire epicenter.
    """
    match = best_iou_fire_match(gt_fire_boxes, pred_fire_boxes)
    if match is None:
        return None
    gt_box, pred_box, iou = match
    gt_anchor = bbox_bottom_center(*gt_box)
    pred_anchor = bbox_bottom_center(*pred_box)
    return {
        "error": normalized_euclidean_distance(gt_anchor, pred_anchor),
        "grid_hit": same_grid_cell(gt_anchor, pred_anchor, grid_size),
        "iou": iou,
    }

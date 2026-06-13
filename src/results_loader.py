"""Loading, status classification, and winner selection for model result files.

PyroFinder stores two distinct kinds of result JSON in ``results/``:

* **Detection results** (``baseline_yolo11n.json``, ``baseline_yolo11s.json``)
  carry object-detection metrics (mAP, precision, recall, F1) and
  ``model_family == "object_detection"``.
* **Operational results** (``yolo11n_operational_metrics.json``,
  ``yolo11s_operational_metrics.json``) carry cost-sensitive alert metrics and
  ``evaluation_type == "operational_alert_metrics"``.

The two must never be mixed: an operational JSON loaded as a detection baseline
would show up with empty mAP/precision/recall, and a detection JSON loaded as an
operational row would show empty hazard metrics.

This module is intentionally dependency-light — only the Python standard library
and ``json`` — so it is importable from the Streamlit app and from unit tests
without pulling pandas, torch, or ultralytics. It never invents metric values:
a missing file yields ``training_in_progress`` and malformed JSON yields
``malformed`` rather than silent defaults.

YOLO11s is the current primary detector and YOLO11n is the lightweight baseline /
fallback; both now have measured result files. When a detector's result files are
absent, the loaders report ``training_in_progress`` and the winner logic refuses
to select it, so a detector with missing files can never win.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# ── Loader status codes ────────────────────────────────────────────────────────

STATUS_OK = "ok"
STATUS_TRAINING_IN_PROGRESS = "training_in_progress"
STATUS_MALFORMED = "malformed"
STATUS_NOT_DETECTION = "not_detection"
STATUS_NOT_OPERATIONAL = "not_operational"

_STATUS_LABELS = {
    STATUS_OK: "Measured",
    STATUS_TRAINING_IN_PROGRESS: "Training in progress",
    STATUS_MALFORMED: "Malformed result file",
    STATUS_NOT_DETECTION: "Not a detection result",
    STATUS_NOT_OPERATIONAL: "Not an operational result",
}

# Embedded ``status`` keywords that mark the metrics in a file as not real /
# not measured. A file whose own status contains any of these can never be
# selected as the winner, even if numeric fields happen to be filled in.
_NON_MEASURED_STATUS_KEYWORDS = ("synthetic", "placeholder", "pending", "training")

# Operational metrics required before a model is eligible to be the winner.
_REQUIRED_OPERATIONAL_KEYS = (
    "hazard_recall",
    "false_alert_rate",
    "operational_alert_score",
)


def status_label(status: str) -> str:
    """Human-readable label for a loader status code."""
    return _STATUS_LABELS.get(status, status)


# ── File loaders ─────────────────────────────────────────────────────────────

def _read_json(path: Path) -> tuple[str, Optional[dict]]:
    """Read and parse JSON, mapping a missing file to ``training_in_progress``.

    A missing result file means the model's results have not been produced yet
    (e.g. YOLO11s training is still running), so it is reported as
    ``training_in_progress`` rather than a hard error. Unparseable content is
    reported as ``malformed`` instead of silently inventing defaults.
    """
    if not path.exists():
        return STATUS_TRAINING_IN_PROGRESS, None
    try:
        return STATUS_OK, json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return STATUS_MALFORMED, None


def load_detection_result(path) -> dict:
    """Load a detection-metrics JSON, validating its kind.

    Returns ``{"status", "data", "path"}``:
        * ``training_in_progress`` — file does not exist yet.
        * ``malformed``            — file exists but is not valid JSON.
        * ``not_detection``        — JSON is an operational file, or its
                                     ``model_family`` is not ``object_detection``.
        * ``ok``                   — a genuine detection result.
    """
    p = Path(path)
    status, data = _read_json(p)
    if status != STATUS_OK:
        return {"status": status, "data": data, "path": str(p)}
    # An operational JSON must never be treated as a detection baseline row.
    if data.get("evaluation_type") == "operational_alert_metrics":
        return {"status": STATUS_NOT_DETECTION, "data": data, "path": str(p)}
    if data.get("model_family") != "object_detection":
        return {"status": STATUS_NOT_DETECTION, "data": data, "path": str(p)}
    return {"status": STATUS_OK, "data": data, "path": str(p)}


def load_operational_result(path) -> dict:
    """Load an operational-alert-metrics JSON, validating its kind.

    Returns ``{"status", "data", "path"}``:
        * ``training_in_progress`` — file does not exist yet.
        * ``malformed``            — file exists but is not valid JSON.
        * ``not_operational``      — JSON is not an ``operational_alert_metrics``
                                     document (e.g. a plain detection baseline).
        * ``ok``                   — a genuine operational result.
    """
    p = Path(path)
    status, data = _read_json(p)
    if status != STATUS_OK:
        return {"status": status, "data": data, "path": str(p)}
    if data.get("evaluation_type") != "operational_alert_metrics":
        return {"status": STATUS_NOT_OPERATIONAL, "data": data, "path": str(p)}
    return {"status": STATUS_OK, "data": data, "path": str(p)}


# ── Selectability + winner logic ───────────────────────────────────────────────

def is_selectable_operational(loaded: dict) -> bool:
    """True only when an operational result is real, measured, and complete.

    A model may be marked as selected only when its result files exist, contain
    measured values, the embedded ``status`` is not synthetic / placeholder /
    pending / training-in-progress, and the required operational metrics are not
    null. Pending or missing results are never selectable.
    """
    if loaded.get("status") != STATUS_OK:
        return False
    data = loaded.get("data") or {}
    embedded_status = str(data.get("status", "")).strip().lower()
    if any(kw in embedded_status for kw in _NON_MEASURED_STATUS_KEYWORDS):
        return False
    metrics = data.get("operational_metrics") or {}
    return all(metrics.get(key) is not None for key in _REQUIRED_OPERATIONAL_KEYS)


def _rank_candidates(candidates: list[dict]) -> Optional[str]:
    """Return the winning model name from pre-built candidate dicts, or None.

    Decision hierarchy (higher is better unless noted):
        1. Hazard Recall            — primary operational metric.
        2. False Alert Rate         — secondary (lower is better).
        3. Operational Alert Score  — weighted summary.
        4. Detection Recall, mAP@0.5 — supporting object-detection metrics.
        5. Inference speed          — only when a measured value exists.
    """
    eligible = [c for c in candidates if c.get("selectable")]
    if not eligible:
        return None

    def sort_key(c: dict):
        return (
            -(c.get("hazard_recall") or 0.0),
            c.get("false_alert_rate") if c.get("false_alert_rate") is not None else float("inf"),
            -(c.get("operational_alert_score") or 0.0),
            -(c.get("detection_recall") or 0.0),
            -(c.get("map50") or 0.0),
            c.get("inference_ms") if c.get("inference_ms") is not None else float("inf"),
        )

    eligible.sort(key=sort_key)
    return eligible[0]["model"]


def build_operational_candidates(operational_items, detection_items=None) -> list[dict]:
    """Build winner candidates from operational (and optional detection) files.

    ``operational_items`` / ``detection_items`` are iterables of
    ``(model_name, json_path)``. Detection files supply only the supporting
    tiebreak metrics (recall, mAP@0.5); a missing or malformed file simply omits
    those, it never blocks loading the operational row.
    """
    det_lookup: dict[str, dict] = {}
    for name, path in (detection_items or []):
        loaded = load_detection_result(path)
        if loaded["status"] == STATUS_OK:
            det_lookup[name] = (loaded["data"] or {}).get("metrics", {}) or {}

    candidates: list[dict] = []
    for name, path in operational_items:
        loaded = load_operational_result(path)
        metrics = (loaded["data"] or {}).get("operational_metrics", {}) if loaded["data"] else {}
        det = det_lookup.get(name, {})
        candidates.append({
            "model": name,
            "status": loaded["status"],
            "selectable": is_selectable_operational(loaded),
            "hazard_recall": metrics.get("hazard_recall"),
            "false_alert_rate": metrics.get("false_alert_rate"),
            "operational_alert_score": metrics.get("operational_alert_score"),
            "detection_recall": det.get("recall"),
            "map50": det.get("map50"),
            # Inference speed is only used when a measured value exists; we have
            # no operational inference-speed field yet, so leave it None.
            "inference_ms": None,
        })
    return candidates


def select_operational_winner(operational_items, detection_items=None) -> Optional[str]:
    """Return the winning model name, or None when no model is selectable.

    A model is only ever selected when its files exist, hold measured values,
    are not flagged synthetic/placeholder/pending/training, and have non-null
    required metrics — so a still-training YOLO11s can never win.
    """
    candidates = build_operational_candidates(operational_items, detection_items)
    return _rank_candidates(candidates)

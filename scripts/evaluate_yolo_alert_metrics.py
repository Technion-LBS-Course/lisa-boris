"""Operational alert + approximate fire-location metrics for a YOLO object detector.

EVALUATION ONLY — this script never trains. It loads an existing fine-tuned
checkpoint (e.g. YOLO11n, or later YOLO11s) on D-Fire, runs inference on the test
split, and reports the cost-sensitive operational metrics PyroFinder uses to rank
models:

  * Alert level: `fire`/`smoke` = hazard, `background` = no hazard. A MISSED HAZARD
    (false negative) is weighted ``fn_weight`` (default 10) — far worse than a FALSE
    ALERT (false positive), weighted ``fp_weight`` (default 1).
  * Approximate fire location: bottom-center anchor of class-1 (fire) boxes only,
    reported as normalized image-space error + 3x3 grid hit. This is an APPROXIMATE
    image-space location, never precise geolocation. Smoke-only images are never
    treated as a fire epicenter.

D-Fire class mapping (verified): class 0 = smoke, class 1 = fire.

YOLO11n is the lightweight object-detection baseline/fallback. YOLO11s remains the
planned primary detector. This script works for either checkpoint via --weights.

Usage
-----
  py scripts/evaluate_yolo_alert_metrics.py \
    --raw-root "<path-to-D-Fire-root>" \
    --weights "models/yolo11n_dfire_best.pt" \
    --model-name "YOLO11n" \
    --conf 0.25 \
    --output-json "results/yolo11n_operational_metrics.json" \
    --output-csv "results/yolo11n_test_predictions.csv"

Ultralytics/torch are imported lazily inside main(), so importing this module is cheap
and never pulls heavy ML dependencies at import time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from statistics import mean, median

# Make `src` importable when run as `py scripts/evaluate_yolo_alert_metrics.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation import (  # noqa: E402  (import after sys.path tweak)
    alert_outcome_from_flags,
    operational_alert_metrics_from_confusion,
    fire_location_error,
    DEFAULT_FN_WEIGHT,
    DEFAULT_FP_WEIGHT,
)

# ── D-Fire class ids (fixed) ───────────────────────────────────────────────────
SMOKE_CLASS = 0
FIRE_CLASS = 1

# ── Defaults ───────────────────────────────────────────────────────────────────
# Documented per-machine default also used by scripts/YOLO11n_baseline.py.
# Always overridable with --raw-root; never assume this path exists.
_DEFAULT_DFIRE_ROOT = Path(
    r"C:\Users\boris.azarov\OneDrive - Technion\Desktop\PyroFinder\RAW_DATA\D-Fire"
)
_DEFAULT_WEIGHTS = "models/yolo11n_dfire_best.pt"
_DEFAULT_MODEL_NAME = "YOLO11n"
_DEFAULT_CONF = 0.25
_DEFAULT_IMGSZ = 640
_DEFAULT_OUTPUT_JSON = "results/yolo11n_operational_metrics.json"
_DEFAULT_OUTPUT_CSV = "results/yolo11n_test_predictions.csv"
_GRID_SIZE = 3


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluation-only operational alert + approximate fire-location metrics "
            "for a YOLO fire/smoke detector on the D-Fire test split. No training."
        )
    )
    p.add_argument("--raw-root", type=Path, default=_DEFAULT_DFIRE_ROOT,
                   help="D-Fire root directory (expects test/images and test/labels).")
    p.add_argument("--weights", type=str, default=_DEFAULT_WEIGHTS,
                   help="Path to the fine-tuned checkpoint (e.g. models/yolo11n_dfire_best.pt).")
    p.add_argument("--model-name", type=str, default=_DEFAULT_MODEL_NAME,
                   help="Label used in the output JSON (e.g. YOLO11n or YOLO11s).")
    p.add_argument("--conf", type=float, default=_DEFAULT_CONF,
                   help="Confidence threshold for counting a predicted fire/smoke box.")
    p.add_argument("--imgsz", type=int, default=_DEFAULT_IMGSZ,
                   help="Inference image size (px).")
    p.add_argument("--fn-weight", type=float, default=DEFAULT_FN_WEIGHT,
                   help="Cost weight for a missed hazard (false negative). Default 10.")
    p.add_argument("--fp-weight", type=float, default=DEFAULT_FP_WEIGHT,
                   help="Cost weight for a false alert (false positive). Default 1.")
    p.add_argument("--output-json", type=Path, default=Path(_DEFAULT_OUTPUT_JSON))
    p.add_argument("--output-csv", type=Path, default=Path(_DEFAULT_OUTPUT_CSV))
    p.add_argument("--device", type=str, default="auto",
                   help="Inference device: 'auto', 'cpu', or a CUDA index like '0'.")
    return p.parse_args()


# ── Label parsing ────────────────────────────────────────────────────────────────

def _parse_label_file(label_path: Path) -> list[tuple]:
    """Return a list of (class_id, x_center, y_center, w, h) from a YOLO label file.

    Missing or empty files (D-Fire background images) yield an empty list.
    """
    boxes: list[tuple] = []
    if not label_path.exists():
        return boxes
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            x_c, y_c, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        boxes.append((cls_id, x_c, y_c, w, h))
    return boxes


def _fire_boxes(boxes: list[tuple]) -> list[tuple]:
    """Class-1 boxes as (x_center, y_center, w, h)."""
    return [(x, y, w, h) for (c, x, y, w, h) in boxes if c == FIRE_CLASS]


def _has_class(boxes: list[tuple], class_id: int) -> bool:
    return any(c == class_id for (c, *_rest) in boxes)


# ── Output helpers ───────────────────────────────────────────────────────────────

def _round_or_none(value, ndigits: int):
    return round(value, ndigits) if value is not None else None


def _save_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved JSON: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    # ── Validate raw dataset ──────────────────────────────────────────────────
    raw_root: Path = args.raw_root
    test_images = raw_root / "test" / "images"
    test_labels = raw_root / "test" / "labels"
    if not raw_root.exists():
        print(f"\nERROR: D-Fire root not found: {raw_root}")
        print("Pass the correct path with --raw-root.")
        return 1
    if not test_images.exists():
        print(f"\nERROR: Expected test images directory not found: {test_images}")
        print("Check that D-Fire is extracted with layout:")
        print("  D-Fire/test/images/   D-Fire/test/labels/")
        return 1
    if not test_labels.exists():
        print(f"\nWARNING: test labels directory not found: {test_labels}")
        print("Images without a label file will be treated as background (no hazard).")

    # ── Validate checkpoint ───────────────────────────────────────────────────
    weights = Path(args.weights)
    if not weights.exists():
        print(f"\nERROR: Checkpoint not found: {weights}")
        print("Train/obtain the checkpoint first, or pass --weights to an existing file.")
        print("YOLO11n baseline runner: py scripts/YOLO11n_baseline.py --train")
        return 1

    # ── Lazy ML imports (kept out of module import time) ──────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nERROR: ultralytics is not installed.")
        print("Install with: pip install ultralytics>=8.3")
        return 1

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    print(f"  Model name      : {args.model_name}")
    print(f"  Weights         : {weights}")
    print(f"  Test images     : {test_images}")
    print(f"  Conf threshold  : {args.conf}")
    print(f"  Image size      : {args.imgsz}")
    print(f"  Device          : {device}")
    print(f"  Cost weights    : FN={args.fn_weight}  FP={args.fp_weight}")

    try:
        model = YOLO(str(weights))
    except Exception as exc:  # corrupt/incompatible checkpoint
        print(f"\nERROR: failed to load checkpoint {weights}: {exc}")
        return 1

    # ── Inference over the test split (streamed, evaluation only) ─────────────
    print("\nRunning inference on the D-Fire test split (no training)...")
    predictions = model.predict(
        source=str(test_images),
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        stream=True,
        save=False,
        verbose=False,
    )

    rows: list[dict] = []
    tp = fn = fp = tn = 0
    gt_fire_image_count = 0          # images whose ground truth contains fire
    location_errors: list[float] = []
    grid_hits = 0
    covered = 0                      # images with BOTH gt fire and predicted fire

    processed = 0
    for result in predictions:
        img_path = Path(result.path)
        stem = img_path.stem
        gt = _parse_label_file(test_labels / f"{stem}.txt")
        gt_fire = _fire_boxes(gt)
        gt_has_fire = len(gt_fire) > 0
        gt_has_smoke = _has_class(gt, SMOKE_CLASS)

        # ── Predicted boxes (already filtered by --conf inside predict) ───────
        pred_fire: list[tuple] = []
        pred_has_smoke = False
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            cls_ids = [int(c) for c in boxes.cls.tolist()]
            xywhn = boxes.xywhn.tolist()
            for c, b in zip(cls_ids, xywhn):
                if c == FIRE_CLASS:
                    pred_fire.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
                elif c == SMOKE_CLASS:
                    pred_has_smoke = True
        pred_has_fire = len(pred_fire) > 0

        hazard_present = gt_has_fire or gt_has_smoke
        hazard_detected = pred_has_fire or pred_has_smoke
        outcome = alert_outcome_from_flags(hazard_present, hazard_detected)
        if outcome == "TP":
            tp += 1
        elif outcome == "FN":
            fn += 1
        elif outcome == "FP":
            fp += 1
        else:
            tn += 1

        # ── Approximate fire location (class-1 fire boxes only) ───────────────
        best_fire_iou = None
        loc_error = None
        loc_grid_hit = None
        if gt_has_fire:
            gt_fire_image_count += 1
            loc = fire_location_error(gt_fire, pred_fire, grid_size=_GRID_SIZE)
            if loc is not None:  # both GT fire and predicted fire exist
                covered += 1
                best_fire_iou = loc["iou"]
                loc_error = loc["error"]
                loc_grid_hit = loc["grid_hit"]
                location_errors.append(loc_error)
                if loc_grid_hit:
                    grid_hits += 1

        rows.append({
            "image_id": stem,
            "split": "test",
            "gt_has_fire": gt_has_fire,
            "gt_has_smoke": gt_has_smoke,
            "pred_has_fire": pred_has_fire,
            "pred_has_smoke": pred_has_smoke,
            "hazard_present": hazard_present,
            "hazard_detected": hazard_detected,
            "alert_outcome": outcome,
            "best_fire_iou": _round_or_none(best_fire_iou, 4),
            "fire_location_error": _round_or_none(loc_error, 6),
            "fire_location_grid_hit": loc_grid_hit,
        })

        processed += 1
        if processed % 500 == 0:
            print(f"  ...processed {processed} images")

    if processed == 0:
        print(f"\nERROR: no images found under {test_images}")
        return 1

    # ── Aggregate ─────────────────────────────────────────────────────────────
    confusion = {
        "tp_alert": tp, "fn_alert": fn, "fp_alert": fp, "tn_alert": tn,
        "total_hazard_cases": tp + fn, "total_background_cases": fp + tn,
    }
    operational = operational_alert_metrics_from_confusion(
        confusion, fn_weight=args.fn_weight, fp_weight=args.fp_weight
    )

    location_metrics = {
        "fire_location_error_mean": _round_or_none(mean(location_errors) if location_errors else None, 6),
        "fire_location_error_median": _round_or_none(median(location_errors) if location_errors else None, 6),
        "fire_location_grid_hit_rate": _round_or_none(grid_hits / covered if covered else None, 4),
        "location_coverage_count": covered,
        "location_coverage_rate": _round_or_none(covered / gt_fire_image_count if gt_fire_image_count else None, 4),
        "gt_fire_image_count": gt_fire_image_count,
        "grid_size": _GRID_SIZE,
        "anchor": "bottom-center (anchor_x = x_center, anchor_y = y_center + height/2)",
        "note": (
            "Approximate image-space fire location from class-1 (fire) bottom-center "
            "anchors, matched by best IoU per image. Not precise geolocation. "
            "Computed only when both ground-truth fire and predicted fire exist."
        ),
    }

    # ── Per-image CSV ─────────────────────────────────────────────────────────
    csv_path: Path = args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id", "split", "gt_has_fire", "gt_has_smoke",
        "pred_has_fire", "pred_has_smoke", "hazard_present", "hazard_detected",
        "alert_outcome", "best_fire_iou", "fire_location_error", "fire_location_grid_hit",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row[k] is None else row[k]) for k in fieldnames})
    print(f"\n  Saved predictions CSV: {csv_path} ({len(rows)} rows)")

    # ── JSON ────────────────────────────────────────────────────────────────────
    result_doc = {
        "model_name": args.model_name,
        "model_family": "object_detection",
        "evaluation_type": "operational_alert_metrics",
        "run_date": date.today().isoformat(),
        "weights": str(weights),
        "confidence_threshold": args.conf,
        "image_size": args.imgsz,
        "fn_weight": operational["fn_weight"],
        "fp_weight": operational["fp_weight"],
        "dataset": {
            "name": "D-Fire",
            "split": "test",
            "num_images_evaluated": processed,
            "class_mapping": {"0": "smoke", "1": "fire"},
        },
        "operational_metrics": operational,
        "location_metrics": location_metrics,
        "notes": (
            "Evaluation only; no training. fire/smoke count as hazard at the alert "
            "level; a missed hazard is weighted far more than a false alert. Location "
            "metrics are approximate image-space estimates, not precise geolocation."
        ),
    }
    _save_json(args.output_json, result_doc)

    # ── Console summary ───────────────────────────────────────────────────────
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  {args.model_name} — operational alert metrics (test split)")
    print(sep)
    print(f"  Images evaluated         : {processed}")
    print(f"  TP / FN / FP / TN        : {tp} / {fn} / {fp} / {tn}")
    print(f"  Hazard recall            : {operational['hazard_recall']:.4f}")
    print(f"  False alert rate         : {operational['false_alert_rate']:.4f}")
    print(f"  Alert precision          : {operational['alert_precision']:.4f}")
    print(f"  Alert F1                 : {operational['alert_f1']:.4f}")
    print(f"  Operational alert score  : {operational['operational_alert_score']:.4f}")
    print(f"  Fire location coverage   : {covered}/{gt_fire_image_count} "
          f"({location_metrics['location_coverage_rate']})")
    print(f"  Mean fire location error : {location_metrics['fire_location_error_mean']}")
    print(f"  Fire grid hit rate       : {location_metrics['fire_location_grid_hit_rate']}")
    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())

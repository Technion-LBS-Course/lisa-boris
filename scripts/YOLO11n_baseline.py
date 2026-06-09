"""
YOLO11n object-detection baseline for PyroFinder.

This is the reproducible local runner for YOLO11n baseline training and
evaluation on D-Fire (fire / smoke classes only).

The final M3 YOLO11n baseline results were produced on Kaggle (Notebook,
Tesla T4 GPU) because local hardware and Colab GPU availability were limited.
This script is kept for reproducibility and future local or cloud re-runs.

Expected result files (already committed / present locally):
  results/baseline_yolo11n.json   -- detection metrics
  results/results_yolo11n.csv     -- per-epoch training curves
  models/yolo11n_dfire_best.pt    -- best checkpoint (local only, Git-ignored)

Starting weights: Ultralytics downloads yolo11n.pt automatically when it is
not found locally. Do not assume the file is committed to the repository.

Usage
-----
  # Evaluate existing local checkpoint:
  python scripts/YOLO11n_baseline.py

  # Train from scratch then evaluate:
  python scripts/YOLO11n_baseline.py --train

  # Custom D-Fire root:
  python scripts/YOLO11n_baseline.py --train --raw-root "C:/path/to/D-Fire"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import yaml  # PyYAML — already in requirements.txt


# ── Default paths ─────────────────────────────────────────────────────────────

_DEFAULT_DFIRE_ROOT = Path(
    r"C:\Users\boris.azarov\OneDrive - Technion\Desktop\PyroFinder\RAW_DATA\D-Fire"
)
# Ultralytics downloads yolo11n.pt automatically if not present locally.
_DEFAULT_WEIGHTS = "yolo11n.pt"
_DEFAULT_TRAINED_WEIGHTS = "models/yolo11n_dfire_best.pt"
_DEFAULT_EPOCHS = 30
_DEFAULT_IMGSZ = 640
_DEFAULT_BATCH = 16
_DEFAULT_DEVICE = "auto"
_DEFAULT_PROJECT = "runs/detect"
_DEFAULT_RUN_NAME = "yolo11n_dfire_baseline"
_DEFAULT_RESULT_PATH = Path("results/baseline_yolo11n.json")
_DATA_YAML_PATH = Path("data/dfire_yolo11n.yaml")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO11n object-detection baseline — train and/or evaluate on D-Fire."
    )
    p.add_argument("--raw-root",         type=Path,  default=_DEFAULT_DFIRE_ROOT)
    p.add_argument("--weights",          type=str,   default=_DEFAULT_WEIGHTS)
    p.add_argument("--trained-weights",  type=Path,  default=Path(_DEFAULT_TRAINED_WEIGHTS))
    p.add_argument("--epochs",           type=int,   default=_DEFAULT_EPOCHS)
    p.add_argument("--imgsz",            type=int,   default=_DEFAULT_IMGSZ)
    p.add_argument("--batch",            type=int,   default=_DEFAULT_BATCH)
    p.add_argument("--device",           type=str,   default=_DEFAULT_DEVICE)
    p.add_argument("--train",            action="store_true",
                   help="Run training before evaluation.")
    p.add_argument("--project",          type=str,   default=_DEFAULT_PROJECT)
    p.add_argument("--name",             type=str,   default=_DEFAULT_RUN_NAME)
    p.add_argument("--result-path",      type=Path,  default=_DEFAULT_RESULT_PATH)
    return p.parse_args()


# ── Data YAML creation ────────────────────────────────────────────────────────

def _make_data_yaml(dfire_root: Path, yaml_path: Path) -> None:
    train_dir = (dfire_root / "train" / "images").as_posix()
    val_dir = (dfire_root / "test" / "images").as_posix()

    data = {
        "path": dfire_root.as_posix(),
        "train": "train/images",
        "val": "test/images",
        "nc": 2,
        "names": {0: "smoke", 1: "fire"},
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  Data YAML written: {yaml_path}")
    print(f"    train : {train_dir}")
    print(f"    val   : {val_dir}")


# ── Metric extraction helpers ─────────────────────────────────────────────────

def _safe_float(results, attr: str) -> float | None:
    """Extract a scalar float from a Ultralytics Results object, or return None."""
    try:
        val = getattr(results, attr, None)
        if val is None:
            return None
        if hasattr(val, "__len__") and len(val) == 0:
            return None
        scalar = float(val) if not hasattr(val, "__iter__") else float(val[0])
        return round(scalar, 4) if scalar == scalar else None  # NaN check
    except Exception:
        return None


def _safe_box_attr(results, path: list[str]) -> float | None:
    """Walk a dotted attribute path on results.box (or results.boxes)."""
    try:
        obj = results.box if hasattr(results, "box") else None
        if obj is None:
            obj = getattr(results, "boxes", None)
        if obj is None:
            return None
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        if hasattr(obj, "__len__"):
            val = float(obj[0]) if len(obj) > 0 else None
        else:
            val = float(obj)
        return round(val, 4) if val is not None and val == val else None
    except Exception:
        return None


def _extract_metrics(val_results) -> dict:
    """Return a metrics dict from Ultralytics validation results."""
    metrics: dict = {}

    # ── mAP values come from results.box.map / results.box.map50 ────────────
    box = getattr(val_results, "box", None)
    if box is not None:
        map50     = _try_scalar(getattr(box, "map50",     None))
        map50_95  = _try_scalar(getattr(box, "map",       None))
        precision = _try_scalar(getattr(box, "mp",        None))
        recall    = _try_scalar(getattr(box, "mr",        None))
    else:
        map50     = _try_scalar(getattr(val_results, "map50",  None))
        map50_95  = _try_scalar(getattr(val_results, "map",    None))
        precision = _try_scalar(getattr(val_results, "mp",     None))
        recall    = _try_scalar(getattr(val_results, "mr",     None))

    metrics["map50"]     = map50
    metrics["map50_95"]  = map50_95
    metrics["precision"] = precision
    metrics["recall"]    = recall

    # F1: compute from p and r if not directly available
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    else:
        f1 = None
    metrics["f1"] = f1

    # ── Per-class metrics ────────────────────────────────────────────────────
    per_class: dict = {}
    try:
        ap_class = getattr(box or val_results, "ap_class_index", None)
        maps      = getattr(box or val_results, "maps",           None)
        ap50      = getattr(box or val_results, "ap50",           None)

        if ap_class is not None and maps is not None:
            class_names = {0: "smoke", 1: "fire"}
            for i, cls_idx in enumerate(ap_class):
                cls_name = class_names.get(int(cls_idx), f"class_{cls_idx}")
                per_class[cls_name] = {
                    "map50":    round(float(ap50[i]),  4) if ap50 is not None and i < len(ap50) else None,
                    "map50_95": round(float(maps[i]),  4) if i < len(maps) else None,
                }
    except Exception:
        pass

    metrics["per_class"] = per_class if per_class else None
    return metrics


def _try_scalar(val) -> float | None:
    if val is None:
        return None
    try:
        import numpy as np
        if isinstance(val, (list, np.ndarray)):
            val = val[0] if len(val) > 0 else None
        if val is None:
            return None
        f = float(val)
        return round(f, 4) if f == f else None  # NaN guard
    except Exception:
        return None


# ── Console summary ───────────────────────────────────────────────────────────

def _print_summary(trained: bool, weights_used: str, yaml_path: Path,
                   metrics: dict, result_path: Path) -> None:
    sep = "-" * 55
    m = metrics
    print(f"\n{sep}")
    print("  YOLO11n baseline — results")
    print(sep)
    print(f"  Training run     : {'yes' if trained else 'no (loaded from checkpoint)'}")
    print(f"  Weights evaluated: {weights_used}")
    print(f"  Data YAML        : {yaml_path}")
    print(f"  mAP@0.5          : {m.get('map50')}")
    print(f"  mAP@0.5:0.95     : {m.get('map50_95')}")
    print(f"  Precision        : {m.get('precision')}")
    print(f"  Recall           : {m.get('recall')}")
    print(f"  F1               : {m.get('f1')}")
    print(f"  Saved JSON       : {result_path}")
    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Validate D-Fire root ─────────────────────────────────────────────────
    dfire_root: Path = args.raw_root
    if not dfire_root.exists():
        print(f"\nERROR: D-Fire root not found: {dfire_root}")
        print("Pass the correct path with --raw-root.")
        sys.exit(1)

    train_images = dfire_root / "train" / "images"
    test_images  = dfire_root / "test"  / "images"
    for p in (train_images, test_images):
        if not p.exists():
            print(f"\nERROR: Expected directory not found: {p}")
            print("Check that D-Fire is fully extracted and the layout is:")
            print("  D-Fire/train/images/  D-Fire/train/labels/")
            print("  D-Fire/test/images/   D-Fire/test/labels/")
            sys.exit(1)

    # ── Import Ultralytics ────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nERROR: ultralytics is not installed.")
        print("Install with: pip install ultralytics>=8.3")
        sys.exit(1)

    # ── Resolve device (auto → cpu when CUDA unavailable) ────────────────────
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"  Device resolved   : {device}")

    # ── Create data YAML ──────────────────────────────────────────────────────
    _make_data_yaml(dfire_root, _DATA_YAML_PATH)

    trained = False
    weights_used: str

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.train:
        print(f"\nStarting YOLO11n training:")
        print(f"  Starting weights : {args.weights}")
        print(f"  Data YAML        : {_DATA_YAML_PATH}")
        print(f"  Epochs           : {args.epochs}")
        print(f"  Image size       : {args.imgsz}")
        print(f"  Batch size       : {args.batch}")
        print(f"  Device           : {device}")
        print(f"  Project          : {args.project}")
        print(f"  Run name         : {args.name}")

        model = YOLO(args.weights)
        model.train(
            data=str(_DATA_YAML_PATH),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            verbose=True,
        )
        trained = True

        # best.pt is created inside the project/name directory
        best_path = Path(args.project) / args.name / "weights" / "best.pt"
        if best_path.exists():
            weights_used = str(best_path)
        else:
            # Ultralytics may use a different directory if exist_ok increments
            # Fall back to the user-specified trained-weights path
            weights_used = str(args.trained_weights)
    else:
        # ── No training — check for existing checkpoint ──────────────────────
        if not args.trained_weights.exists():
            print(
                f"\nNo trained YOLO11n checkpoint found at: {args.trained_weights}\n"
                "Run with --train first:\n"
                "  python scripts/YOLO11n_baseline.py --train\n"
            )
            sys.exit(0)
        weights_used = str(args.trained_weights)
        print(f"\nUsing existing checkpoint: {weights_used}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\nRunning validation on test split...")
    eval_model = YOLO(weights_used)
    val_results = eval_model.val(
        data=str(_DATA_YAML_PATH),
        imgsz=args.imgsz,
        split="val",
        device=device,
        verbose=True,
    )

    metrics = _extract_metrics(val_results)

    # ── Build result document ─────────────────────────────────────────────────
    result_doc = {
        "model_name": "YOLO11n (object detection baseline)",
        "model_family": "object_detection",
        "run_date": date.today().isoformat(),
        "task": "two-class object detection: fire / smoke",
        "dataset": {
            "name": "D-Fire",
            "source": "RAW_DATA/D-Fire (train/ + test/ pre-split)",
            "train_size": 17221,
            "test_size": 4306,
            "class_mapping": {"0": "smoke", "1": "fire"},
            "label_format": "YOLO normalized bounding boxes",
            "evaluation_split": "test folder used as YOLO val split",
        },
        "model_params": {
            "model": "YOLO11n",
            "starting_weights": args.weights,
            "trained_weights": weights_used,
            "imgsz": args.imgsz,
            "epochs": args.epochs if args.train else None,
            "batch": args.batch,
            "device": device,
        },
        "metrics": metrics,
        "notes": (
            "YOLO11n is the lightweight object-detection baseline for PyroFinder. "
            "It is not an image-level classifier and should be compared using "
            "detection metrics (mAP, precision, recall), not accuracy."
        ),
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result_path: Path = args.result_path
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _print_summary(trained, weights_used, _DATA_YAML_PATH, metrics, result_path)


if __name__ == "__main__":
    main()

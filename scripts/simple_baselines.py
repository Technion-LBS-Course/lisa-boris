"""
Simple sklearn baselines — fire/smoke image-level classification.

Pipeline:
  1. Load images from D-Fire (train/ and test/ splits as-is from the dataset).
     Falls back to data/samples/dfire/ on a machine without the raw data.
  2. Derive image-level label from YOLO label files:
       class 1 (fire) present → "fire"
       class 0 (smoke) only  → "smoke"
       empty file            → "background"
  3. Extract color features (60 values per image — see extract_features()).
  4. Train LogisticRegression (StandardScaler + LR) and RandomForestClassifier.
  5. Save metrics to results/ as JSON compatible with the app Baseline tab.

D-Fire class mapping: 0 = smoke, 1 = fire  (verified in CLAUDE.md)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier  # noqa: F401 — imported for reference
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────

DFIRE_ROOT = Path(
    r"C:\Users\boris.azarov\OneDrive - Technion\Desktop\PyroFinder\RAW_DATA\D-Fire"
)
SAMPLES_ROOT = Path("data/samples/dfire")
RESULTS_DIR = Path("results")

# Set to a positive integer to limit images per split (useful for quick runs).
MAX_PER_SPLIT: int | None = None


# ── Label derivation ─────────────────────────────────────────────────────────

def derive_label(label_path: Path) -> str:
    """Image-level label from a YOLO label file."""
    text = label_path.read_text().strip()
    if not text:
        return "background"
    classes = {int(line.split()[0]) for line in text.splitlines() if line.strip()}
    return "fire" if 1 in classes else "smoke"


# ── Feature extraction ───────────────────────────────────────────────────────

_HIST_BINS = 16  # bins per channel for color histogram


def extract_features(image_path: Path) -> np.ndarray:
    """Extract 60-value color feature vector from one image.

    Feature vector (total 60 values):
      - RGB mean per channel   → 3
      - RGB std per channel    → 3
      - HSV mean per channel   → 3
      - HSV std per channel    → 3
      - Color histogram 16-bin × 3 RGB channels → 48
    Total: 60

    Fire pixels are orange/red (high R, low B) and high-saturation;
    smoke pixels are gray (low saturation, mid-value). Histograms capture
    this better than flattened pixels at the cost of spatial information.
    """
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    rgb = np.array(img, dtype=np.float32) / 255.0  # (64, 64, 3)
    hsv = np.array(img.convert("HSV"), dtype=np.float32) / 255.0

    rgb_mean_std = np.array(
        [rgb[:, :, c].mean() for c in range(3)] +
        [rgb[:, :, c].std()  for c in range(3)]
    )
    hsv_mean_std = np.array(
        [hsv[:, :, c].mean() for c in range(3)] +
        [hsv[:, :, c].std()  for c in range(3)]
    )

    hist = np.concatenate([
        np.histogram(rgb[:, :, c].flatten(), bins=_HIST_BINS, range=(0, 1))[0]
        for c in range(3)
    ]).astype(np.float32)
    hist /= hist.sum() + 1e-8  # normalize to sum ≈ 1

    return np.concatenate([rgb_mean_std, hsv_mean_std, hist])


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_split(images_dir: Path, labels_dir: Path, limit: int | None) -> tuple:
    """Return (X, y, names) for one split folder."""
    X, y, names = [], [], []
    rng = np.random.default_rng(42)
    paths = np.array(sorted(images_dir.glob("*.jpg")))
    rng.shuffle(paths)
    if limit is not None:
        paths = paths[:limit]
    for img_path in paths:
        label_path = labels_dir / img_path.with_suffix(".txt").name
        if not label_path.exists():
            continue
        X.append(extract_features(img_path))
        y.append(derive_label(label_path))
        names.append(img_path.name)
    return np.array(X), np.array(y), names


def build_dataset() -> tuple:
    """Load train and test sets, preferring the full D-Fire raw data."""
    if DFIRE_ROOT.exists():
        print(f"Using full D-Fire dataset: {DFIRE_ROOT}")
        train_imgs = DFIRE_ROOT / "train" / "images"
        train_lbls = DFIRE_ROOT / "train" / "labels"
        test_imgs  = DFIRE_ROOT / "test"  / "images"
        test_lbls  = DFIRE_ROOT / "test"  / "labels"

        print("Loading train split...")
        X_train, y_train, _ = load_split(train_imgs, train_lbls, MAX_PER_SPLIT)
        print("Loading test split...")
        X_test,  y_test,  _ = load_split(test_imgs,  test_lbls,  MAX_PER_SPLIT)

    else:
        print(f"D-Fire root not found - falling back to samples: {SAMPLES_ROOT}")
        X, y, _ = load_split(
            SAMPLES_ROOT / "images", SAMPLES_ROOT / "labels", limit=None
        )
        _, counts = np.unique(y, return_counts=True)
        can_stratify = all(c >= 2 for c in counts)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if can_stratify else None,
        )

    return X_train, X_test, y_train, y_test


# ── Metrics helpers ──────────────────────────────────────────────────────────

_CLASSES_ORDERED = ["background", "fire", "smoke"]

_FEATURES_META = {
    "description": "Color statistics + normalized color histogram",
    "vector_length": 60,
    "components": [
        {"name": "RGB mean per channel",       "size": 3},
        {"name": "RGB std per channel",        "size": 3},
        {"name": "HSV mean per channel",       "size": 3},
        {"name": "HSV std per channel",        "size": 3},
        {"name": "Color histogram 16-bin × 3", "size": 48},
    ],
    "image_resize": "64x64",
    "normalization": "pixel values / 255; histogram normalized to sum=1",
}

_DATASET_META_TEMPLATE = {
    "name": "D-Fire",
    "source": "RAW_DATA/D-Fire (train/ + test/ pre-split)",
    "label_derivation": "class 1 present → fire; class 0 only → smoke; empty file → background",
}


def _class_dist(y: np.ndarray) -> dict:
    labels, counts = np.unique(y, return_counts=True)
    return {str(lbl): int(cnt) for lbl, cnt in zip(labels, counts)}


def _build_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(
        y_test, y_pred,
        labels=_CLASSES_ORDERED,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred, labels=_CLASSES_ORDERED)

    clf_report = {
        cls: {
            "precision": round(report[cls]["precision"], 4),
            "recall":    round(report[cls]["recall"],    4),
            "f1":        round(report[cls]["f1-score"],  4),
            "support":   int(report[cls]["support"]),
        }
        for cls in _CLASSES_ORDERED
        if cls in report
    }

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "classification_report": clf_report,
        "macro_avg": {
            "precision": round(report["macro avg"]["precision"], 4),
            "recall":    round(report["macro avg"]["recall"],    4),
            "f1":        round(report["macro avg"]["f1-score"],  4),
        },
        "weighted_avg": {
            "precision": round(report["weighted avg"]["precision"], 4),
            "recall":    round(report["weighted avg"]["recall"],    4),
            "f1":        round(report["weighted avg"]["f1-score"],  4),
        },
        "confusion_matrix": {
            "labels": _CLASSES_ORDERED,
            "matrix": cm.tolist(),
        },
    }


def _print_summary(name: str, metrics: dict) -> None:
    cr = metrics["classification_report"]
    sep = "-" * 50
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  F1 macro        : {metrics['macro_avg']['f1']:.4f}")
    print(f"  Fire recall     : {cr.get('fire',  {}).get('recall', 0):.4f}")
    print(f"  Smoke recall    : {cr.get('smoke', {}).get('recall', 0):.4f}")


def _save_result(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved to: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = build_dataset()

    print(f"\nFeature vector length : {X_train.shape[1]}")
    print(f"Train size            : {len(y_train)}")
    print(f"Test  size            : {len(y_test)}")
    print("Train distribution    :", _class_dist(y_train))
    print("Test  distribution    :", _class_dist(y_test))

    dataset_meta = {
        **_DATASET_META_TEMPLATE,
        "train_size": int(len(y_train)),
        "test_size":  int(len(y_test)),
        "class_distribution": {
            "train": _class_dist(y_train),
            "test":  _class_dist(y_test),
        },
    }

    run_date = date.today().isoformat()

    # ── A. Logistic Regression ────────────────────────────────────────────────

    print("\nTraining Logistic Regression...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred    = lr_pipe.predict(X_test)
    lr_metrics = _build_metrics(y_test, lr_pred)
    _print_summary("Logistic Regression (color features)", lr_metrics)

    lr_doc = {
        "model_name": "Logistic Regression (color features)",
        "run_date":   run_date,
        "task":       "image-level classification: fire / smoke / background",
        "dataset":    dataset_meta,
        "features":   _FEATURES_META,
        "model_params": {
            "pipeline":     "StandardScaler + LogisticRegression",
            "max_iter":     1000,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "metrics": lr_metrics,
        "notes": (
            "Simple sklearn baseline using handcrafted color features. "
            "Must beat DummyClassifier macro F1=0.21 and achieve recall>0 on fire and smoke."
        ),
    }
    _save_result(RESULTS_DIR / "baseline_logistic_regression.json", lr_doc)

    # ── B. Random Forest ─────────────────────────────────────────────────────

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred    = rf.predict(X_test)
    rf_metrics = _build_metrics(y_test, rf_pred)
    _print_summary("Random Forest (color features)", rf_metrics)

    rf_doc = {
        "model_name": "Random Forest (color features)",
        "run_date":   run_date,
        "task":       "image-level classification: fire / smoke / background",
        "dataset":    dataset_meta,
        "features":   _FEATURES_META,
        "model_params": {
            "n_estimators":     200,
            "max_depth":        None,
            "min_samples_split": 2,
            "min_samples_leaf":  1,
            "class_weight":     "balanced",
            "random_state":     42,
            "n_jobs":           -1,
        },
        "metrics": rf_metrics,
        "notes": (
            "Simple sklearn baseline using handcrafted color features. "
            "Must beat DummyClassifier macro F1=0.21 and achieve recall>0 on fire and smoke."
        ),
    }
    _save_result(RESULTS_DIR / "baseline_random_forest.json", rf_doc)

    print("\nDone.")

"""
Dummy baseline — fire/smoke image-level classification.

Pipeline:
  1. Load images from D-Fire (train/ and test/ splits as-is from the dataset).
     Falls back to data/samples/dfire/ on a machine without the raw data.
  2. Derive image-level label from YOLO label files:
       class 1 (fire) present → "fire"
       class 0 (smoke) only  → "smoke"
       empty file            → "background"
  3. Extract color + texture features (see extract_features()).
  4. DummyClassifier baseline (most_frequent strategy).
  5. Report accuracy and classification report.

D-Fire class mapping: 0 = smoke, 1 = fire  (verified in CLAUDE.md)
"""

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────

DFIRE_ROOT = Path(
    r"C:\Users\boris.azarov\OneDrive - Technion\Desktop\PyroFinder\RAW_DATA\D-Fire"
)
SAMPLES_ROOT = Path("data/samples/dfire")

# Set to a positive integer to limit images per split (useful for quick runs).
# Set to None to use the full dataset.
MAX_PER_SPLIT: int | None = None


# ── Label derivation ─────────────────────────────────────────────────────────

def derive_label(label_path: Path) -> str:
    """Image-level label from a YOLO label file.

    If any fire box (class 1) is present → 'fire'.
    Smoke-only boxes → 'smoke'.
    Empty file (no detections) → 'background'.
    """
    text = label_path.read_text().strip()
    if not text:
        return "background"
    classes = {int(line.split()[0]) for line in text.splitlines() if line.strip()}
    return "fire" if 1 in classes else "smoke"


# ── Feature extraction ───────────────────────────────────────────────────────

_HIST_BINS = 16   # bins per channel for color histogram

def extract_features(image_path: Path) -> np.ndarray:
    """Extract color and texture features from one image.

    Features (total 54 values):
      - Per-channel mean and std for R, G, B  → 6 values
      - Per-channel mean and std for H, S, V  → 6 values
      - Color histogram (16 bins × 3 channels) → 48 values

    Rationale: fire pixels are orange/red (high R, low B) and high-saturation;
    smoke pixels are gray (low saturation, mid-value). Raw pixel histograms
    capture this better than flattened pixels at the cost of spatial info.
    """
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    rgb = np.array(img, dtype=np.float32) / 255.0   # (64, 64, 3)

    hsv = np.array(img.convert("HSV"), dtype=np.float32) / 255.0

    # Per-channel mean + std
    rgb_mean_std = np.array(
        [rgb[:, :, c].mean() for c in range(3)] +
        [rgb[:, :, c].std()  for c in range(3)]
    )
    hsv_mean_std = np.array(
        [hsv[:, :, c].mean() for c in range(3)] +
        [hsv[:, :, c].std()  for c in range(3)]
    )

    # Color histograms per channel
    hist = np.concatenate([
        np.histogram(rgb[:, :, c].flatten(), bins=_HIST_BINS, range=(0, 1))[0]
        for c in range(3)
    ]).astype(np.float32)
    hist /= hist.sum() + 1e-8   # normalize to sum ≈ 1

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


def build_dataset():
    """Load train and test sets, preferring the full D-Fire raw data."""
    if DFIRE_ROOT.exists():
        print(f"Using full D-Fire dataset: {DFIRE_ROOT}")
        train_imgs = DFIRE_ROOT / "train" / "images"
        train_lbls = DFIRE_ROOT / "train" / "labels"
        test_imgs  = DFIRE_ROOT / "test"  / "images"
        test_lbls  = DFIRE_ROOT / "test"  / "labels"

        print("Loading train split…")
        X_train, y_train, _ = load_split(train_imgs, train_lbls, MAX_PER_SPLIT)
        print("Loading test split…")
        X_test,  y_test,  _ = load_split(test_imgs,  test_lbls,  MAX_PER_SPLIT)

    else:
        print(f"D-Fire root not found — falling back to samples: {SAMPLES_ROOT}")
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


# ── Main ─────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = build_dataset()

print(f"\nFeature vector length : {X_train.shape[1]}")
print(f"Train size            : {len(y_train)}")
print(f"Test  size            : {len(y_test)}")
print("Train distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Test  distribution:", dict(zip(*np.unique(y_test,  return_counts=True))))

# ── DummyClassifier baseline ─────────────────────────────────────────────────

dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)

print(f"\nDummy accuracy : {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Strategy: always predict the most frequent training class.")
print("A real model must beat this baseline on F1 for fire and smoke.")

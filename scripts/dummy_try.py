"""
Dummy baseline — fire/smoke image-level classification.

Pipeline:
  1. Load each sample image from data/samples/dfire/images/
  2. Derive image-level label from its YOLO label file (fire / smoke / background)
  3. Extract simple pixel statistics as features (DummyClassifier ignores them,
     but the pipeline mirrors what a real model would receive)
  4. Train/test split → DummyClassifier (most_frequent strategy)
  5. Report accuracy and classification report

D-Fire class mapping: 0 = smoke, 1 = fire  (verified in CLAUDE.md)
"""

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

IMAGES_DIR = Path("data/samples/dfire/images")
LABELS_DIR = Path("data/samples/dfire/labels")


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


def extract_features(image_path: Path) -> np.ndarray:
    """Flatten image to a 1D array (32×32×3 = 3072 values)."""
    return np.array(
        Image.open(image_path).convert("RGB").resize((32, 32)), dtype=np.float32
    ).flatten() / 255.0


# ── Build dataset ────────────────────────────────────────────────────────────

X, y, names = [], [], []

for img_path in sorted(IMAGES_DIR.glob("*.jpg")):
    label_path = LABELS_DIR / img_path.with_suffix(".txt").name
    if not label_path.exists():
        continue
    X.append(extract_features(img_path))
    y.append(derive_label(label_path))
    names.append(img_path.name)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(y)} images from {IMAGES_DIR}")
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# ── Train / test split ───────────────────────────────────────────────────────

# Stratify only when every class has ≥ 2 samples (required by sklearn)
can_stratify = all(c >= 2 for c in counts)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if can_stratify else None,
)

print(f"\nSplit -> train: {len(y_train)}  test: {len(y_test)}")
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
print("A real model must beat this baseline on recall for fire and smoke.")

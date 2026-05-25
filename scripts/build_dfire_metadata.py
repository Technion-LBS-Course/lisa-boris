"""Build D-Fire metadata CSV from local raw dataset files.

Usage:
    python scripts/build_dfire_metadata.py \
        --raw-root "C:/path/to/D-Fire" \
        --output data/dfire_metadata.csv \
        [--max-images N] \
        [--copy-samples data/samples/dfire --sample-count 20]

D-Fire class mapping (no data.yaml found in this download):
    class 0 -> smoke
    class 1 -> fire
Verified by comparing scan results against official D-Fire image-category counts.
See docs/M2_DATA_EDA.md for full documentation of this assumption.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ── Class mapping constant ─────────────────────────────────────────────────────
# D-Fire has no data.yaml in this download.
# Mapping verified by cross-checking scan results with official D-Fire category counts:
#   fire_only=1164, smoke_only=5867, fire_and_smoke=4658, background=9838.
# Swapping 0<->1 produces the exact documented counts, confirming:
#   0 = smoke, 1 = fire
DFIRE_CLASS_MAP: dict[int, str] = {0: "smoke", 1: "fire"}

# ── Split inference ────────────────────────────────────────────────────────────

def _infer_split(image_path: Path, raw_root: Path) -> str:
    """Infer train/val/test split from path components."""
    try:
        rel = image_path.relative_to(raw_root)
        parts = [p.lower() for p in rel.parts]
    except ValueError:
        parts = [p.lower() for p in image_path.parts]

    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return "unknown"


# ── Label file resolution ──────────────────────────────────────────────────────

def _find_label_file(image_path: Path) -> Path | None:
    """Locate the YOLO label .txt for an image using common dataset layouts."""
    stem = image_path.stem

    # Same-level labels sibling: images/ -> labels/
    candidates: list[Path] = []
    parent = image_path.parent

    # images/X.jpg -> labels/X.txt  (standard layout)
    if parent.name.lower() in ("images",):
        labels_dir = parent.parent / "labels"
        candidates.append(labels_dir / f"{stem}.txt")

    # Flat layout: same directory
    candidates.append(parent / f"{stem}.txt")

    # One level up / labels/
    candidates.append(parent.parent / "labels" / f"{stem}.txt")

    for c in candidates:
        if c.exists():
            return c
    return None


# ── Spatial helpers ────────────────────────────────────────────────────────────

def _to_third(v: float) -> int:
    """Map a 0–1 normalized coordinate to a thirds index (1=first, 2=second, 3=third)."""
    if v < 1 / 3:
        return 1
    if v < 2 / 3:
        return 2
    return 3


def _bbox_iou(b1: dict, b2: dict) -> float:
    """Compute IoU between two YOLO-format bbox dicts (x_center, y_center, w, h)."""
    x1_lo = b1["x_center"] - b1["w"] / 2
    x1_hi = b1["x_center"] + b1["w"] / 2
    y1_lo = b1["y_center"] - b1["h"] / 2
    y1_hi = b1["y_center"] + b1["h"] / 2

    x2_lo = b2["x_center"] - b2["w"] / 2
    x2_hi = b2["x_center"] + b2["w"] / 2
    y2_lo = b2["y_center"] - b2["h"] / 2
    y2_hi = b2["y_center"] + b2["h"] / 2

    ix = max(0.0, min(x1_hi, x2_hi) - max(x1_lo, x2_lo))
    iy = max(0.0, min(y1_hi, y2_hi) - max(y1_lo, y2_lo))
    intersection = ix * iy
    union = b1["w"] * b1["h"] + b2["w"] * b2["h"] - intersection
    return intersection / union if union > 0 else 0.0


# ── Label parsing ──────────────────────────────────────────────────────────────

def _parse_label_file(label_path: Path) -> list[dict]:
    """Parse a YOLO label file. Returns list of box dicts."""
    boxes: list[dict] = []
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
        except ValueError:
            continue
        class_name = DFIRE_CLASS_MAP.get(cls_id)
        if class_name is None:
            warnings.warn(
                f"Unknown class id {cls_id} in {label_path}. Skipping box."
            )
            continue
        boxes.append({"class_name": class_name, "x_center": x_c, "y_center": y_c, "w": w, "h": h})
    return boxes


# ── Image pixel statistics ─────────────────────────────────────────────────────

# Near-black threshold: perceived brightness below this fraction counts as a dark pixel.
_DARK_PIXEL_THRESHOLD = 30 / 255
# Thumbnail size used for pixel stat computation (speed vs. precision trade-off).
_STAT_THUMB_SIZE = (64, 64)


def _compute_image_stats(img: Image.Image) -> dict:
    """Compute pixel-level statistics from a PIL image using a 64×64 thumbnail.

    IMPORTANT: values are derived from the stored JPEG/PNG file as it exists on
    disk — not raw sensor data. JPEG compression, gamma correction, and any
    pre-processing applied by the original dataset creators all affect these numbers.
    """
    thumb = img.resize(_STAT_THUMB_SIZE, Image.LANCZOS).convert("RGB")
    arr = np.asarray(thumb, dtype=np.float32) / 255.0  # shape (64, 64, 3)

    # ITU-R BT.601 luma — perceived brightness, not raw luminance
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    return {
        "mean_brightness": round(float(gray.mean()), 4),
        "dark_pixel_ratio": round(float((gray < _DARK_PIXEL_THRESHOLD).mean()), 4),
        # Mean of per-channel standard deviation — higher = more colourful / varied scene
        "color_std_mean": round(float(np.mean([arr[:, :, c].std() for c in range(3)])), 4),
    }


# ── Per-image row ──────────────────────────────────────────────────────────────

def _build_row(image_path: Path, raw_root: Path) -> tuple[dict, bool] | tuple[None, bool]:
    """Build one metadata row for an image. Returns (row_dict, is_empty_label).

    Returns (None, False) on fatal error (corrupt/unreadable image).
    is_empty_label is True when a label file exists but contains no valid boxes.
    """
    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
            pixel_stats = _compute_image_stats(img)
    except Exception as exc:
        warnings.warn(f"Cannot open {image_path}: {exc}. Skipping.")
        return None, False

    split = _infer_split(image_path, raw_root)
    label_path = _find_label_file(image_path)
    has_label = label_path is not None

    boxes: list[dict] = []
    is_empty_label = False
    if has_label:
        boxes = _parse_label_file(label_path)
        if not boxes:
            is_empty_label = True

    fire_boxes = [b for b in boxes if b["class_name"] == "fire"]
    smoke_boxes = [b for b in boxes if b["class_name"] == "smoke"]

    has_fire = len(fire_boxes) > 0
    has_smoke = len(smoke_boxes) > 0

    if has_fire and has_smoke:
        category = "fire_and_smoke"
    elif has_fire:
        category = "fire_only"
    elif has_smoke:
        category = "smoke_only"
    else:
        category = "background"

    areas = [b["w"] * b["h"] for b in boxes]
    aspect_ratios = [b["w"] / b["h"] if b["h"] > 0 else float("nan") for b in boxes]

    fire_areas = [b["w"] * b["h"] for b in fire_boxes]
    smoke_areas = [b["w"] * b["h"] for b in smoke_boxes]

    # ── Spatial position ─────────────────────────────────────────────────────
    _nan = float("nan")
    fire_mx = float(np.mean([b["x_center"] for b in fire_boxes])) if fire_boxes else _nan
    fire_my = float(np.mean([b["y_center"] for b in fire_boxes])) if fire_boxes else _nan
    smoke_mx = float(np.mean([b["x_center"] for b in smoke_boxes])) if smoke_boxes else _nan
    smoke_my = float(np.mean([b["y_center"] for b in smoke_boxes])) if smoke_boxes else _nan

    # thirds grid: 0 = class not present, 1/2/3 = position third
    fire_tc = _to_third(fire_mx) if fire_boxes else 0
    fire_tr = _to_third(fire_my) if fire_boxes else 0
    smoke_tc = _to_third(smoke_mx) if smoke_boxes else 0
    smoke_tr = _to_third(smoke_my) if smoke_boxes else 0

    # relative smoke vs fire (only when both present; negative dy = smoke above fire, y=0 is top)
    if fire_boxes and smoke_boxes:
        smoke_dy = smoke_my - fire_my
        smoke_dx = smoke_mx - fire_mx
        mean_iou = float(np.mean([_bbox_iou(f, s) for f in fire_boxes for s in smoke_boxes]))
    else:
        smoke_dy = _nan
        smoke_dx = _nan
        mean_iou = _nan

    row = {
        "image_id": image_path.stem,
        "split": split,
        "image_path": str(image_path),
        "label_path": str(label_path) if has_label else "",
        "has_label": has_label,
        "has_fire": has_fire,
        "has_smoke": has_smoke,
        "image_category": category,
        "num_fire_boxes": len(fire_boxes),
        "num_smoke_boxes": len(smoke_boxes),
        "total_boxes": len(boxes),
        "mean_bbox_area": float(np.mean(areas)) if areas else 0.0,
        "median_bbox_area": float(np.median(areas)) if areas else 0.0,
        "max_bbox_area": float(np.max(areas)) if areas else 0.0,
        "mean_bbox_aspect_ratio": float(np.nanmean(aspect_ratios)) if aspect_ratios else 0.0,
        # Per-class bbox area (0.0 when class absent)
        "fire_mean_bbox_area": float(np.mean(fire_areas)) if fire_areas else 0.0,
        "smoke_mean_bbox_area": float(np.mean(smoke_areas)) if smoke_areas else 0.0,
        # Total normalized area covered by each class in this image
        "fire_bbox_coverage": float(sum(fire_areas)),
        "smoke_bbox_coverage": float(sum(smoke_areas)),
        "image_width": img_w,
        "image_height": img_h,
        "source_dataset": "D-Fire",
        **pixel_stats,  # mean_brightness, dark_pixel_ratio, color_std_mean
        # Spatial position (normalized 0–1; NaN when class absent)
        "fire_mean_x_center": fire_mx,
        "fire_mean_y_center": fire_my,
        "smoke_mean_x_center": smoke_mx,
        "smoke_mean_y_center": smoke_my,
        # Thirds grid (1/2/3 = left/center/right or top/mid/bottom; 0 = class absent)
        "fire_thirds_col": fire_tc,
        "fire_thirds_row": fire_tr,
        "smoke_thirds_col": smoke_tc,
        "smoke_thirds_row": smoke_tr,
        # Smoke relative to fire (only for fire_and_smoke; negative dy = smoke above fire)
        "smoke_dy_vs_fire": smoke_dy,
        "smoke_dx_vs_fire": smoke_dx,
        "fire_smoke_mean_iou": mean_iou,
    }
    return row, is_empty_label


# ── Sample image copying ───────────────────────────────────────────────────────

_BOX_COLORS = {"fire": (255, 80, 0), "smoke": (120, 120, 200)}


def _draw_boxes(image_path: Path, label_path: Path | None, out_path: Path) -> None:
    """Copy image to out_path and draw YOLO bounding boxes if label exists."""
    with Image.open(image_path).convert("RGB") as img:
        if label_path and label_path.exists():
            boxes = _parse_label_file(label_path)
            draw = ImageDraw.Draw(img)
            w, h = img.size
            for b in boxes:
                x1 = (b["x_center"] - b["w"] / 2) * w
                y1 = (b["y_center"] - b["h"] / 2) * h
                x2 = (b["x_center"] + b["w"] / 2) * w
                y2 = (b["y_center"] + b["h"] / 2) * h
                color = _BOX_COLORS.get(b["class_name"], (255, 255, 0))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1 + 2, y1 + 2), b["class_name"], fill=color)
        img.save(out_path)


def copy_samples(
    df: pd.DataFrame,
    samples_dir: Path,
    sample_count: int = 20,
) -> None:
    """Copy sample images (preferring labeled ones) to samples_dir with drawn boxes."""
    samples_dir = Path(samples_dir)
    img_out = samples_dir / "images"
    lbl_out = samples_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # Prefer images that have labels and detections
    labeled = df[df["has_label"]].copy()
    with_boxes = labeled[labeled["total_boxes"] > 0]
    background = labeled[labeled["total_boxes"] == 0]

    sample_rows = pd.concat([with_boxes, background]).head(sample_count)
    if len(sample_rows) < sample_count:
        extra = df[~df.index.isin(sample_rows.index)].head(sample_count - len(sample_rows))
        sample_rows = pd.concat([sample_rows, extra])

    copied = 0
    for _, row in sample_rows.iterrows():
        src_img = Path(row["image_path"])
        label_path = Path(row["label_path"]) if row["label_path"] else None
        if not src_img.exists():
            continue
        out_img = img_out / src_img.name
        try:
            _draw_boxes(src_img, label_path, out_img)
        except Exception as exc:
            warnings.warn(f"Could not copy sample {src_img.name}: {exc}")
            continue
        if label_path and label_path.exists():
            shutil.copy2(label_path, lbl_out / label_path.name)
        copied += 1

    print(f"Copied {copied} sample images to {samples_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────

def build_metadata(
    raw_root: Path,
    output: Path,
    max_images: int | None = None,
    copy_samples_dir: Path | None = None,
    sample_count: int = 20,
) -> pd.DataFrame:
    raw_root = Path(raw_root).resolve()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        print(f"ERROR: raw-root does not exist: {raw_root}", file=sys.stderr)
        sys.exit(1)

    # Warn about missing class mapping file
    yaml_candidates = list(raw_root.rglob("data.yaml")) + list(raw_root.rglob("dataset.yaml"))
    if not yaml_candidates:
        print(
            "WARNING: No data.yaml found. Using hard-coded class map: "
            "{0: 'smoke', 1: 'fire'} (verified against D-Fire documented category counts). "
            "See docs/M2_DATA_EDA.md for details."
        )

    # Discover images
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths: list[Path] = []
    for ext in image_extensions:
        image_paths.extend(raw_root.rglob(f"*{ext}"))
        image_paths.extend(raw_root.rglob(f"*{ext.upper()}"))

    image_paths = sorted(set(image_paths))
    total_found = len(image_paths)
    print(f"Found {total_found} image files under {raw_root}")

    if max_images is not None:
        image_paths = image_paths[:max_images]
        print(f"Limiting to {max_images} images (--max-images).")

    rows: list[dict] = []
    skipped = 0
    empty_labels = 0

    for i, img_path in enumerate(image_paths):
        if (i + 1) % 1000 == 0:
            print(f"  Processing {i + 1}/{len(image_paths)}...")
        row, is_empty = _build_row(img_path, raw_root)
        if row is None:
            skipped += 1
        else:
            rows.append(row)
            if is_empty:
                empty_labels += 1

    df = pd.DataFrame(rows)

    # Type coercion
    bool_cols = ["has_label", "has_fire", "has_smoke"]
    int_cols = [
        "num_fire_boxes", "num_smoke_boxes", "total_boxes", "image_width", "image_height",
        "fire_thirds_col", "fire_thirds_row", "smoke_thirds_col", "smoke_thirds_row",
    ]
    float_cols = [
        "mean_bbox_area", "median_bbox_area", "max_bbox_area", "mean_bbox_aspect_ratio",
        "fire_mean_bbox_area", "smoke_mean_bbox_area", "fire_bbox_coverage", "smoke_bbox_coverage",
        "mean_brightness", "dark_pixel_ratio", "color_std_mean",
    ]

    for c in bool_cols:
        df[c] = df[c].astype(bool)
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["image_category"] = df["image_category"].astype(str)
    df["split"] = df["split"].astype(str)
    df["source_dataset"] = df["source_dataset"].astype(str)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["image_id"])
    after = len(df)
    if before != after:
        print(f"Removed {before - after} duplicate image_id rows.")

    df = df.reset_index(drop=True)

    # Save
    df.to_csv(output, index=False)
    print(f"\nSaved metadata to {output}  ({len(df)} rows)")

    # Summary (ASCII only — safe for all Windows console encodings)
    sep = "-" * 52
    print(f"\n{sep}")
    print("  Summary")
    print(sep)
    print(f"  Total rows           : {len(df)}")
    print(f"  Skipped/corrupt      : {skipped}")
    print(f"  Missing labels       : {(~df['has_label']).sum()}")
    print(f"  Empty label files    : {empty_labels}")
    print("\n  Split counts:")
    for split_name, cnt in df["split"].value_counts().items():
        print(f"    {split_name:<12} {cnt}")
    print("\n  image_category counts:")
    for cat, cnt in df["image_category"].value_counts().items():
        print(f"    {cat:<20} {cnt}")
    print(f"\n  Total fire boxes     : {df['num_fire_boxes'].sum()}")
    print(f"  Total smoke boxes    : {df['num_smoke_boxes'].sum()}")
    print(sep)

    if copy_samples_dir is not None:
        copy_samples(df, Path(copy_samples_dir), sample_count=sample_count)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build D-Fire metadata CSV from local raw dataset."
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Path to the D-Fire root directory (contains train/ and/or test/ subdirs).",
    )
    parser.add_argument(
        "--output",
        default="data/dfire_metadata.csv",
        help="Output path for the metadata CSV (default: data/dfire_metadata.csv).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional: limit total images processed (for quick testing).",
    )
    parser.add_argument(
        "--copy-samples",
        default=None,
        metavar="SAMPLES_DIR",
        help="If set, copy up to --sample-count images with drawn boxes to this directory.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=20,
        help="Number of sample images to copy (default: 20). Requires --copy-samples.",
    )

    args = parser.parse_args()
    build_metadata(
        raw_root=Path(args.raw_root),
        output=Path(args.output),
        max_images=args.max_images,
        copy_samples_dir=Path(args.copy_samples) if args.copy_samples else None,
        sample_count=args.sample_count,
    )


if __name__ == "__main__":
    main()

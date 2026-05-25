# M2 Data & EDA — PyroFinder

**Course:** Technion 016833 — Location-Based Services: Data Science
**Milestone:** M2 checkpoint (26/05/2026)
**Team:** Lisa + Boris

---

## Goal for 26/05

- Load D-Fire images and YOLO label files from local storage.
- Generate a clean CSV metadata file (`data/dfire_metadata.csv`).
- Display a working Streamlit EDA page with metrics, charts, filters, a table preview, and one written insight.
- Keep all existing tests passing.

---

## Dataset Location

| Item | Path |
|---|---|
| Raw D-Fire root | `C:\Users\boris.azarov\OneDrive - Technion\Desktop\PyroFinder\RAW_DATA\D-Fire` |
| Generated metadata CSV | `data/dfire_metadata.csv` (tracked by Git) |
| Optional sample images | `data/samples/dfire/images/` (not tracked by Git) |

The raw D-Fire images and labels are **never committed to Git**.

---

## Dataset Layout (observed after full ZIP extraction)

```
D-Fire/
├── train/
│   ├── images/     17,221 images
│   └── labels/     17,221 YOLO .txt label files
└── test/
    ├── images/      4,306 images
    └── labels/      4,306 YOLO .txt label files
Total: 21,527 images — matches official D-Fire documentation.
```

> **Note:** A partial ZIP extraction performed earlier was missing `train/labels/`. The metadata was regenerated after the full extraction was confirmed. The partial CSV (if it existed) was overwritten.

---

## Class Mapping

No `data.yaml` or `classes.txt` was found in this D-Fire download.
The class mapping was **verified empirically** by cross-checking scan results with official D-Fire category counts.

### Verification

The official D-Fire documentation reports:

| Category | Images |
|---|---|
| fire-only | 1,164 |
| smoke-only | 5,867 |
| fire+smoke | 4,658 |
| background | 9,838 |

A first scan using `{0: fire, 1: smoke}` produced `fire_only=5,867` and `smoke_only=1,164` — the exact inverse of documentation.
Swapping the mapping to `{0: smoke, 1: fire}` produces counts that match the documentation exactly.

### Confirmed class mapping

| Class ID | Class Name |
|---|---|
| 0 | smoke |
| 1 | fire |

This mapping is defined as the constant `DFIRE_CLASS_MAP` at the top of `scripts/build_dfire_metadata.py`.

---

## How to Generate the Metadata CSV

From the repository root:

```bash
python scripts/build_dfire_metadata.py \
  --raw-root "C:\Users\boris.azarov\OneDrive - Technion\Desktop\PyroFinder\RAW_DATA\D-Fire" \
  --output data/dfire_metadata.csv
```

To also copy annotated sample images (with bounding boxes drawn):

```bash
python scripts/build_dfire_metadata.py \
  --raw-root "C:\Users\boris.azarov\OneDrive - Technion\Desktop\PyroFinder\RAW_DATA\D-Fire" \
  --output data/dfire_metadata.csv \
  --copy-samples data/samples/dfire \
  --sample-count 20
```

The script always overwrites the output CSV — it is safe to rerun after a partial extraction.

---

## Actual Metadata Counts (regenerated after full extraction)

| Metric | Value |
|---|---|
| Total rows | 21,527 |
| Duplicate image_id rows | 0 |
| Missing label files | 0 |
| Empty label files (0 boxes) | 9,838 |
| Skipped / corrupt images | 0 |

**Split counts:**

| Split | Images |
|---|---|
| train | 17,221 |
| test | 4,306 |

**Image category counts:**

| Category | Images |
|---|---|
| background | 9,838 |
| smoke_only | 5,867 |
| fire_and_smoke | 4,658 |
| fire_only | 1,164 |

**Bounding box totals:**

| Class | Boxes |
|---|---|
| fire | 14,692 |
| smoke | 11,865 |

These counts match the official D-Fire documentation exactly.

---

## Expected CSV Columns

| Column | Type | Description |
|---|---|---|
| `image_id` | str | Filename stem (no extension) |
| `split` | str | `train`, `test`, or `unknown` |
| `image_path` | str | Absolute path to the image file |
| `label_path` | str | Absolute path to the label file (empty if none) |
| `has_label` | bool | True if a matching YOLO label file was found |
| `has_fire` | bool | True if at least one fire bounding box exists |
| `has_smoke` | bool | True if at least one smoke bounding box exists |
| `image_category` | str | `fire_only`, `smoke_only`, `fire_and_smoke`, or `background` |
| `num_fire_boxes` | int | Count of fire bounding boxes |
| `num_smoke_boxes` | int | Count of smoke bounding boxes |
| `total_boxes` | int | Total bounding boxes (fire + smoke) |
| `mean_bbox_area` | float | Mean normalized bounding box area (w×h) |
| `median_bbox_area` | float | Median normalized bounding box area |
| `max_bbox_area` | float | Max normalized bounding box area |
| `mean_bbox_aspect_ratio` | float | Mean bounding box w/h ratio |
| `image_width` | int | Image width in pixels (from Pillow) |
| `image_height` | int | Image height in pixels (from Pillow) |
| `source_dataset` | str | Always `D-Fire` for this dataset |

---

## Cleaning Decisions

| Situation | Treatment |
|---|---|
| Image with no matching label file | `has_label=False`, category=`background`, all box counts = 0 |
| Image with an empty label file (0 bytes or blank) | `has_label=True`, but boxes=[], category=`background`, bbox stats = 0 |
| YOLO line with unknown class ID | Logged as a warning; box is skipped |
| Corrupt image (Pillow cannot open) | Skipped with a warning; counted in "skipped" total |
| Duplicate `image_id` rows | Removed; first occurrence kept |
| Background images bbox statistics | Filled with 0.0 (not NaN) |
| `valid` split name | Normalized to `val` |

---

## What the Streamlit Dashboard Shows (26/05)

- **6 metrics:** total images, fire images, smoke images, background images, mean boxes/image, median boxes/image.
- **Chart 1:** Bar chart — image count by category (fire_only / smoke_only / fire_and_smoke / background).
- **Chart 2:** Histogram — bounding box count per image (for images with at least one box).
- **Chart 3:** Bar chart — image count per split (train / test).
- **Sidebar filters:** image category (multiselect), split (multiselect), has_fire (All/Yes/No), has_smoke (All/Yes/No).
- **Table preview:** 20 rows with key columns.
- **EDA insight:** written data-driven observation.
- **Sample images expander:** shows up to 20 annotated images if `data/samples/dfire/images/` exists.

---

## Current EDA Insight (full dataset)

The D-Fire dataset (21,527 images) is heavily class-imbalanced: 9,838 images (45.7%) are background with no fire or smoke annotations. Smoke-only images (5,867; 27.3%) are far more common than fire-only images (1,164; 5.4%), and fire-and-smoke images make up 21.6% (4,658). Fire bounding boxes (14,692) outnumber smoke bounding boxes (11,865) per image on average, meaning fire regions tend to be more numerous when they do appear. This imbalance means future model evaluation must prioritize fire recall: a model that defaults to predicting "background" or "smoke-only" would have high accuracy but would miss the most critical detections. Weighted loss or oversampling of fire_only and fire_and_smoke categories should be considered during YOLO11s fine-tuning.

---

## What Is Intentionally NOT Done in M2

- No YOLO11s training or fine-tuning.
- No YOLO11n baseline run or mAP computation.
- No deployment or live RTSP ingestion.
- No additional datasets (HPWREN, FURG, Smart Fire, Kaggle, etc.).
- No automatic image-to-map registration.

These items are planned for M3 and beyond.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `data/dfire_metadata.csv not found` | Run the metadata script above |
| `streamlit run app.py` crashes on import | Run `python -m pytest` and check `requirements.txt` |
| `FileNotFoundError` for raw images | Check that D-Fire is fully extracted to the path above |
| Corrupt image warnings | Normal for a small number of files; they are skipped automatically |
| Wrong class names in CSV | Update `DFIRE_CLASS_MAP` in `scripts/build_dfire_metadata.py` |
| UnicodeEncodeError in summary print | Fixed — script now uses ASCII-only separators |

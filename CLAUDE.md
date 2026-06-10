# PyroFinder — Claude Code Context

## Project Goal

PyroFinder is a real-time fire outbreak detection and monitoring system using cameras that already exist at the customer site. It detects fire and smoke, estimates the approximate event location, generates alerts, and supports ongoing operational monitoring and model improvement — without requiring new hardware.

## Product Scope

- **Operations & Learning Dashboard** — primary course MVP deliverable. Dataset inspection, EDA, inference, model comparison, experiment tracking, alert log.
- **Central Control Dashboard (basic)** — part of course MVP. Camera metadata table, basic map, image polygon setup, alert history.
- **Mobile Customer App** — future / out of scope this semester.
- **Emergency Viewer Dashboard** — future / optional. Not part of first MVP.

## ML Scope

- **Primary model:** Ultralytics YOLO11s (`yolo11s.pt`), fine-tuned on D-Fire.
- **Baseline / fallback:** Ultralytics YOLO11n (`yolo11n.pt`). Speed baseline only — not an equal parallel model.
- **Classes:** `fire` and `smoke` only. No other classes.
- **Task:** two-class object detection (bounding box + class + confidence).
- Do not train any class other than `fire` and `smoke`.
- Do not use YOLOv12.
- Always specify YOLO11s or YOLO11n — do not write generic "YOLO."

## Data Sources

- **D-Fire** — primary training and held-out test evaluation dataset. URL: https://github.com/gaia-solutions-on-demand/DFireDataset. 21,527 images. CC0 1.0.
- **Smart Fire System Dataset** — supplementary training / external validation.
- **Aerial Rescue OD** — robustness validation. Use Fire class only; Vehicle/Human as background negatives.
- **Fire Detection in YOLO Format** — supplementary after class verification. Small dataset.
- **FURG Fire Dataset** — video validation for temporal behavior and multi-frame tracking.
- All large datasets stay outside Git. Normalize all labels to `fire` / `smoke` before use.

## Mapping / Geo Scope

- Mapping is an **offline, pre-event setup stage** — not solved during a live fire event.
- Geo data is **operational configuration**, not YOLO training data.
- Supported: image polygons (named zones), map polygons, reference points (image ↔ map), camera GPS + metadata.
- **All location outputs are approximate.** Never claim precise geolocation.
- Automatic image-to-map registration is a future feature — not required for course MVP.

## Repository Structure

<!-- Updated 2026-06-09: added YOLO11n baseline results, scripts/YOLO11n_baseline.py, models/ (local only) -->
<!-- Updated 2026-06-10: added src/evaluation.py + cost-sensitive operational alert metrics, scripts/evaluate_yolo_alert_metrics.py, tests/test_evaluation.py -->

```
app.py                              — Streamlit entry point (multi-tab shell)
src/data.py                         — dataset loading, inspection, Data Card utilities
src/eda.py                          — EDA helpers: summary metrics, category/split counts, bbox stats, pixel stats, spatial analysis
src/viz.py                          — on-the-fly YOLO box annotation (D-Fire class map: 0=smoke, 1=fire)
src/ui.py                           — shared UI palette, CAT_COLORS, CLASS_COLORS, apply_chart_theme()
src/model.py                        — model metadata, metrics plan, evaluation helpers
src/detection.py                    — DetectionResult dataclass, class validation
src/tracking.py                     — multi-frame confirmation, apparent direction estimation
src/mapping.py                      — mapping modes, polygon helpers, approximate location formatting
src/alerts.py                       — alert record creation, status validation
src/evaluation.py                   — cost-sensitive operational alert metrics (hazard recall, false alert rate, operational alert score) + approximate fire-location helpers; pure stdlib, no ML imports
scripts/build_dfire_metadata.py     — generates data/dfire_metadata.csv from raw D-Fire root
scripts/dummy_try.py                — M3 sklearn baseline: full D-Fire loading, feature extraction, DummyClassifier
scripts/simple_baselines.py         — M3: Logistic Regression and Random Forest classifiers on D-Fire (+ operational_metrics block + prediction CSVs)
scripts/YOLO11n_baseline.py         — M3: YOLO11n object-detection baseline runner (reproducible; final run on Kaggle)
scripts/evaluate_yolo_alert_metrics.py — M3: evaluation-only operational alert + approximate fire-location metrics for a YOLO checkpoint on D-Fire test (no training)
results/baseline_dummy_classifier.json     — DummyClassifier metrics + operational_metrics (full D-Fire, 2026-06-05)
results/baseline_logistic_regression.json  — Logistic Regression metrics + operational_metrics (full D-Fire)
results/baseline_random_forest.json        — Random Forest metrics + operational_metrics (full D-Fire)
results/baseline_yolo11n.json              — YOLO11n detection metrics (mAP, P, R, F1; Kaggle, 2026-06-09)
results/results_yolo11n.csv                — YOLO11n per-epoch training curves
results/yolo11n_operational_metrics.json   — YOLO11n operational alert + location metrics (generated by evaluate_yolo_alert_metrics.py)
results/predictions_*.csv                  — per-image alert prediction tables (generated on demand)
models/                             — local only; Git-ignored. Contains yolo11n_dfire_best.pt.
tests/test_smoke.py                 — import smoke tests, unit tests for core helpers
tests/test_evaluation.py            — unit tests for src/evaluation.py (alert confusion, cost weighting, location helpers)
docs/M2_DATA_EDA.md                 — data workflow, class mapping, cleaning decisions, actual counts
docs/M2_dashboard.md                — dashboard design notes
docs/M2_GAP_LIST.md                 — known gaps and open items as of M2
docs/AI_AGENT_SYSTEM.md             — AI agent architecture notes
docs/Literature_review.md           — literature and related work
docs/market_survey_wildfire_existing_sensors.md — competitor / market landscape
```

## Coding Conventions

- Python 3.10+. Type hints where useful.
- Small, testable functions. No monolithic scripts.
- No secrets in code. No large files in Git.
- Prefer simple Streamlit MVP over premature abstractions.
- Functions must be importable without loading heavy ML models unless explicitly called.
- Clear English variable names. Comments only when the why is non-obvious.

## Current MVP Priority

<!-- Updated 2026-06-09: sklearn classifiers done, YOLO11n baseline done -->

1. ~~Streamlit shell running without errors~~ ✓ Done (M2)
2. ~~Dataset inspection and metadata display~~ ✓ Done (M2)
3. ~~Basic EDA — class distribution, bounding box statistics, image samples~~ ✓ Done (M2)
4. ~~Uploaded image/video inference placeholder~~ ✓ Done (M2)
5. ~~sklearn baseline pipeline — full D-Fire loading, feature extraction (60-dim), DummyClassifier~~ ✓ Done (M3 start, 2026-06-05)
6. ~~Real sklearn classifiers vs baseline — Logistic Regression, Random Forest~~ ✓ Done (M3, 2026-06-05)
7. YOLO11s model loading / fine-tuning — **next** (requires fine-tuned checkpoint on D-Fire)
8. ~~YOLO11n baseline benchmark~~ ✓ Done (M3, Kaggle, 2026-06-09) — see M3 YOLO11n section below
9. Alert log from test runs
10. Camera metadata table
11. Manual image polygon and map linking placeholders

## Data — M3 Status

- D-Fire raw data is local and outside Git (path varies per machine).
- Full dataset: 21,527 images (train: 17,221 + test: 4,306), all with matching label files.
- `data/dfire_metadata.csv` is committed to Git. The app runs fully on a fresh clone using only this CSV — no local raw dataset required.
- `data/samples/dfire/images/` — 20 committed sample images (~1.1 MB); `data/samples/dfire/labels/` — matching YOLO label files. These are used as the fallback when local raw D-Fire paths are unavailable.
- `docs/M2_DATA_EDA.md` documents the data workflow, class mapping, cleaning decisions, and actual counts.
- **D-Fire class mapping (verified):** class 0 = smoke, class 1 = fire. Confirmed by comparing scan results against official category counts.
- To re-generate `data/dfire_metadata.csv` from raw D-Fire: `python scripts/build_dfire_metadata.py --raw-root "<path-to-D-Fire-root>" --output data/dfire_metadata.csv`

### M3 Sklearn Baseline Pipeline

- `scripts/dummy_try.py` — loads the full D-Fire dataset using its pre-existing train/test split. Falls back to `data/samples/dfire/` on machines without raw data.
- Feature vector: 60 values per image — RGB mean+std (6), HSV mean+std (6), color histogram 16-bin×3 channels (48). Images resized to 64×64.
- Image-level label derived from YOLO boxes: fire if class 1 present, smoke if class 0 only, background if empty label file.
- **DummyClassifier baseline results (full D-Fire, 2026-06-05):**
  - Train: 17,221 images — background 7,833 / fire 4,707 / smoke 4,681
  - Test: 4,306 images — background 2,005 / fire 1,115 / smoke 1,186
  - Accuracy: 0.47 · F1 macro: 0.21 · fire recall: 0.00 · smoke recall: 0.00
- `scripts/simple_baselines.py` — Logistic Regression and Random Forest classifiers on the same 60-dim feature vector.
- **Logistic Regression:** ~0.61 accuracy · ~0.62 F1 macro · fire and smoke recall > 0.
- **Random Forest:** ~0.86 accuracy · ~0.85 F1 macro · strongest classical baseline.
- All three saved to `results/` as separate JSON files for model comparison.

### M3 YOLO11n Object-Detection Baseline

YOLO11n is the lightweight **object-detection baseline and fallback** for PyroFinder.
It is **not** an image-level classifier and must **not** be compared to sklearn accuracy.
Evaluation uses detection metrics: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1.

- **Training platform:** Kaggle Notebook, Tesla T4 GPU
- **Dataset:** D-Fire — train: 17,221 images · test: 4,306 images
- **Classes:** 0 = smoke, 1 = fire
- **Image size:** 640 px · **Epochs:** 30 · **Batch:** 16
- **Final metrics (epoch 30):**
  - mAP@0.5: **0.747**
  - mAP@0.5:0.95: 0.4249
  - Precision: 0.7397
  - Recall: 0.6825
  - F1: 0.7099
- Result JSON: `results/baseline_yolo11n.json`
- Training curve CSV: `results/results_yolo11n.csv`
- Local ignored checkpoint: `models/yolo11n_dfire_best.pt`
- Reproducible runner: `scripts/YOLO11n_baseline.py`

YOLO11n is the **baseline**. YOLO11s remains the planned primary detector.
YOLO11s should be compared to YOLO11n using detection metrics, not to sklearn classifiers.

## Future Model Layer — SAM 3.1

- SAM 3.1 (Meta / Facebook Research, March 2026) is a segmentation model, not a detector.
- Intended role: second stage after YOLO11s detection.
  - Pipeline: YOLO11s bbox → SAM 3.1 mask refinement → polygon for mapping / tracking.
- Object Multiplex (SAM 3.1 feature) enables multi-object video tracking — relevant future upgrade for `src/tracking.py`.
- SAM 3.1 text-prompt video segmentation is a candidate for `src/mapping.py` polygon auto-suggestion.
- SAM 3.1 does NOT replace YOLO11s for fire/smoke classification — it has no built-in class labels.
- Not part of M2 or the basic M3 scope. Document here for planning; do not add implementation yet.
- Checkpoint: `facebook/sam3.1` on Hugging Face.

## Forbidden / Out of Scope

- No emergency dispatch integration
- No full mobile app implementation
- No live RTSP production streaming
- No true physical fire-spread simulation or prediction
- No fully automatic image-to-map registration
- No precise geolocation claims
- No dedicated hardware assumptions
- Do not call PyroFinder an "early warning system"

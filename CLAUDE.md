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

```
app.py              — Streamlit entry point (multi-tab shell)
src/data.py         — dataset loading, inspection, Data Card utilities
src/model.py        — model metadata, metrics plan, evaluation helpers
src/detection.py    — DetectionResult dataclass, class validation
src/tracking.py     — multi-frame confirmation, apparent direction estimation
src/mapping.py      — mapping modes, polygon helpers, approximate location formatting
src/alerts.py       — alert record creation, status validation
tests/test_smoke.py — import smoke tests, unit tests for core helpers
```

## Coding Conventions

- Python 3.10+. Type hints where useful.
- Small, testable functions. No monolithic scripts.
- No secrets in code. No large files in Git.
- Prefer simple Streamlit MVP over premature abstractions.
- Functions must be importable without loading heavy ML models unless explicitly called.
- Clear English variable names. Comments only when the why is non-obvious.

## Current MVP Priority

1. Streamlit shell running without errors
2. Dataset inspection and metadata display
3. Basic EDA — class distribution, bounding box statistics, image samples
4. Uploaded image/video inference placeholder
5. YOLO11s model loading (later, requires dataset)
6. YOLO11n baseline benchmark (later)
7. Alert log from test runs
8. Camera metadata table
9. Manual image polygon and map linking placeholders

## Data — M2 Status

- D-Fire raw data is local and outside Git: `C:\Users\boris.azarov\OneDrive - Technion\Desktop\PyroFinder\RAW_DATA\D-Fire`
- Full dataset: 21,527 images (train: 17,221 + test: 4,306), all with matching label files.
- `data/dfire_metadata.csv` is the committed working dataset for M2. Generate with `scripts/build_dfire_metadata.py`.
- `docs/M2_DATA_EDA.md` documents the data workflow, class mapping, cleaning decisions, and actual counts.
- **D-Fire class mapping (verified):** class 0 = smoke, class 1 = fire. Confirmed by comparing scan results against official category counts.
- M2 focuses on descriptive EDA only — no YOLO11s training, no YOLO11n baseline run, no deployment.

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

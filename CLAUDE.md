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

<!-- Updated 2026-06-05: added scripts/, docs/ listing -->

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
scripts/build_dfire_metadata.py     — generates data/dfire_metadata.csv from raw D-Fire root
tests/test_smoke.py                 — import smoke tests, unit tests for core helpers
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

<!-- Updated 2026-06-05: items 1–4 completed at M2 -->

1. ~~Streamlit shell running without errors~~ ✓ Done (M2)
2. ~~Dataset inspection and metadata display~~ ✓ Done (M2)
3. ~~Basic EDA — class distribution, bounding box statistics, image samples~~ ✓ Done (M2)
4. ~~Uploaded image/video inference placeholder~~ ✓ Done (M2)
5. YOLO11s model loading (M3 — requires fine-tuned checkpoint)
6. YOLO11n baseline benchmark (M3)
7. Alert log from test runs
8. Camera metadata table
9. Manual image polygon and map linking placeholders

## Data — M2 Status

- D-Fire raw data is local and outside Git (path varies per machine).
- Full dataset: 21,527 images (train: 17,221 + test: 4,306), all with matching label files.
- `data/dfire_metadata.csv` is committed to Git. The app runs fully on a fresh clone using only this CSV — no local raw dataset required.
- `data/samples/dfire/images/` — 20 committed sample images (~1.1 MB); `data/samples/dfire/labels/` — matching YOLO label files. These are used as the fallback when local raw D-Fire paths are unavailable.
- `docs/M2_DATA_EDA.md` documents the data workflow, class mapping, cleaning decisions, and actual counts.
- **D-Fire class mapping (verified):** class 0 = smoke, class 1 = fire. Confirmed by comparing scan results against official category counts.
- M2 focuses on descriptive EDA only — no YOLO11s training, no YOLO11n baseline run, no deployment.
- To re-generate `data/dfire_metadata.csv` from raw D-Fire: `python scripts/build_dfire_metadata.py --raw-root "<path-to-D-Fire-root>" --output data/dfire_metadata.csv`

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

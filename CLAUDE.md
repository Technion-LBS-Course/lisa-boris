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
<!-- Updated 2026-06-12: added src/results_loader.py + src/inference.py for YOLO11s integration -->
<!-- Updated 2026-06-13: YOLO11s fine-tuning + detection/operational evaluation complete; measured YOLO11s result files added; YOLO11s is now the selected primary detector -->

```
app.py                              — Streamlit entry point (thin shell: page config, theme, sidebar mode select, dispatch to dashboard renderers)
src/data.py                         — dataset loading, inspection, Data Card utilities
src/eda.py                          — EDA helpers: summary metrics, category/split counts, bbox stats, pixel stats, spatial analysis
src/viz.py                          — on-the-fly YOLO box annotation (D-Fire class map: 0=smoke, 1=fire)
src/ui.py                           — shared UI palette, CAT_COLORS, CLASS_COLORS, apply_chart_theme()
src/model.py                        — model metadata, metrics plan, evaluation helpers
src/detection.py                    — DetectionResult dataclass, class validation
src/tracking.py                     — multi-frame confirmation, apparent direction estimation
src/mapping.py                      — mapping modes, polygon helpers, approximate location formatting
src/alerts.py                       — alert record creation, status validation
src/evaluation.py                   — operational alert metrics (hazard recall, false alert rate, alert precision, alert F1, alert F2 [primary]) + approximate fire-location helpers; pure stdlib, no ML imports
src/results_loader.py               — load/classify detection vs operational result JSON (status: ok / training_in_progress / malformed / wrong-kind) + Alert F2-based winner selection; pure stdlib, no ML imports
src/inference.py                    — lazy YOLO11n/YOLO11s detector loading (ultralytics imported inside functions only) + single-image detection; fine-tuned D-Fire checkpoints only, never pretrained weights; validates fire/smoke-only classes
src/llm.py                          — Groq LLM helper for operational text + zone-setup assist (NOT the fire detector — that stays YOLO). Reads GROQ_API_KEY from st.secrets/env, uses OS cert store via truststore; `groq` is imported lazily inside get_client() so the module imports without the package. extract_zones() structures a free-text area description into zone records (name/type/priority/alert-label, no coordinates); extract_operational_zones() returns richer raw records (adds object_to_find + low/medium/high priority) for src/zone_agent.py to sanitize; detect_zone_boxes() asks a Groq vision model (Llama 4 Scout) for APPROXIMATE normalized ROI boxes per named area (rough, verify-only — NOT real detection); sanitize_zone_records()/parse_box_norm() are pure/testable.
src/agent_schemas.py                — shared vocabulary for the operational agents: priority normalization (low/medium/high ↔ 1-10 int), image-zone-type inference, prompt-injection/off-intent detector, compass/downwind helpers. Pure, no Streamlit/ML.
src/zone_agent.py                   — Setup/Configuration Agent: parse_zone_description() turns free-text area descriptions into operational image-zone records (object_to_find, zone_name, low/medium/high priority, requires_user_confirmation) via Groq when a key exists, else a deterministic local parser; prompt-injection filtered; build_zone_table_entry() supports pending_manual_polygon. object_to_find is a zone target, never a detector class. Pure except a lazy llm call.
src/segmentation_assist.py          — Image-Zones setup tooling: refine a rough ROI box (from Groq Vision, a manual box, or clicked-point bounds) into a cleaner editable polygon via LOCAL OpenCV GrabCut (box-prompted; NO model weights, NO network, NEVER calls YOLO11s or Groq). validate_roi_box()/polygon_from_box_fallback()/mask_to_polygon()/refine_box_to_mask(); box schema {x_min,y_min,x_max,y_max} with box_norm_from_xyxy() adapter for the Groq Vision [x0,y0,x1,y1] list; polygon is normalized [{x,y},...]. Import-safe: cv2/numpy/PIL imported lazily inside functions; no Streamlit; runs only when called. Never raises except on a degenerate box (UI-guarded); on any failure returns ok=False so the UI offers the box-as-polygon fallback.
src/incident_agent.py               — Incident Assistant: build_incident_context() (matched image zone, approximate map point, apparent image-plane direction, optional downwind), recommend_actions(), draft messages (owner/neighbor/farm worker/fire-dept summary), create_incident_alert(); optional Groq wording polish. Drafts and recommends only — never contacts anyone or dispatches. Pure, no ML.
src/weather.py                      — Risk Advisory: Open-Meteo current-weather over stdlib urllib (NO API key) with a deterministic offline mock fallback; fire_weather_risk() + assess_risk() produce preventive advisories tied to configured zones. Normalized Weather (temperature_c, relative_humidity, wind_speed_kmh, wind_direction_deg, source, is_live); fetch_weather() validates coords, tries live, falls back to mock on any failure. Pure risk logic; no Streamlit/ML import; no network at import.
src/dashboards/                     — dashboard renderers; app.py dispatches one render() per dashboard mode
src/dashboards/model_helpers.py     — shared model/comparison rendering helpers (per-model views, classification/object-detection/operational comparisons) + cached detector loader; ML imports stay lazy
src/dashboards/operations_learning.py — Operations & Learning dashboard renderer (6 tabs)
src/dashboards/central_control.py   — Central Control dashboard renderer, six tabs: Camera Metadata, Map Reference Points, Image Zones, Export & Generate, Incident Assistant, Risk Advisory. Includes an "Import saved configuration (JSON)" uploader (inverse of export). Image Zones default to AI-assisted: "Structure from text" (src/zone_agent.parse_zone_description → operational zone records with object_to_find + low/medium/high priority; Groq-or-local; accept as pending or draw each polygon) and "Detect on image" (src/llm.detect_zone_boxes → approximate vision ROI boxes); a "Switch to manual drawing" button gives the click-to-draw flow. Once a rough ROI box exists (AI ROI, manual box, or the bounding box of clicked points), a "Selected ROI box" selector + "Refine selected box with segmentation" button runs src/segmentation_assist (local OpenCV GrabCut, on click only — not YOLO/Groq) to produce a cleaner polygon the operator accepts (loaded into the editor for save) or falls back to the box-as-polygon; wired into both the AI and manual panels. Incident Assistant uses src/incident_agent; Risk Advisory uses src/weather. The shared frame uploader also offers a "Demo: image sequence" loader (folder path — default TESTING/sequance — or multi-file upload) that resizes all frames to one common size and steps through them (Prev/Next/slider), driving cc_uploaded_image so Image Zones and the Incident Assistant run per selected frame — reuses run_detection, no YOLO changes. No module-level YOLO/torch imports.
src/dashboards/m2_dashboard.py      — M2 dashboard orchestrator (delegates to src/dashboards/m2/)
src/dashboards/m2/                  — M2 tab modules: problem_understanding, literature_review, market_review, dataset_eda
src/dashboards/m3_dashboard.py      — M3 dashboard orchestrator (delegates to src/dashboards/m3/)
src/dashboards/m3/                  — M3 tab modules: overview, models, model_comparison, inference_demo
src/dashboards/m4_dashboard.py      — M4 dashboard renderer; reuses four Central Control tabs (Camera Metadata, Image Zones, Incident Assistant, Risk Advisory) via src/dashboards/central_control, sharing its cc_* session state (only one mode renders per run, so widget keys never collide). M4 tailors the Incident Assistant with flags — _tab_incident_assistant(show_intro=False, allow_manual_point=False, sequence_view=True, show_drafts=False): hides the workflow blurb, the manual hazard-point fallback, and the Draft-messages section, and adds an in-tab demo-sequence loader + slider that auto-runs YOLO per selected frame and overlays boxes (cached in cc_seq_det). When a sequence is loaded the manual "Run YOLO11s" button is hidden and _auto_assess_sequence_frame() builds the incident (summary/conversation/recommendations) automatically as the slider moves — guarded by cc_incident_seq_idx so each frame assesses once. In M4 the sequence panel runs with drive_shared_frame=False: the incident slider drives only the incident's own per-frame detection, while Image Zones / Camera Metadata keep the stable reference frame (first sequence frame). M4's render() shows no top frame uploader or import-config panel — the camera frame comes only from the demo sequence loaded inside the Incident Assistant tab. M4 selects its four sections with a keyed st.segmented_control (not st.tabs) so the active section is stored in session state and survives every rerun (st.tabs has no key and resets to the first tab on reruns — uploads, sliders, etc.); only the active section renders. The shared sequence flow is also kept rerun-free where practical. No module-level ML imports.
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
results/yolo11n_operational_metrics.json   — YOLO11n operational alert + location metrics (Kaggle, 2026-06-10)
results/yolo11n_test_predictions.csv       — YOLO11n per-image alert outcome + fire-location error table (D-Fire test, used for failure analysis)
results/baseline_yolo11s.json              — YOLO11s detection metrics (mAP, P, R, F1; Kaggle, 2026-06-12; measured)
results/results_yolo11s.csv                — YOLO11s per-epoch training curves
results/yolo11s_operational_metrics.json   — YOLO11s operational alert + location metrics (Kaggle, 2026-06-12; measured)
results/yolo11s_test_predictions.csv       — YOLO11s per-image alert outcome + fire-location error table (D-Fire test, used for failure analysis)
results/predictions_*.csv                  — additional per-image alert prediction tables generated on demand
models/                             — yolo11n_dfire_best.pt (~5 MB) and yolo11s_dfire_best.pt (~19 MB) are committed (gitignore exceptions) so the public Streamlit Cloud app runs inference on a fresh clone. All other weights / *.pt files stay Git-ignored.
tests/test_smoke.py                 — import smoke tests, unit tests for core helpers
tests/test_evaluation.py            — unit tests for src/evaluation.py (alert confusion, cost weighting, location helpers)
tests/test_results_loader.py        — unit tests for src/results_loader.py (status classification, winner selection, pending/malformed handling) — temp files only, no weights
tests/test_inference.py             — unit tests for src/inference.py (checkpoint paths, class validation, missing-checkpoint guard) — no real weights, no ultralytics import
tests/test_llm.py                   — unit tests for src/llm.sanitize_zone_records + parse_box_norm (type-mapping, priority clamp/default, dropping unnamed/non-dict entries, box clamp/reorder/degenerate) — pure, no network/Groq call
tests/test_zone_agent.py            — unit tests for src/zone_agent + src/agent_schemas (parse, priority normalization, zone-type inference, prompt-injection filtering, local fallback, Groq branch via monkeypatch) — pure, no network
tests/test_segmentation_assist.py   — unit tests for src/segmentation_assist (import-safety/no heavy top-level imports, ROI box clamp/reorder/degenerate, box↔xyxy adapters, polygon fallback + pixel conversion, mask→polygon via OpenCV, controlled failure/fallback for bad image + unavailable backend) — cv2/numpy paths skip cleanly if absent
tests/test_incident_agent.py        — unit tests for src/incident_agent (context assembly, recommendations, draft wording/guardrails, alert record, compass/downwind) — pure, no network
tests/test_weather.py               — unit tests for src/weather (fire-weather scoring, advisory generation, mock determinism, provider selection with/without key, error fallback) — pure, no network
tests/test_dashboards_smoke.py      — dashboard import smoke tests (render() present; no ultralytics/torch imported at module import)
docs/M2_DATA_EDA.md                 — data workflow, class mapping, cleaning decisions, actual counts
docs/M2_dashboard.md                — dashboard design notes
docs/M2_GAP_LIST.md                 — known gaps and open items as of M2
docs/AI_AGENT_SYSTEM.md             — AI agent architecture notes
docs/Literature_review.md           — literature and related work
docs/market_survey_wildfire_existing_sensors.md — competitor / market landscape
```

## Central Control Operational Agents

The Central Control dashboard hosts three operator-facing agents across six tabs (Camera Metadata, Map Reference Points, Image Zones, Export & Generate, Incident Assistant, Risk Advisory). All three are text/config assistants — none run YOLO, and `central_control.py` keeps no module-level ML imports.

- **Setup / Configuration Agent** (Image Zones tab · `src/zone_agent.py`, `src/agent_schemas.py`): free-text → structured image-zone records (`object_to_find`, `zone_name`, low/medium/high priority, optional `zone_type`/`notes`, `requires_user_confirmation`). Uses Groq when `GROQ_API_KEY` exists, else a deterministic local parser (offline). Prompt-injection / off-intent lines are filtered; a missing name/target → a clarification request. Zones can be accepted as `pending_manual_polygon` and drawn later. `object_to_find` is a monitoring target, never a detector class — classes stay fire/smoke. **Segmentation-assisted polygon refinement** (`src/segmentation_assist.py`): after a rough ROI box exists (Groq Vision, a manual box, or the bounding box of clicked points), an on-click **local OpenCV GrabCut** segmentation refines it into a cleaner polygon the operator accepts (or falls back to the box-as-polygon). Setup tooling only — no model weights, no network, and it never calls YOLO11s or Groq; the fire/smoke detector stays YOLO11s.
- **Incident Assistant** (Incident Assistant tab · `src/incident_agent.py`): the operator runs the **YOLO11s** fire/smoke detector on the current frame (button; lazy load, YOLO11n fallback), which builds the incident — matched image zone, estimated map point (homography), apparent in-frame movement, and Open-Meteo weather/wind + downwind. Produces operational recommendations, an **operator chat** (free-form via Groq when `GROQ_API_KEY` is set — grounded in the incident facts by a constrained system prompt — with a deterministic keyword responder as offline fallback), and draft messages (owner / neighbor / farm worker / fire-department summary; worker messages use zone + task, never coordinates). Confirm / false-alarm actions write to a session alert log. Drafts and recommends only — never contacts anyone or dispatches automatically. Groq Vision (Image Zones) is not the detector; YOLO11s is.
- **Risk Advisory** (Risk Advisory tab · `src/weather.py`): a preventive, weather-aware advisory compared against configured zones/priorities. Weather comes from **Open-Meteo** (no API key required) with a deterministic offline mock fallback; if live weather is unavailable the UI shows a clear fallback/demo banner. The user sets a check interval; refresh is manual (Streamlit has no safe background scheduler). Advisory only — not an early-warning alert, ignition prediction, or dispatch.

Secrets: `GROQ_API_KEY` (AI zone setup + optional wording polish) is the only optional key — read from `st.secrets`/env only, never logged or committed; without it the zone agent uses its deterministic local parser. `groq` is imported lazily inside `src/llm.get_client()` so the app and its fallback import cleanly without the package. The Risk Advisory uses **Open-Meteo, which requires no API key**; if live weather is unavailable it falls back to a deterministic offline mock.

## Coding Conventions

- Python 3.10+. Type hints where useful.
- Small, testable functions. No monolithic scripts.
- No secrets in code. No large files in Git.
- Prefer simple Streamlit MVP over premature abstractions.
- Functions must be importable without loading heavy ML models unless explicitly called.
- Clear English variable names. Comments only when the why is non-obvious.

## Current MVP Priority

<!-- Updated 2026-06-09: sklearn classifiers done, YOLO11n baseline done -->
<!-- Updated 2026-06-12: cost-sensitive alert metrics done, YOLO11n operational alert evaluation done -->
<!-- Updated 2026-06-13: YOLO11s fine-tuning + detection/operational evaluation done; YOLO11s is the selected primary detector -->

1. ~~Streamlit shell running without errors~~ ✓ Done (M2)
2. ~~Dataset inspection and metadata display~~ ✓ Done (M2)
3. ~~Basic EDA — class distribution, bounding box statistics, image samples~~ ✓ Done (M2)
4. ~~Uploaded image/video inference placeholder~~ ✓ Done (M2)
5. ~~sklearn baseline pipeline — full D-Fire loading, feature extraction (60-dim), DummyClassifier~~ ✓ Done (M3 start, 2026-06-05)
6. ~~Real sklearn classifiers vs baseline — Logistic Regression, Random Forest~~ ✓ Done (M3, 2026-06-05)
7. ~~YOLO11n baseline benchmark~~ ✓ Done (M3, Kaggle, 2026-06-09) — see M3 YOLO11n section below
8. ~~Cost-sensitive operational alert metric implementation (`src/evaluation.py` + `tests/test_evaluation.py`)~~ ✓ Done (M3, 2026-06-10)
9. ~~YOLO11n operational alert evaluation~~ ✓ Done (M3, Kaggle, 2026-06-10) — see M3 YOLO11n Operational Evaluation section below
10. ~~YOLO11s fine-tuning + detection and operational evaluation~~ ✓ Done (M3, Kaggle, 2026-06-12) — measured; YOLO11s is now the selected primary detector. See M3 YOLO11s section below.
11. Alert log from test runs
12. Camera metadata table
13. Manual image polygon and map linking placeholders

> Central Control now also provides an AI-assisted zone-setup agent, an Incident Assistant (with a session alert log + confirm/false-alarm), and a weather-based Risk Advisory — see **Central Control Operational Agents** above. Camera metadata, reference points, and manual/AI image polygons are implemented in Central Control.

**Next result-analysis task:** Detailed analysis of YOLO11s and YOLO11n results — false negatives, false positives, hazard subtypes, confidence-threshold implications, and approximate location errors (using `results/yolo11s_test_predictions.csv` and `results/yolo11n_test_predictions.csv`) — before creating `docs/M3_RESULTS_SUMMARY.md`.

### M3 YOLO11s — measured results; selected primary detector (2026-06-12)

YOLO11s is the **current primary detector**. Fine-tuning and evaluation are **complete**, the measured result files exist in `results/`, and **no synthetic / placeholder values are used anywhere**. YOLO11n stays the lightweight speed baseline / fallback. The app loads these files as measured (status **Measured**); object-detection metrics and operational alert metrics stay in separate tables and are never mixed with sklearn Macro F1.

Measured result files:

```text
models/yolo11s_dfire_best.pt              — fine-tuned checkpoint (committed via gitignore exception, ~19 MB)
results/baseline_yolo11s.json             — object-detection metrics (mAP, P, R, F1)
results/results_yolo11s.csv               — per-epoch training curves
results/yolo11s_operational_metrics.json  — operational alert + approximate fire-location metrics
results/yolo11s_test_predictions.csv      — per-image alert outcome + fire-location error table
```

**YOLO11s object-detection metrics (D-Fire test split; image size 640, 30 epochs requested, batch 16, Kaggle Tesla T4):**

- mAP@0.5: **0.7668** (YOLO11n 0.7470)
- mAP@0.5:0.95: 0.4414 (YOLO11n 0.4249)
- Precision: 0.7573 (YOLO11n 0.7397)
- Recall: 0.6967 (YOLO11n 0.6825)
- F1: 0.7257 (YOLO11n 0.7099)
- Per-class: smoke — mAP@0.5 0.8222 / mAP@0.5:0.95 0.5054 / P 0.8028 / R 0.7563 / F1 0.7789; fire — mAP@0.5 0.7115 / mAP@0.5:0.95 0.3774 / P 0.7119 / R 0.6370 / F1 0.6724.

**YOLO11s operational alert metrics (D-Fire test split, evaluation only — no training; confidence 0.25, 4,306 images):**

- Alert-level confusion: TP 2,156 · FN 145 · FP 37 · TN 1,968
- Hazard Recall: **0.9370** (YOLO11n 0.9331)
- False Alert Rate: 0.0185 (YOLO11n 0.0209)
- Alert Precision: 0.9831 (YOLO11n 0.9808)
- Alert F1: 0.9595 (YOLO11n 0.9563)
- Alert F2-score (primary decision metric; F-beta with beta=2): **0.9459** (YOLO11n 0.9423)
- Approximate fire-location (bottom-center anchor of class-1 fire boxes; image-space only, never precise geolocation): coverage 1,040 / 1,115 (rate 0.9327) · mean error 0.013499 · median 0.005478 · 3×3 grid hit rate 0.9644.

**Selection outcome:** YOLO11s is the **selected detector**. Complete measured detection and operational result files make it *eligible*; what *selects* it is winning the measured operational comparison (`src/results_loader.select_operational_winner`): the primary decision metric is the higher Operational Alert Score, which already encodes its component Hazard Recall (driven by false negatives) and False Alert Rate (driven by false positives) at the documented 10:1 weight; YOLO11s also leads on those components (higher Hazard Recall, lower False Alert Rate) and on supporting detection Recall / mAP@0.5. Eligibility is gated on existing files, measured values, a non-synthetic/non-pending status, and non-null required metrics; without these files YOLO11s would not be eligible — but eligibility alone does not select it, the measured comparison does.

YOLO11s operational/location metrics are reproducible (evaluation only, no training) with:

```
python scripts/evaluate_yolo_alert_metrics.py --raw-root "<path-to-D-Fire-root>" --weights "models/yolo11s_dfire_best.pt" --model-name "YOLO11s" --conf 0.25 --output-json "results/yolo11s_operational_metrics.json" --output-csv "results/yolo11s_test_predictions.csv"
```

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
- Committed checkpoint (gitignore exception, ~5 MB): `models/yolo11n_dfire_best.pt`
- Reproducible runner: `scripts/YOLO11n_baseline.py`

YOLO11n is the **baseline / fallback**. YOLO11s is now the measured, selected primary detector (see M3 YOLO11s section below).
YOLO11s is compared to YOLO11n using detection metrics, not to sklearn classifiers.

### M3 YOLO11n Operational Evaluation

This is a cost-sensitive, image-level **alert** evaluation of the fine-tuned YOLO11n checkpoint — separate from, and complementary to, the object-detection metrics above. At the alert level, `fire` and `smoke` both count as a hazard, and each image is reduced to hazard detected / not detected. A missed hazard (false negative) is weighted **10×** a false alert (false positive). The standard YOLO11n detection metrics in `results/baseline_yolo11n.json` remain object-detection metrics; the two evaluations must not be presented as interchangeable.

- This was **inference/evaluation only** — no training or retraining occurred during this evaluation.
- It ran on the **full 4,306-image D-Fire test split**.
- Confidence threshold: **0.25**. Image size: 640.
- Produced on **Kaggle** with a **Tesla T4** GPU on **2026-06-10**.
- YOLO11n remains the lightweight **baseline/fallback**; YOLO11s is now the measured, selected **primary detector**.

Run configuration: FN weight 10 · FP weight 1.

Alert-level confusion (image-level):

- TP alert: 2,147 · FN alert: 154 · FP alert: 42 · TN alert: 1,963

Operational alert metrics:

- Hazard Recall: **0.9331**
- False Alert Rate: 0.0209
- Alert Precision: 0.9808
- Alert F1: 0.9563
- Weighted Error Cost: 1,582
- Operational Alert Score: **0.9368**

Approximate fire-location metrics (bottom-center anchor of class-1 fire boxes; `anchor_x = x_center`, `anchor_y = y_center + height/2`; image-space only, never precise geolocation):

- Ground-truth fire images: 1,115
- Location coverage: 1,020 / 1,115 (rate 0.9148)
- Mean fire location error: 0.01343 · Median: 0.005704
- 3×3 fire-location grid hit rate: 0.9559

Output files:

- `results/yolo11n_operational_metrics.json`
- `results/yolo11n_test_predictions.csv` — per-image alert outcome + fire-location error table for failure analysis
- Implementation: `src/evaluation.py` · Runner: `scripts/evaluate_yolo_alert_metrics.py` (evaluation only) · Tests: `tests/test_evaluation.py`

## Forbidden / Out of Scope

- No emergency dispatch integration
- No full mobile app implementation
- No live RTSP production streaming
- No true physical fire-spread simulation or prediction
- No fully automatic image-to-map registration
- No precise geolocation claims
- No dedicated hardware assumptions
- Do not call PyroFinder an "early warning system"

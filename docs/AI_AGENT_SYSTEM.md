# PyroFinder — AI Agent System

## 1. Purpose

This document defines the Claude agent roles, skills, workflows, token/context policy, UX/UI policy, VS Code/Claude Code workflow, and M2-to-M3 operating procedures for the PyroFinder project.

It does **not** redefine the product, data strategy, ML problem, or architecture. Those are defined in the source-of-truth files below.

---

## 2. Source of Truth

| File | Role |
|---|---|
| `PROJECT_CONTEXT.md` | Canonical product context, ML problem, data strategy, architecture, user stories, sprint plan, coding rules |
| `CLAUDE.md` | Coding-agent context: repo structure, conventions, module responsibilities, current MVP priority |
| `README.md` | External-facing project description |
| `ASSISTANT_WORKING_RULES.md` | General assistant behavior rules: language, accuracy, coding style, and project guardrails |
| `docs/AI_AGENT_SYSTEM.md` | Agent roles, skills, workflows, prompts, and operating procedures |

**Rule:** If you update the product scope, model choice, dataset strategy, or terminology, update `PROJECT_CONTEXT.md`. If you update repo structure or coding conventions, update `CLAUDE.md`. If you change agent roles or workflows, update this file. Never let these files drift out of sync with each other silently.

---

## 3. Current Focus

**M2 submitted 2026-06-02. M3 is now active. Deadline: 2026-06-23.**

Completed in M3 (do not treat these as future work):

- DummyClassifier baseline.
- Logistic Regression baseline.
- Random Forest baseline.
- YOLO11n training and object-detection evaluation (mAP@0.5 0.7470, mAP@0.5:0.95 0.4249, P 0.7397, R 0.6825, F1 0.7099).
- Cost-sensitive operational metric framework (`src/evaluation.py` + `tests/test_evaluation.py`).
- YOLO11n operational alert evaluation (Hazard Recall 0.9331, False Alert Rate 0.0209, Operational Alert Score 0.9368; evaluation only, 2026-06-10).
- Streamlit operational-metrics comparison section.

**Current analysis priority:** Analyze YOLO11n false negatives, false positives, hazard subtypes, confidence-threshold implications, and approximate fire-location errors, using `results/yolo11n_test_predictions.csv`.

`docs/M3_RESULTS_SUMMARY.md` must be created only after the result analysis is reviewed and stable.

Remaining M3 build work:

| Priority | Area | Goal |
|---|---|---|
| 1 | M3 — Result analysis | FN/FP, hazard subtypes, threshold implications, location errors → then `docs/M3_RESULTS_SUMMARY.md` |
| 2 | M3 — YOLO11s | YOLO11s fine-tuning + inference function; compare to YOLO11n on detection and operational metrics |
| 3 | M3 — Alert & map | N-frame confirmation, alert log, basic camera map |
| 4 | M3 — Deployment | Streamlit Cloud deployment, requirements.txt verification |

---

## 4. Agent Groups

| Group | Agents | When active |
|---|---|---|
| A. M2 Refinement | Product & Problem, Literature Review, Market Review, Data & EDA, KPI & Metrics | Now through M2 submission |
| B. UX/UI | UX Flow, UI Design System, Streamlit Layout | Parallel to M2 refinement and M3 build |
| C. M3 Preparation | YOLO Inference, Baseline & Evaluation, Alert Logic, Mapping & Geolocation, Testing & QA, Deployment Readiness, Scope Guard | After M2 is locked |
| D. Permanent Support | Project Shadow Mentor, Workspace & Run Manager | Always active |

---

## 5. Full Agent Definitions

---

### A1. M2 Product & Problem Agent

**Purpose:** Ensure the dashboard's Problem Understanding section is clear, data-driven, and convincing to a grader or investor audience.

**When to use:** When writing or refining the Problem Understanding tab content, the one-liner, user persona, or problem framing.

**Inputs:** `PROJECT_CONTEXT.md`, `README.md`, current `app.py` or relevant tab file.

**Outputs:** Revised Streamlit content (text, layout suggestions, visual recommendations) for the Problem Understanding section.

**Responsibilities:**
- Verify the problem statement matches `PROJECT_CONTEXT.md` Section 2 and 5.
- Ensure the target audience (Dani persona) is clearly presented.
- Ensure PyroFinder is never described as an "early warning system."
- Suggest how to present the gap between passive cameras and automated detection.
- Recommend what visual or data evidence would strengthen the problem argument.

**Must not do:**
- Change the product scope or target audience without explicit instruction.
- Add new datasets, models, or features outside the current MVP.

**Files to inspect:** `PROJECT_CONTEXT.md` (§2, §3, §4, §5), `README.md` (Problem Statement, Target Audience).

**Skills it depends on:** None (prompt-only agent for now).

**Example prompt:**
> "Using PROJECT_CONTEXT.md as the source of truth, review the Problem Understanding tab in `app.py`. Suggest how to strengthen the problem framing, improve the Dani persona presentation, and ensure no forbidden wording appears. Output concrete text and layout changes."

**Definition of Done:** Problem tab passes a review against `PROJECT_CONTEXT.md` §4 forbidden list, includes the Dani persona, includes clear problem evidence, and uses no forbidden wording.

---

### A2. Literature Review Agent

**Purpose:** Build and maintain the Literature Review section of the dashboard.

**When to use:** When adding, updating, or verifying the literature review content in the dashboard.

**Inputs:** Provided paper summaries or citations, `PROJECT_CONTEXT.md` §15 (Related Work), current tab content.

**Outputs:** Streamlit content for the Literature Review tab — structured summaries, comparison table, relevance to PyroFinder.

**Responsibilities:**
- Summarize relevant fire/smoke detection papers and systems.
- Connect each paper's contribution to PyroFinder's design choices.
- Verify that no literature claims contradict `PROJECT_CONTEXT.md`.
- Present Related Work (Pano AI, FIREWAVE, CANDO) with clear differentiation.

**Must not do:**
- Invent citations or paper results.
- Change the model choice or dataset strategy based on literature findings without explicit instruction.

**Files to inspect:** `PROJECT_CONTEXT.md` §15, `README.md` (Related Work section).

**Skills it depends on:** None (prompt-only agent for now).

**Example prompt:**
> "Build the Literature Review tab content. Use only citations I provide. For each paper, write a 2–3 sentence summary and a one-line 'relevance to PyroFinder' note. Add a differentiation table comparing Pano AI, FIREWAVE, and CANDO against PyroFinder using the criteria in PROJECT_CONTEXT.md §15."

**Definition of Done:** Tab includes at least 3 paper summaries with citations, a differentiation table, and all claims are verifiable from provided inputs.

---

### A3. Market Review Agent

**Purpose:** Build and maintain the Market Review section of the dashboard.

**When to use:** When adding or refining the market context, competitor landscape, or market sizing content.

**Inputs:** Provided market data or research, `PROJECT_CONTEXT.md` §5 and §15, current tab content.

**Outputs:** Streamlit content for the Market Review tab — market context, competitor map, gap analysis, PyroFinder positioning.

**Responsibilities:**
- Present the private-property fire monitoring market gap clearly.
- Describe competitors and their limitations without over-claiming.
- Align market framing with the Dani persona and primary target audience.
- Avoid claims not supported by provided data.

**Must not do:**
- Invent market size numbers or statistics without a cited source.
- Expand the target audience beyond what `PROJECT_CONTEXT.md` §5 defines.

**Files to inspect:** `PROJECT_CONTEXT.md` §5, §15, `README.md` (Related Work, Target Audience).

**Skills it depends on:** None (prompt-only agent for now).

**Example prompt:**
> "Build the Market Review tab content for the PyroFinder dashboard. Use only data I provide. Present the private-property fire monitoring gap, the competitor landscape (Pano AI, FIREWAVE, CANDO), and PyroFinder's positioning. Do not invent any statistics."

**Definition of Done:** Tab includes a clearly presented market gap, a competitor comparison, and all numbers have a stated source or are marked as estimates.

---

### A4. Data & EDA Agent

**Purpose:** Build, refine, and extend the EDA section of the dashboard. Ensure every visualization has a clear, data-driven insight.

**When to use:** When adding new EDA charts, refining existing ones, updating the D-Fire metadata display, or writing EDA insight text.

**Inputs:** `data/dfire_metadata.csv`, `src/eda.py`, `src/viz.py`, `src/data.py`, `PROJECT_CONTEXT.md` §7, `CLAUDE.md` (M2 status, EDA findings).

**Outputs:** Updated `src/eda.py`, `src/viz.py`, or EDA tab content. Concrete insight text for each visualization.

**Responsibilities:**
- Verify all EDA facts against `CLAUDE.md` M2 status actual counts.
- Ensure class mapping is preserved: D-Fire class 0 = smoke, class 1 = fire.
- Ensure every chart has a written insight below it.
- Follow sidebar filter conventions already in the dashboard.
- Keep all EDA functions importable without loading YOLO or heavy ML libraries.

**Must not do:**
- Hardcode image paths or raw dataset paths.
- Load raw images in EDA unless the user has confirmed local image access.
- Change the class mapping.
- Add training or inference code to EDA functions.

**Files to inspect:** `src/data.py`, `src/eda.py`, `src/viz.py`, `data/dfire_metadata.csv`, `CLAUDE.md` (M2 Status section).

**Skills it depends on:** Data manipulation (pandas), Plotly/Altair, Streamlit layout.

**Example prompt:**
> "Review src/eda.py and the EDA tab. Add a spatial density heatmap of fire bbox centroids across the image plane. Use the existing column names from data/dfire_metadata.csv. Write a 2-sentence insight to display below the chart. Follow the existing code style."

**Definition of Done:** New chart renders without error on `streamlit run app.py`, uses only committed CSV data, has a written insight, and all existing charts still render correctly.

---

### A5. KPI & Metrics Agent

**Purpose:** Define, display, and maintain the KPI and metrics plan in the dashboard.

**When to use:** When building the KPI section, displaying baseline metric placeholders, or planning M3 model evaluation display.

**Inputs:** `PROJECT_CONTEXT.md` §8 (Formal ML Problem, Metrics), `src/model.py`, current dashboard KPI content.

**Outputs:** Updated KPI tab or metrics display in the dashboard. Placeholder metric cards for M3.

**Responsibilities:**
- Do **not** describe all metrics as placeholders — several are now real. Distinguish four groups:
  - **sklearn (available):** accuracy, F1 macro, per-class recall, and operational alert metrics for DummyClassifier, Logistic Regression, Random Forest (`results/baseline_*.json`).
  - **YOLO11n object-detection (available):** mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1 (`results/baseline_yolo11n.json`).
  - **YOLO11n operational (available):** Hazard Recall, False Alert Rate, Alert Precision, Alert F1, Operational Alert Score, plus approximate fire-location metrics (`results/yolo11n_operational_metrics.json`).
  - **YOLO11s (pending):** detection + operational metrics still to be produced once the checkpoint is trained — mark these as pending.
- Read real numbers from the `results/` JSON files; only YOLO11s values are placeholders.
- Keep detection metrics and operational metrics visually separated — they are complementary, not interchangeable.
- Align metric definitions with `PROJECT_CONTEXT.md` §8 and §12.
- Design metric display for easy update when YOLO11s numbers arrive.

**Must not do:**
- Display fake model performance numbers as real results.
- Mark already-available sklearn / YOLO11n metrics as "N/A — awaits M3".
- Add metrics not defined in `PROJECT_CONTEXT.md` §8 / §12.
- Compare sklearn macro F1 directly against YOLO mAP as if they were the same task.

**Files to inspect:** `PROJECT_CONTEXT.md` §8 and §12, `src/model.py`, `src/evaluation.py`, `results/baseline_*.json`, `results/yolo11n_operational_metrics.json`.

**Skills it depends on:** Streamlit layout, metric card design.

**Example prompt:**
> "Build the KPI display for the Operations & Learning Dashboard. Show the available sklearn, YOLO11n detection, and YOLO11n operational metrics from the results/ JSON files as real values, keeping detection and operational metrics in separate groups. Mark only the YOLO11s row as pending. Use a two-column layout."

**Definition of Done:** Available metrics show real values from `results/`, only YOLO11s is marked pending, detection and operational metrics are clearly separated, and the layout renders cleanly on `streamlit run app.py`.

---

### B6. UX Flow Agent

**Purpose:** Review and improve the user flow and information hierarchy across the full dashboard.

**When to use:** Before starting a new tab, after completing a tab draft, or when the dashboard feels cluttered or confusing.

**Inputs:** Current `app.py` and tab files, `PROJECT_CONTEXT.md` §6B (Operations & Learning Dashboard capabilities).

**Outputs:** UX flow diagram or written flow description, list of specific layout changes, revised tab order or navigation suggestions.

**Responsibilities:**
- Map the current user journey through the dashboard.
- Identify information hierarchy problems (what should be prominent vs. secondary).
- Recommend tab order, sidebar organization, and content grouping.
- Ensure each tab has a clear purpose and does not overlap with others.
- Align flow with the M2 dashboard story: Problem → Literature → Market → EDA + KPI.

**Must not do:**
- Redesign the product structure or add product features.
- Recommend complex frontend frameworks — keep it Streamlit.

**Files to inspect:** `app.py`, current tab structure, `PROJECT_CONTEXT.md` §6B.

**Skills it depends on:** Streamlit layout patterns, UX principles for data dashboards.

**Example prompt:**
> "Review the current tab structure in app.py. Map the user flow from tab 1 to the last tab. Identify where the story breaks down or where information hierarchy is unclear. Suggest a revised tab order and the top 3 layout improvements for M2 dashboard story: Problem → Literature → Market → EDA + KPI."

**Definition of Done:** A clear written flow map, a prioritized list of layout changes, and at least one concrete Streamlit code suggestion.

---

### B7. UI Design System Agent

**Purpose:** Define and enforce a consistent visual identity for PyroFinder's Streamlit dashboard.

**When to use:** When setting up the visual theme, defining color palette, typography, or when inconsistency is noticed across tabs.

**Inputs:** Current `app.py`, existing tab styles, `PROJECT_CONTEXT.md` (product identity).

**Outputs:** A defined color palette, spacing rules, chart style guidelines, and reusable Streamlit UI patterns as Python snippets.

**Responsibilities:**
- Define a forest / wildfire / tactical / rugged-tech visual identity.
- Recommend a practical Streamlit color and font approach (CSS overrides, metric cards, chart themes).
- Produce reusable UI helper snippets that can live in `src/ui.py` or inline.
- Keep styling practical and not over-engineered for an MVP.
- Define and enforce Plotly chart theme via `apply_chart_theme()` in `src/ui.py`. All new charts must call `apply_chart_theme()` before `st.plotly_chart()`.
- `src/ui.py` must be importable without loading any ML or geo libraries.

**Must not do:**
- Recommend replacing Streamlit with a React or Vue frontend.
- Add design complexity that slows down development.

**Files to inspect:** `app.py`, any existing CSS or theme config.

**Skills it depends on:** Streamlit theming, Plotly color schemes, CSS-in-Streamlit patterns.

**Example prompt:**
> "Define a PyroFinder visual identity for the Streamlit dashboard. Suggest a color palette (dark background, fire/smoke accent colors), a consistent chart style for Plotly charts, and reusable metric card patterns. Output copy-paste-ready Python snippets."

**Definition of Done:** A documented palette and style guide (even a short one), at least 2 reusable UI snippets, and the result looks noticeably more consistent than before.

---

### B8. Streamlit Layout Agent

**Purpose:** Implement specific Streamlit layout improvements — column structure, sidebar, expanders, metric cards, chart placement.

**When to use:** When a specific tab needs layout work, when columns or spacing is broken, or when implementing a UX recommendation.

**Inputs:** Current tab file or `app.py` section, UX Flow Agent recommendations, UI Design System Agent guidelines.

**Outputs:** Updated Streamlit code for the relevant tab or section.

**Responsibilities:**
- Implement concrete layout changes using Streamlit columns, expanders, containers, and tabs.
- Ensure charts and text are readable at classroom projector resolution.
- Keep sidebar filters logically organized.
- Reduce visual clutter in each tab.
- Use `src/ui.py` for all color values — no raw hex strings in `app.py`.
- Call `apply_chart_theme(fig)` before every `st.plotly_chart()` call.
- For `px.bar()` charts, also set `bargap=0.25` and `bargroupgap=0.1`.

**Must not do:**
- Change data logic, EDA functions, or model code while implementing layout changes.
- Introduce JavaScript or non-Streamlit components without explicit instruction.

**Files to inspect:** Target tab file, `app.py`.

**Skills it depends on:** Streamlit layout API.

**Example prompt:**
> "Refactor the EDA tab layout in app.py. Move the sidebar filters to a logical order (split, category, then image flags). Replace the full-width charts with a two-column layout where appropriate. Add an `st.info()` insight below each chart. Do not change any data or EDA logic."

**Definition of Done:** The revised layout renders without errors, passes a visual check at 1280×800 resolution, and all existing data still displays correctly.

---

### C9. YOLO Inference Agent

**Purpose:** Build and validate the YOLO11s inference pipeline for uploaded images and videos.

**When to use:** At M3 start, when implementing the inference tab, or when testing the detection path.

**Inputs:** `src/detection.py`, `src/viz.py`, `PROJECT_CONTEXT.md` §8 and §9, `CLAUDE.md` (model scope).

**Outputs:** Updated or new `src/detection.py`, inference tab code in `app.py`, bounding box overlay using Pillow/OpenCV.

**Responsibilities:**
- Load YOLO11s from `yolo11s.pt` only when called — not at import.
- Accept uploaded image or video as input.
- Run inference and return `DetectionResult` objects.
- Draw bounding boxes with class labels and confidence scores.
- Keep classes strictly `fire` and `smoke`.
- Handle the case where no model is available gracefully.

**Must not do:**
- Train the model inside the dashboard.
- Add classes other than `fire` and `smoke`.
- Use YOLO11n as the primary model.
- Load model weights at import time.

**Files to inspect:** `src/detection.py`, `src/viz.py`, `CLAUDE.md`.

**Skills it depends on:** Ultralytics YOLO11, OpenCV, Pillow, Streamlit file uploader.

**Example prompt:**
> "Build the inference function in src/detection.py. Accept a PIL image or numpy array. Load yolo11s.pt only when called. Return a list of DetectionResult objects. Handle missing model weights gracefully with a clear warning. Keep classes strictly to fire and smoke."

**Definition of Done:** Function is importable, returns correct types, draws bounding boxes on a test image, and fails gracefully when weights are missing.

---

### C10. Baseline & Evaluation Agent

**Purpose:** Maintain the sklearn baselines, the YOLO11n object-detection and operational evaluations, and the model comparison flow. The sklearn baselines, YOLO11n training/detection metrics, the cost-sensitive operational metric framework, and the YOLO11n operational evaluation are all **already complete** — the current job is analysis and the eventual YOLO11s comparison, not re-running finished work.

**When to use:** At M3, when analyzing existing results, extending the comparison display, or evaluating YOLO11s once its checkpoint exists.

**Inputs:** `src/model.py`, `src/evaluation.py`, `PROJECT_CONTEXT.md` §8 and §12, the existing sklearn result JSONs, `results/baseline_yolo11n.json`, `results/yolo11n_operational_metrics.json`, `results/yolo11n_test_predictions.csv`, YOLO inference pipeline from C9.

**Outputs:** Result analysis, updated comparison display in the Baseline tab, and (later) YOLO11s result JSONs produced the same way as YOLO11n. Full experiment metadata logged for each run.

**Responsibilities:**
- Read the existing sklearn JSON result files (`results/baseline_dummy_classifier.json`, `results/baseline_logistic_regression.json`, `results/baseline_random_forest.json`).
- Read `results/baseline_yolo11n.json` for YOLO11n object-detection metrics.
- Read `results/yolo11n_operational_metrics.json` for YOLO11n operational alert + approximate fire-location metrics.
- Use `results/yolo11n_test_predictions.csv` for per-image failure analysis (false negatives, false positives, hazard subtypes, location errors).
- Preserve the `FN=10`, `FP=1` operational cost policy unless explicitly changed.
- Distinguish **image-level sklearn classifiers** from **YOLO11n object detection** — different tasks, different granularities.
- Do **not** compare sklearn macro F1 directly against YOLO mAP as though they were the same task.
- Use **Hazard Recall** as the primary operational decision metric.
- Use **False Alert Rate** as the main secondary operational metric.
- Use **Operational Alert Score** as the weighted ranking summary.
- Treat location metrics as **approximate image-space metrics only** — never precise geolocation.
- When YOLO11s is ready, evaluate it with the same detection and operational scripts and declare it primary only if it improves mAP@0.5 and recall (and does not regress operational metrics) at acceptable inference speed.
- Log experiment results with full metadata (model, dataset, split, hyperparameters, metrics, timestamp).

**Must not do:**
- Present YOLO11n as an equal primary model.
- Display fake performance numbers.
- Run training inside the dashboard.
- Change the feature extraction pipeline when comparing classifiers (break comparability).
- Change the `FN=10`/`FP=1` cost policy without explicit instruction.

**Files to inspect:** `src/model.py`, `src/evaluation.py`, `scripts/evaluate_yolo_alert_metrics.py`, the `results/baseline_*.json` files, `results/yolo11n_operational_metrics.json`, `results/yolo11n_test_predictions.csv`, `PROJECT_CONTEXT.md` §8 and §12.

**Skills it depends on:** scikit-learn, pandas, YOLO inference pipeline (C9), Plotly comparison charts.

**M3 results already saved:**
- Sklearn (2026-06-05): DummyClassifier (Acc 0.47, F1 macro 0.21), Logistic Regression (~0.61 / ~0.62), Random Forest (~0.86 / ~0.85); full D-Fire train 17,221 / test 4,306.
- YOLO11n detection (2026-06-09, Kaggle): mAP@0.5 0.7470, mAP@0.5:0.95 0.4249, P 0.7397, R 0.6825, F1 0.7099.
- YOLO11n operational (2026-06-10, Kaggle, evaluation only): Hazard Recall 0.9331, False Alert Rate 0.0209, Operational Alert Score 0.9368; approximate fire-location coverage 0.9148, grid hit rate 0.9559.

**Example prompt:**
> "Using results/yolo11n_test_predictions.csv, break down YOLO11n false negatives and false positives by hazard subtype (fire-only, smoke-only, fire+smoke) and summarize the approximate fire-location error distribution. Read the metrics from the results/ JSON files — do not re-run the model. Keep detection and operational metrics separate."

**Definition of Done:** The analysis reads from committed `results/` files (no re-training), distinguishes detection from operational metrics, uses Hazard Recall / False Alert Rate / Operational Alert Score correctly, and the comparison renders in the Baseline tab.

---

### C11. Alert Logic Agent

**Purpose:** Build and validate the multi-frame confirmation and alert generation logic.

**When to use:** At M3, when implementing the alert tab, confirmation logic, or alert log display.

**Inputs:** `src/alerts.py`, `src/tracking.py`, `PROJECT_CONTEXT.md` §9 (Detection, Tracking, Confirmation).

**Outputs:** Updated `src/alerts.py` and `src/tracking.py`, alert log display in dashboard.

**Responsibilities:**
- Implement N-frame confirmation: fire or smoke must appear above threshold in N consecutive frames.
- N must be configurable.
- Create alert records matching the schema in `PROJECT_CONTEXT.md` §13.
- Display alert log in the dashboard with timestamp, class, confidence, camera, and approximate location.

**Must not do:**
- Trigger alerts on single-frame detections.
- Add emergency dispatch integration.
- Claim precise geolocation in alert records.

**Files to inspect:** `src/alerts.py`, `src/tracking.py`, `PROJECT_CONTEXT.md` §9 and §13.

**Skills it depends on:** YOLO inference pipeline (C9), pandas.

**Example prompt:**
> "Implement the multi-frame confirmation function in src/tracking.py. Accept a list of DetectionResult objects from consecutive frames. Return True when the same class appears above threshold in N consecutive frames. N is a parameter. Keep it testable and importable without loading YOLO."

**Definition of Done:** Function is importable, passes a unit test with mock DetectionResult objects, N is configurable, and the test covers both the confirmed and not-yet-confirmed cases.

---

### C12. Mapping & Approximate Geolocation Agent

**Purpose:** Build the mapping and approximate geolocation prototype for the dashboard.

**When to use:** At M3, when implementing image polygons, map display, or approximate location output.

**Inputs:** `src/mapping.py`, `PROJECT_CONTEXT.md` §10 and §11.

**Outputs:** Updated `src/mapping.py`, basic Folium or pydeck map in the Central Control Dashboard tab, image polygon display prototype.

**Responsibilities:**
- Implement named image polygon lookup: given a detection centroid, return the matching polygon name.
- Display camera locations on a basic map.
- Format approximate location output using the allowed wording from `PROJECT_CONTEXT.md` §9.
- Keep automatic image-to-map registration out of scope for MVP.

**Must not do:**
- Claim precise geolocation.
- Implement automatic image-to-map registration.
- Use a paid map provider.

**Files to inspect:** `src/mapping.py`, `PROJECT_CONTEXT.md` §10, §11.

**Skills it depends on:** Folium, Shapely, Streamlit map components.

**Example prompt:**
> "Implement the polygon lookup function in src/mapping.py. Accept a normalized (x, y) centroid and a list of named polygon dicts with vertices in normalized coordinates. Return the matching polygon name or None. Keep it importable without loading any geo libraries at module level."

**Definition of Done:** Function is importable, passes a unit test with simple polygon data, and returns the correct polygon name or None.

---

### C13. Testing & QA Agent

**Purpose:** Maintain and extend the test suite. Ensure imports, helpers, and core logic are covered.

**When to use:** After any code change, before any commit, or when writing a new module.

**Inputs:** `tests/test_smoke.py`, all `src/` modules, `CLAUDE.md` (conventions).

**Outputs:** Updated `tests/test_smoke.py` or new test files, `pytest` run output.

**Responsibilities:**
- Ensure all `src/` modules are importable without errors.
- Write unit tests for core helper functions: polygon lookup, N-frame confirmation, DetectionResult creation, alert record creation, class validation.
- Keep all tests runnable with `python -m pytest` from the repo root.
- Ensure no test loads YOLO weights or raw image files from outside Git.

**Must not do:**
- Write tests that require local raw dataset paths.
- Write tests that load YOLO model weights.
- Break existing passing tests.

**Files to inspect:** `tests/test_smoke.py`, all `src/` modules.

**Skills it depends on:** pytest, Python unittest patterns.

**Example prompt:**
> "Review tests/test_smoke.py. Add unit tests for the polygon lookup function in src/mapping.py and the N-frame confirmation function in src/tracking.py. Use mock data only. Ensure all tests pass with python -m pytest."

**Definition of Done:** `python -m pytest` exits with 0 failures, new tests cover at least the happy path and one edge case for each new function.

---

### C14. Deployment Readiness Agent

**Purpose:** Prepare the repo and app for public Streamlit deployment before the M3 deadline.

**When to use:** At M3, 1–2 weeks before the M3 deadline, or when the app is ready for a public URL.

**Inputs:** `requirements.txt`, `app.py`, `.gitignore`, `CLAUDE.md`, `PROJECT_CONTEXT.md` §17 (M3 exit criteria).

**Outputs:** Verified `requirements.txt`, `.gitignore` check, deployment checklist, recommended Streamlit Cloud settings.

**Responsibilities:**
- Verify `requirements.txt` is complete and pinned where needed.
- Check `.gitignore` excludes raw data, model weights, `.env`, and `__pycache__`.
- Ensure app loads without raw dataset access (use graceful fallbacks or placeholder data paths).
- Confirm the public Streamlit URL requirement from M3 exit criteria.
- Produce a deployment checklist.

**Must not do:**
- Deploy without explicit user approval.
- Remove graceful error handling.

**Files to inspect:** `requirements.txt`, `.gitignore`, `app.py`, `CLAUDE.md`.

**Skills it depends on:** Streamlit Cloud deployment, Python packaging.

**Example prompt:**
> "Review requirements.txt and app.py for deployment readiness. List any missing dependencies, any import that will fail without local raw data, and any .gitignore gaps. Produce a deployment checklist for Streamlit Cloud."

**Definition of Done:** Checklist is complete, `requirements.txt` is verified, app loads on Streamlit Cloud without errors on a machine that has no local raw data.

---

### C15. Scope Guard Agent

**Purpose:** Warn when a request would expand scope beyond the current MVP or semester plan.

**When to use:** Whenever a new feature, integration, or capability is proposed.

**Inputs:** `PROJECT_CONTEXT.md` §4 (What PyroFinder Is Not), §19 (Out of Scope), §17 (Sprint plan), the proposed feature description.

**Outputs:** A clear scope verdict: In Scope, M3 Candidate, Future/Post-MVP, or Out of Scope. Brief rationale and recommended alternative if out of scope.

**Responsibilities:**
- Check the proposal against `PROJECT_CONTEXT.md` §4 and §19.
- Warn about forbidden areas: early warning wording, live RTSP, emergency dispatch, mobile app, precise geolocation, YOLOv12, non-fire/smoke classes.
- Recommend the simplest in-scope alternative when a proposal is out of scope.

**Must not do:**
- Approve out-of-scope work.
- Rewrite the product definition.

**Files to inspect:** `PROJECT_CONTEXT.md` §4, §19, §17.

**Skills it depends on:** None (reasoning-only agent).

**Example prompt:**
> "Is it in scope to add live RTSP camera streaming to the M2 dashboard? Check against PROJECT_CONTEXT.md §19 and give a scope verdict."

**Definition of Done:** A clear verdict (In Scope / M3 Candidate / Future / Out of Scope) with rationale in under 100 words.

---

### D16. Project Shadow Mentor

**Purpose:** Act as the always-on project co-pilot. Keep the project aligned with M2/M3 goals, recommend the next best task, warn about risks, and produce handoff summaries.

**When to use:** At the start of every session, after completing a task, when feeling stuck, or when unsure what to work on next.

**Inputs:** `PROJECT_CONTEXT.md`, `CLAUDE.md`, `README.md`, `AI_AGENT_SYSTEM.md`, current progress description from the user.

**Outputs:** Next task recommendation, scope warning (if needed), command reminders, handoff summary, recommendation to update source-of-truth files.

**Responsibilities:**
- Recommend the next best task aligned with M2 or M3 goals.
- Keep the user aware of M2 exit criteria and M3 Definition of Ready (see §13 and §14).
- Remind about useful commands (`streamlit run app.py`, `python -m pytest`, metadata script).
- Warn about scope creep before it happens.
- Recommend when to update `README.md`, `CLAUDE.md`, or `PROJECT_CONTEXT.md`.
- Suggest when a task is too large and should be split.
- Warn when the Claude context window is getting large and a fresh conversation is advisable.
- Produce concise handoff summaries using the template in §17.
- Help classify any proposed task as M2, M3, Final, or Out of Scope.

**Must not do:**
- Make product, scope, or architecture decisions without explicit user instruction.
- Invent task deadlines not stated in `PROJECT_CONTEXT.md`.
- Approve out-of-scope work.

**Files to inspect:** All four source-of-truth files, current user status description.

**Skills it depends on:** None (reasoning and planning agent).

**Example prompt:**
> "I just finished the EDA tab. What should I work on next to be ready for M2 submission? What is still missing from the M2 Definition of Done?"

**Definition of Done:** A prioritized next-step recommendation with rationale, a scope check, and a command reminder — all in under 200 words.

---

### D17. Workspace & Run Manager

**Purpose:** Manage the development environment, run scripts safely, diagnose problems, and maintain Git hygiene.

**When to use:** When environment setup fails, when running scripts for the first time, when managing `.venv`, or before any expensive run.

**Inputs:** `requirements.txt`, `CLAUDE.md`, script commands, error output from terminal.

**Outputs:** Exact commands for Windows/Git Bash/PowerShell, environment fix steps, lightweight smoke check commands, Git hygiene recommendations.

**Responsibilities:**
- Provide exact commands for activating `.venv`, installing requirements, running Streamlit, and running pytest.
- Diagnose environment problems from error output.
- Recommend lightweight smoke checks before expensive operations (e.g., `python -c "import src.data"` before running full metadata build).
- Recommend overnight or batch execution for expensive jobs.
- Recommend a Git commit before any risky operation.
- Help rerun scripts safely after data or code changes.
- Maintain `.gitignore` correctness.

**Must not do:**
- Delete files without explicit approval.
- Run expensive jobs (metadata build, training, full dataset scan) without user confirmation.
- Push to Git without explicit instruction.

**Files to inspect:** `requirements.txt`, `CLAUDE.md`, `scripts/build_dfire_metadata.py`, `.gitignore`.

**Skills it depends on:** Python environment management, Git, Windows/PowerShell/Git Bash commands.

**Example prompt:**
> "I'm getting a ModuleNotFoundError for ultralytics when I run streamlit run app.py. I'm on Windows with Git Bash. Give me the exact commands to check my venv, verify the install, and fix it."

**Definition of Done:** The user can run the relevant command successfully, with no environment errors.

---

## 6. Claude Skills List

Skills are reusable prompt fragments or helper patterns. These are the minimal skills to create first (see §18 for priority):

| # | Skill | Purpose | M2 or M3 |
|---|---|---|---|
| 1 | `eda-insight-writer` | Generate a 2-sentence data-driven insight for any EDA chart | M2 |
| 2 | `streamlit-tab-template` | Boilerplate for a clean, consistent Streamlit tab | M2 |
| 3 | `scope-check` | Quick scope verdict for any proposed feature | Both |
| 4 | `handoff-summary` | Produce a concise handoff summary for context switching | Both |
| 5 | `detection-result-factory` | Create mock DetectionResult objects for testing | M3 |
| 6 | `metric-card-layout` | Reusable metric card pattern for the KPI section | M3 |
| 7 | `approximate-location-formatter` | Format location output using allowed wording | M3 |
| 8 | `experiment-record-builder` | Build a complete experiment record dict | M3 |
| 9 | `plotly-chart-theme` | Apply PyroFinder visual theme to any Plotly figure via `apply_chart_theme()` in `src/ui.py`. Call before every `st.plotly_chart()`. | M2+ |

---

## 7. Agent Collaboration Workflow

Standard task flow for any dashboard feature:

```
Shadow Mentor → recommend next task
↓
Scope Guard → confirm it's in scope
↓
Relevant domain agent (EDA / UX / Inference / etc.) → produce the work
↓
Testing & QA Agent → verify it passes tests
↓
Streamlit Layout Agent → polish the layout if needed
↓
Shadow Mentor → confirm M2/M3 alignment and recommend next step
```

For any code change, always:
1. Run `python -m pytest` before committing.
2. Run `streamlit run app.py` and do a visual check.
3. Commit with a clear message before starting the next task.

---

## 8. Workflow From Today Until M2 *(HISTORICAL — M2 submitted 2026-06-02)*

M2 submission deadline: **02/06/2026** — completed.

| Step | Task | Agent | Output |
|---|---|---|---|
| 1 | Audit current dashboard against M2 exit criteria | Shadow Mentor | Gap list |
| 2 | Refine Problem Understanding tab | M2 Product & Problem Agent | Revised tab |
| 3 | Build or refine Literature Review tab | Literature Review Agent | Revised tab |
| 4 | Build or refine Market Review tab | Market Review Agent | Revised tab |
| 5 | Add missing EDA charts and insights | Data & EDA Agent | Updated src/eda.py |
| 6 | Build KPI/metrics placeholder section | KPI & Metrics Agent | KPI display |
| 7 | UX flow review | UX Flow Agent | Flow map + change list |
| 8 | Apply layout improvements | Streamlit Layout Agent | Revised app.py tabs |
| 9 | Apply visual identity | UI Design System Agent | Consistent styling |
| 10 | Run full test suite | Testing & QA Agent | Green pytest |
| 11 | Final M2 review against exit criteria | Shadow Mentor | Go/no-go |

---

## 9. Workflow From M2 To M3

M3 submission deadline: **23/06/2026**

| Phase | Task | Agent | Output | Status |
|---|---|---|---|---|
| 1 | Lock M2 dashboard with a Git tag | Workspace & Run Manager | `git tag m2-final` | — |
| 2 | DummyClassifier baseline pipeline (60-dim features, full D-Fire) | Baseline & Evaluation Agent | `scripts/dummy_try.py`, `results/baseline_dummy_classifier.json` | ✅ Done 2026-06-05 |
| 3 | Add Baseline tab to Operations & Learning Dashboard | Baseline & Evaluation Agent | Baseline tab in `app.py` | ✅ Done 2026-06-06 |
| 4 | Add Logistic Regression and Random Forest classifiers | Baseline & Evaluation Agent | `scripts/simple_baselines.py`, new results JSON | ✅ Done 2026-06-05 |
| 5 | Display sklearn model comparison in dashboard | Baseline & Evaluation Agent | Comparison table in Baseline tab | ✅ Done |
| 6 | Build YOLO11n baseline benchmark | Baseline & Evaluation Agent | `results/baseline_yolo11n.json`, `scripts/YOLO11n_baseline.py` | ✅ Done 2026-06-09 (Kaggle) |
| 6b | Build cost-sensitive operational metric framework | Baseline & Evaluation Agent | `src/evaluation.py`, `tests/test_evaluation.py` | ✅ Done 2026-06-10 |
| 6c | YOLO11n operational alert evaluation (evaluation only) | Baseline & Evaluation Agent | `results/yolo11n_operational_metrics.json`, `results/yolo11n_test_predictions.csv` | ✅ Done 2026-06-10 (Kaggle) |
| 7 | Build YOLO11s inference + fine-tuning | YOLO Inference Agent | Updated `src/detection.py` | Next |
| 8 | Build N-frame confirmation logic | Alert Logic Agent | Updated `src/tracking.py` | — |
| 9 | Build alert record creation | Alert Logic Agent | Updated `src/alerts.py` | — |
| 10 | Add inference tab to dashboard | YOLO Inference Agent | New inference tab | — |
| 11 | Add alert log tab | Alert Logic Agent | Alert log display | — |
| 12 | Add model comparison display (YOLO11s vs YOLO11n) | Baseline & Evaluation Agent | Comparison table | — |
| 13 | Add basic map + camera metadata | Mapping & Geolocation Agent | Basic map tab | — |
| 14 | Add image polygon prototype | Mapping & Geolocation Agent | Polygon display | — |
| 15 | Extend test suite | Testing & QA Agent | Green pytest | — |
| 16 | Deployment readiness check | Deployment Readiness Agent | Deployment checklist | — |
| 17 | Deploy to Streamlit Cloud | User + Workspace Manager | Public URL | — |
| 18 | Final M3 review | Shadow Mentor | Go/no-go | — |

---

## 10. Token / Context Management Strategy

**Problem:** Long conversations accumulate context that makes Claude less accurate and more likely to contradict earlier decisions.

**Rules:**
- Start a fresh Claude conversation for each distinct agent task.
- Never carry a conversation beyond ~15,000 tokens of productive work without a handoff summary.
- Always start a session by providing the relevant source-of-truth files (`PROJECT_CONTEXT.md`, `CLAUDE.md`, `AI_AGENT_SYSTEM.md`) and a brief current-state description.
- When switching agents, produce a handoff summary first (see §17).
- If a task requires more than one agent, split it across conversations.
- When Claude starts repeating itself, hedging more than usual, or producing inconsistent output, assume context overload and start a fresh session.

**Session start template:**
```
Use PROJECT_CONTEXT.md as the source of truth. Use CLAUDE.md as the coding-agent context.
Current state: [brief status — what is done, what is broken, what I need next]
Task: [specific task for this session]
```

---

## 11. UX/UI Strategy

**Visual identity:** Forest / wildfire / tactical / rugged-tech. Dark background. Fire and smoke accent colors (amber, orange, gray-smoke). Clean sans-serif typography. High contrast for classroom projection.

**Information hierarchy principles:**
- Lead with the problem and the user, not the technology.
- Each tab has a single clear purpose — no mixed concerns.
- Every chart has a written insight immediately below it.
- Metrics are never just numbers — always include a brief interpretation.
- Use `st.info()`, `st.metric()`, and `st.expander()` to separate primary content from supporting detail.

**Streamlit layout rules:**
- Use `st.columns()` for side-by-side charts — avoid full-width stacking for everything.
- Sidebar filters are grouped by topic: first data scope (split, category), then image flags.
- Avoid more than 3 sidebar sections.
- Use `st.expander()` for raw data tables and sample images — keep them collapsed by default.
- Minimum font size for charts: 12pt. Use `plotly` chart `layout.font.size` to enforce this.
- Color palette for Plotly charts: use a consistent theme (e.g., `plotly_dark` or a custom palette) across all charts.

**Dashboard story (M2):**
```
Tab 1: Problem Understanding  →  Why does this problem exist and who has it?
Tab 2: Literature Review      →  What has been done and what is the gap?
Tab 3: Market Review          →  Who else is solving this and why are we different?
Tab 4: Dataset & EDA          →  What does the data tell us?
Tab 5: KPI & Metrics Plan     →  How will we measure success?
```

---

## 12. Recommended VS Code / Claude Code Workflow

**Directory:** Work from the repo root at all times.

**Virtual environment (Windows/Git Bash):**
```bash
python -m venv .venv
source .venv/Scripts/activate      # Git Bash
# or
.venv\Scripts\activate             # PowerShell
pip install -r requirements.txt
```

**Run dashboard:**
```bash
streamlit run app.py
```

**Run tests:**
```bash
python -m pytest
```

**Generate metadata CSV:**
```bash
python scripts/build_dfire_metadata.py \
  --raw-root "C:\Users\boris.azarov\OneDrive - Technion\Desktop\PyroFinder\RAW_DATA\D-Fire" \
  --output data/dfire_metadata.csv
```

**Smoke check before heavy runs:**
```bash
python -c "import src.data; import src.eda; import src.viz; print('imports OK')"
```

**Git hygiene:**
```bash
git status                          # check before any commit
git add -p                          # stage selectively
git commit -m "feat: clear message"
git tag m2-final                    # tag at milestone submission
```

**Claude Code tips:**
- Open one agent task per Claude Code session.
- Provide the source-of-truth files at the start of each session.
- Never ask Claude Code to modify `PROJECT_CONTEXT.md`, `CLAUDE.md`, or `AI_AGENT_SYSTEM.md` unless that is the explicit task.
- Use Claude Code for code edits; use the Claude chat interface for planning, review, and handoffs.

---

## 13. Definition of Done for M2

M2 is done when all of the following are true:

- [ ] Streamlit dashboard loads without errors on `streamlit run app.py`.
- [ ] Tab 1 (Problem Understanding) presents the problem, the Dani persona, and uses no forbidden wording.
- [ ] Tab 2 (Literature Review) includes at least 3 paper summaries and a differentiation table.
- [ ] Tab 3 (Market Review) includes a market gap description and a competitor comparison.
- [ ] Tab 4 (Dataset & EDA) includes at least 3 visualizations, each with a written insight, using only committed CSV data.
- [ ] Tab 5 (KPI & Metrics) displays all 7 planned metrics as placeholders, clearly marked as "awaits M3."
- [ ] D-Fire metadata counts match `CLAUDE.md` M2 status: 21,527 images, class 0 = smoke, class 1 = fire.
- [ ] `python -m pytest` passes with 0 failures.
- [ ] No raw image paths are hardcoded in committed code.
- [ ] No model weights are loaded at import time.
- [ ] `.gitignore` excludes raw data, model weights, `.env`, `__pycache__`, and `.venv`.
- [ ] `README.md` M2 section is updated to reflect current dashboard state.
- [ ] A Git tag `m2-final` is created at submission.

---

## 14. Definition of Ready for M3

M3 has started as of 2026-06-05. Prerequisites:

- [x] M2 Definition of Done fully checked off (submitted 2026-06-02).
- [ ] `git tag m2-final` exists.
- [x] `requirements.txt` is verified and complete (in use).
- [x] All M2 tests pass.
- [x] DummyClassifier sklearn baseline complete (`results/baseline_dummy_classifier.json`).
- [ ] The YOLO11s inference path is planned in `src/detection.py` (stub or full).
- [ ] YOLO11n baseline scope is confirmed: same data, same split, same metrics.
- [ ] Alert schema in `src/alerts.py` matches `PROJECT_CONTEXT.md` §13.
- [ ] Deployment target (Streamlit Cloud) is confirmed and accessible.

---

## 15. First 10 Prompts To Run

Run these in sequence to establish the agent system and assess current state.

1. **Shadow Mentor — current state assessment**
   > "Use PROJECT_CONTEXT.md, CLAUDE.md, and AI_AGENT_SYSTEM.md as context. M2 is submitted and M3 is active (due 2026-06-23). Sklearn baselines, the YOLO11n object-detection baseline, the cost-sensitive operational metric framework, and the YOLO11n operational alert evaluation are all complete. What is the gap to the M3 deliverable and what should I work on first?"

2. **Scope Guard — check a proposed feature**
   > "Is it in scope for M2 to add a satellite map with fire heatmap overlay? Check against PROJECT_CONTEXT.md §4 and §19."

3. **UX Flow Agent — audit current flow**
   > "Review the current tab structure in app.py and map the user flow. Identify the top 3 hierarchy problems. Suggest a revised tab order for the M2 story: Problem → Literature → Market → EDA → KPI."

4. **M2 Product & Problem Agent — refine problem tab**
   > "Review the Problem Understanding tab content. Verify it matches PROJECT_CONTEXT.md §2 and §5. Suggest improvements to the Dani persona and problem framing. Flag any forbidden wording."

5. **Literature Review Agent — build the tab**
   > "Build the Literature Review tab content. Summarize these papers: [paste summaries]. Add a differentiation table comparing Pano AI, FIREWAVE, and CANDO against PyroFinder."

6. **Data & EDA Agent — add a missing chart**
   > "Add a scatter plot of smoke bbox area vs. fire bbox area for fire+smoke images. Use data/dfire_metadata.csv. Write a 2-sentence insight. Follow the code style in src/eda.py."

7. **UI Design System Agent — define visual identity**
   > "Define a PyroFinder visual identity for the Streamlit dashboard. Output a color palette, a Plotly chart theme, and 2 reusable metric card snippets."

8. **KPI & Metrics Agent — show real and pending metrics**
   > "Build the KPI display for the Operations & Learning Dashboard. Show the available sklearn, YOLO11n detection, and YOLO11n operational metrics from the results/ JSON files as real values, keeping detection and operational metrics in separate groups. Mark only the YOLO11s row as pending. Use a two-column layout."

9. **Testing & QA Agent — verify test suite**
   > "Run python -m pytest and show me the output. Then review tests/test_smoke.py and suggest what unit tests are missing for the current src/ modules."

10. **Shadow Mentor — produce handoff summary**
    > "I am ending this session. Produce a handoff summary using the template in AI_AGENT_SYSTEM.md §17."

---

## 16. Recommended Daily Workflow

```
Session start
  └── Paste source-of-truth files (PROJECT_CONTEXT.md, CLAUDE.md)
  └── Run Shadow Mentor prompt: "What is the next best task?"
  └── Confirm scope with Scope Guard if uncertain

Work session
  └── One task per Claude conversation (keep context clean)
  └── Run tests after every code change: python -m pytest
  └── Run app after every layout change: streamlit run app.py
  └── Commit before switching tasks: git commit -m "feat: ..."

Session end
  └── Run Shadow Mentor: "Produce a handoff summary"
  └── Update CLAUDE.md or PROJECT_CONTEXT.md if anything changed
  └── Close the Claude conversation and start fresh next session
```

---

## 17. Handoff Summary Template

Use this template at the end of every session or before switching tasks.

```markdown
## PyroFinder Handoff Summary

**Date:** YYYY-MM-DD
**Session goal:** [What you set out to do]
**Status:** [Done / Partial / Blocked]

**What was completed:**
- [Item 1]
- [Item 2]

**What is still open:**
- [Item 1]
- [Item 2]

**Files changed:**
- [file path and brief description]

**Commands to know:**
- streamlit run app.py
- python -m pytest
- [any other relevant command]

**Next recommended task:**
[One sentence]

**Warnings or risks:**
[Any scope, context, or technical risk to flag]

**M2 checklist status:**
[ ] Problem tab done
[ ] Literature tab done
[ ] Market tab done
[ ] EDA tab done (N visualizations with insights)
[ ] KPI tab done
[ ] Tests passing
[ ] README updated
```

---

## 18. Minimal Skills To Create First

Build these skills before M3. Each is a reusable prompt fragment.

| Priority | Skill | Description |
|---|---|---|
| 1 | `eda-insight-writer` | Input: chart description and key numbers. Output: 2-sentence insight starting with a data observation and ending with an ML implication. |
| 2 | `streamlit-tab-template` | Input: tab name and content spec. Output: clean Streamlit tab boilerplate with consistent layout, sidebar handling, and `st.info()` insight block. |
| 3 | `scope-check` | Input: proposed feature description. Output: verdict (In Scope / M3 Candidate / Future / Out of Scope) with one-line rationale. |
| 4 | `handoff-summary` | Input: session status description. Output: filled handoff summary using the §17 template. |
| 5 | `detection-result-factory` | Input: class name, confidence, bbox tuple. Output: a mock `DetectionResult` object for use in unit tests. |

---

## 19. Agents That Should Stay As Prompts For Now

These agents do not need to be built as formal skills or automated flows yet. Run them as single prompts in a fresh Claude conversation.

| Agent | Reason |
|---|---|
| Literature Review Agent | Content is human-verified and low-frequency — a prompt is sufficient |
| Market Review Agent | Market data is human-provided — no automation needed yet |
| M2 Product & Problem Agent | Tab is written once — a review prompt is enough |
| Scope Guard Agent | Fast reasoning task — no automation needed |
| Shadow Mentor | Best run interactively at session start/end |
| UI Design System Agent | One-time setup task — document the output and reuse it |

Agents that will need more structure by M3:

| Agent | Why it needs more structure |
|---|---|
| YOLO Inference Agent | Code complexity, model loading, file handling |
| Baseline & Evaluation Agent | Reproducible evaluation pipeline with logging |
| Testing & QA Agent | Grows with the test suite |
| Deployment Readiness Agent | Checklist-driven with external dependencies |

---

---

## 20. Cross-Cutting Code Rules

These rules apply to every agent that writes or reviews code for PyroFinder.

1. **No raw hex strings in `app.py`.** All colors come from `src/ui.py` — use `PYRO_COLORS`, `CAT_COLORS`, `CLASS_COLORS`, or `SPLIT_COLORS`.
2. **`apply_chart_theme()` is mandatory.** Every Plotly figure must call `apply_chart_theme(fig)` immediately before `st.plotly_chart()`.
3. **Bar charts get bargap.** All `px.bar()` figures must also call `fig.update_layout(bargap=0.25, bargroupgap=0.1)`.
4. **`src/` is import-safe.** Every module under `src/` must be importable without loading ML models, datasets, or geo libraries.
5. **Two classes only.** No module may reference classes other than `fire` and `smoke`.
6. **Location is always approximate.** No code may claim precise geolocation.

---

## 21. Repository Structure

```text
project-root/
├── README.md
├── CLAUDE.md                   ← Claude Code working context
├── PROJECT_CONTEXT.md          ← product scope, ML problem, datasets
├── requirements.txt
├── .gitignore
├── app.py                      ← Streamlit entry point (multi-tab shell)
├── src/
│   ├── __init__.py
│   ├── data.py                 ← dataset loading, metadata helpers
│   ├── eda.py                  ← EDA helper functions
│   ├── viz.py                  ← on-the-fly YOLO box annotation
│   ├── ui.py                   ← PyroFinder color palette and Plotly chart theme
│   ├── model.py                ← model metadata, metrics plan
│   ├── detection.py            ← DetectionResult dataclass, class validation
│   ├── tracking.py             ← multi-frame confirmation, direction estimation
│   ├── mapping.py              ← mapping modes, polygon helpers, location formatting
│   ├── alerts.py               ← alert record creation, status validation
│   └── evaluation.py           ← cost-sensitive operational alert metrics + approximate fire-location helpers (pure stdlib)
├── scripts/
│   ├── build_dfire_metadata.py ← generates data/dfire_metadata.csv
│   ├── dummy_try.py            ← M3 sklearn baseline: DummyClassifier on full D-Fire
│   ├── simple_baselines.py     ← M3: Logistic Regression and Random Forest baselines
│   ├── YOLO11n_baseline.py     ← M3: YOLO11n object-detection baseline runner (reproducible)
│   └── evaluate_yolo_alert_metrics.py ← M3: evaluation-only operational alert + fire-location metrics (no training)
├── results/
│   ├── baseline_dummy_classifier.json    ← DummyClassifier metrics (2026-06-05)
│   ├── baseline_logistic_regression.json ← Logistic Regression metrics
│   ├── baseline_random_forest.json       ← Random Forest metrics
│   ├── baseline_yolo11n.json             ← YOLO11n detection metrics
│   ├── results_yolo11n.csv               ← YOLO11n per-epoch training curves
│   ├── yolo11n_operational_metrics.json  ← YOLO11n operational alert + fire-location metrics
│   └── yolo11n_test_predictions.csv      ← YOLO11n per-image alert outcome + fire-location error table
├── data/
│   ├── .gitkeep
│   ├── dfire_metadata.csv      ← 36-column generated CSV (committed)
│   └── samples/dfire/          ← 20 sample images + labels (committed fallback)
├── docs/
│   ├── AI_AGENT_SYSTEM.md      ← agent roles, skill catalogue (this file)
│   ├── M2_DATA_EDA.md          ← data workflow, class mapping, cleaning decisions
│   ├── M2_dashboard.md         ← dashboard design notes
│   ├── M2_GAP_LIST.md          ← known gaps and open items as of M2
│   ├── Literature_review.md    ← literature and related work
│   └── market_survey_wildfire_existing_sensors.md ← competitor/market landscape
├── SprintPlan/
│   └── SPRINT_PLAN.md
├── notebooks/
│   └── 01_eda.ipynb
└── tests/
    ├── test_smoke.py
    └── test_evaluation.py      ← unit tests for src/evaluation.py (alert confusion, cost weighting, location helpers)
```

Future additions expected for M3:
- `pages/` — Streamlit multi-page app pages
- YOLO11s detection + operational result JSONs (once the checkpoint is trained)

---

*PyroFinder · AI Agent System · Technion Course 016833 · Last updated: 2026-06-12*

---

## PyroFinder M2 UI / Visual Design System

**Style name:** Röki-inspired Nordic low-poly wildfire dashboard

**Description:**
Stylized low-poly Nordic folklore aesthetic. Matte 3D / flat-shaded feeling. Twilight forest atmosphere. Deep evergreen shadows. Misty icy-blue valley. Pink-purple Scandinavian dusk sky. Small warm ember/fire accents. Cinematic, serious, classroom-ready. Glassmorphism dashboard cards over animated background.

**Background video:** `design_images/Nordic_Forest_LowPolymp_.mp4`
If the video is missing, the theme falls back to a gradient using twilight_sky, deep_fjord, and pine_shadow. Never crashes.

**Official color palette:**
```python
PYRO_UI_COLORS = {
    "twilight_sky":  "#1E2336",
    "deep_fjord":    "#2B3248",
    "stone_surface": "#3E445E",
    "dusk_rose":     "#E07A8A",
    "nordic_lilac":  "#8F8CC7",
    "pine_shadow":   "#264036",
    "morning_mist":  "#D6D7E6",
    "frost_white":   "#F3F4F8",
    "raven_gray":    "#A5A8B8",
    "ember_glow":    "#E4573D",
    "hud_cyan":      "#8CE9FF",
}
```

**Official glass constants:**
```python
PYRO_GLASS = {
    "main_panel":         "rgba(30, 35, 54, 0.58)",
    "sidebar":            "rgba(30, 35, 54, 0.88)",
    "card":               "rgba(62, 68, 94, 0.64)",
    "card_hover":         "rgba(62, 68, 94, 0.78)",
    "soft_border":        "rgba(214, 215, 230, 0.16)",
    "strong_border":      "rgba(214, 215, 230, 0.30)",
    "background_overlay": "rgba(14, 18, 34, 0.64)",
}
```

**Chart color assignments:**
- fire: #E4573D
- smoke: #A5A8B8
- background/negative: #3E445E
- train: #8F8CC7
- val: #D6D7E6
- test: #E07A8A

**Agent rules for visual tasks:**
- All visual changes go through src/ui.py. Do not inline CSS in tab files.
- Do not change dashboard content, text, charts, or data when doing visual-only tasks.
- inject_pyrofinder_theme() must be called once, immediately after st.set_page_config().
- apply_chart_theme() must be called on every Plotly figure before st.plotly_chart().

**Reusable image-generation prompt for background frames:**
A wide cinematic 16:9 background illustration for PyroFinder, in a stylized low-poly Nordic folklore minimalist style inspired by Röki. Clean flat-shaded 3D geometric shapes, soft matte surfaces, minimal vector-like textures, muted Scandinavian twilight palette, deep evergreen forest shadows, icy blue mist, pastel pink-purple sky, and small warm ember accents.
Scene: a quiet private farm at the edge of a forested mountain valley. Rolling hills, dark pine trees, a winding dirt path, a small rustic Nordic farmhouse and barn, a simple wooden fence marking the farm border, distant snow-capped mountains, soft fog in the valley, and a carved rune stone in the foreground. Serene but slightly tense fairytale atmosphere, cinematic composition, high artistic detail, no people, no text, no logos.
For fire frames: fire starts near the farm border fence, continuous connected burning patch, no disconnected flames. Optional subtle wildfire-detection HUD corner brackets around active fire area.
For no-fire frames: no fire, no smoke, no embers, no scorch marks, no HUD brackets.

**Animation rule for Streamlit:**
Use a short MP4/WebM background video with a dark overlay. Keep it muted, looped, and fixed behind app content. Do not animate Streamlit content unless explicitly requested.

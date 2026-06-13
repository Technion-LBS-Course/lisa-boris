# M3 Submission Requirements — PyroFinder

## 1. Title and document purpose

**Document:** `docs/M3_SUBMISSION_REQUIREMENTS.md`
**Milestone:** M3 — Predictive modelling milestone for PyroFinder (Technion course 016833, Location-Based Services: Data Science).
**Status of this document:** Requirements and acceptance specification. It is not a results report.

This document translates the **official M3 course requirements** into **concrete, auditable PyroFinder requirements**. For every requirement it records the **current repository evidence**, the **incomplete or pending evidence**, and the **final acceptance criterion**. It is written to be checked against the real working tree, not against assumptions.

Throughout, four layers are kept separate and explicitly labelled:

1. **Official course requirement** — what the course demands of every M3 submission.
2. **PyroFinder-specific interpretation** — how that requirement maps onto a two-class fire/smoke detection product.
3. **Current repository evidence** — what the committed and working-tree files actually demonstrate today.
4. **Final acceptance criterion** — the bar that must be met before submission.

Source-of-truth precedence used when files disagree (highest first):

1. Official M3 requirements supplied for this milestone (course submission rules, including the deadline).
2. `PROJECT_CONTEXT.md` — canonical product and ML scope.
3. `CLAUDE.md` — current repository structure and coding-agent status.
4. Actual source code, result JSON/CSV files, checkpoints, and tests — implementation evidence.
5. `README.md` and `docs/AI_AGENT_SYSTEM.md` — supporting documentation, with stale statements flagged.

Contradictions are **not silently reconciled**; they are stated in the evidence or risk sections.

---

## 2. Official milestone summary

M3 moves the project from **Descriptive** (M2: dataset inspection and EDA dashboard) to **Predictive**.

The required transformation is from "a dashboard that *describes* data" to "an application in which a model *trained on real data* produces a *prediction* for new user input." A descriptive dashboard alone does **not** satisfy M3. A model must have learned from real data and must produce a prediction, classification, object detection, or cluster assignment for a new input.

The required application flow is:

```text
Input → Predictive model → Output
```

---

## 3. Submission deadline and format

| Item | Value |
|---|---|
| **Official submission deadline** | **Meeting 8 — Tuesday, June 16, 2026** |
| **Submission format** | Streamlit application |
| **Repository** | GitHub (this repository) |

> **Stale-documentation risk.** Some project files may still carry an older M3 date such as **June 23, 2026**. For this milestone the authoritative deadline is **Tuesday, June 16, 2026**. If any in-repository file states June 23, 2026 as the M3 submission date, that statement is **stale** and must be treated as superseded by this document. This document does not edit those files; it only flags the discrepancy (see §19).

---

## 4. Required deliverables

### 4.1 Multiple algorithms (official)

- At least **2–3 suitable candidate algorithms**.
- Candidates must be **trained on real project data** (D-Fire).
- Candidates used for a **direct ranking** must solve the **same predictive task**.

### 4.2 Comparison and model selection (official)

- A **comparison table** based on the M2 KPI.
- Clear **train/test separation**.
- Evaluation on **unseen test data**, not training performance alone.
- A **selected winner**.
- A **written, KPI-based winner justification**.

### 4.3 Streamlit submission (official)

- The selected model runs inside a **basic Streamlit application**.
- The user supplies an **example input**.
- The model returns a **real output**.
- The predictive path is **functional, not a static mock-up**.

### 4.4 Updated GitHub repository (official)

The repository must contain:

- training code;
- model-evaluation / model-comparison code;
- Streamlit application code;
- README instructions for running the application;
- README summary of measured model results;
- README statement identifying the selected winner and why it was selected.

---

## 5. What is not required until M4

The following are **explicitly out of scope for M3** and must not be treated as M3 acceptance blockers:

- a fully polished production interface;
- an AI agent inside the application;
- full cloud deployment;
- a production mobile application;
- a production emergency-services interface;
- advanced alert-management workflows;
- automatic image-to-map registration;
- precise geolocation;
- physical fire-spread prediction;
- emergency dispatch integration.

A simple but valid predictive Streamlit flow is **more important for M3 than UI polish**.

---

## 6. Official checklist

- [ ] At least 2–3 suitable algorithms trained on real D-Fire data.
- [ ] Algorithms used for a direct ranking solve the same predictive task.
- [ ] Comparison table based on the M2 KPI.
- [ ] Clear train/test separation; final metrics on unseen test data.
- [ ] A single selected winner.
- [ ] A written, KPI-based winner justification.
- [ ] Selected model runs in a basic Streamlit app.
- [ ] User supplies an input; model returns a real output.
- [ ] Predictive path is functional (not a static mock-up).
- [ ] GitHub contains training code, evaluation/comparison code, and Streamlit code.
- [ ] README: run instructions, measured-result summary, named winner + justification.

---

## 7. Proposed grading criteria

| Criterion | Weight |
|---|---:|
| Comparison of multiple algorithms and justified selection | 30% |
| Selected algorithm works correctly on real data | 25% |
| Valid Streamlit input → output submission | 20% |
| Correct KPI evaluation with train/test separation | 15% |
| Code and README quality in GitHub | 10% |
| **Total** | **100%** |

---

## 8. PyroFinder predictive-task definition

PyroFinder's repository currently contains **two distinct predictive-task families**. They are complementary; they are **not** the same task.

### 8.1 Image-level classification baselines (sklearn)

The sklearn models classify the **whole image** as exactly one of:

- `background`
- `fire`
- `smoke`

Current candidates:

- DummyClassifier (most-frequent strategy)
- Logistic Regression
- Random Forest

Properties:

- Use a 60-dimensional handcrafted color feature vector (RGB mean+std, HSV mean+std, 16-bin × 3-channel histogram; image resized to 64×64).
- Image-level label derivation: class 1 present → `fire`; class 0 only → `smoke`; empty label file → `background`.
- **Do not** produce bounding boxes.
- **Do not** localize a fire or smoke region.

These are **valid classical predictive baselines**, but they do **not** implement the complete PyroFinder object-detection requirement (no localization, no approximate location output).

### 8.2 Object-detection candidates (YOLO11)

The object detectors predict, per image/frame:

- bounding boxes;
- class labels;
- confidence scores.

The only allowed detection classes are strictly:

- `fire`
- `smoke`

Detector candidates:

- **YOLO11n** — trained lightweight **baseline and fallback**.
- **YOLO11s** — **current primary detector**, with measured detection and operational results in `results/`.

> Terminology: always write `YOLO11n` or `YOLO11s` where the version matters. Do not use a generic "YOLO" for a model-specific claim.

---

## 9. Candidate-model families and comparability rules

The following rules are mandatory and must be visible in the submission's comparison logic and README.

1. **Image-level sklearn classification and YOLO11 object detection are different predictive tasks.**
2. **sklearn Macro F1 must not be directly compared with YOLO11 mAP** as though they measure the same output.
3. sklearn **accuracy, Macro F1, and per-class recall** are valid for comparing the sklearn classifiers **with each other**.
4. Detection **mAP, Precision, Recall, and F1** are valid for comparing **YOLO11n and YOLO11s with each other**.
5. **Operational alert metrics** may be computed for both families **after reducing predictions to**:

   ```text
   hazard detected / no hazard detected
   ```

   This is a useful common operational view. It does **not** make the underlying predictive tasks identical.
6. **Approximate fire-location metrics apply only to object detectors** — sklearn classifiers produce no bounding boxes, so their location metrics are N/A.
7. The most defensible final PyroFinder detector comparison is:

   ```text
   YOLO11n versus YOLO11s
   ```

   using the **same** D-Fire train/test split, the **same** confidence-threshold policy, the **same** object-detection evaluation process, the **same** operational alert evaluation process, and otherwise **equivalent evaluation conditions**.
8. The sklearn models may remain as **supporting classical baselines**.
9. The sklearn models **must not** be placed in a misleading direct ranking against YOLO11n / YOLO11s using incompatible metrics.
10. **YOLO11s must not be declared the winner before real measured YOLO11s results exist.**
11. A **pending, synthetic, malformed, incomplete, or training-in-progress** result must **never** win model selection.

**Repository evidence (today):** `src/results_loader.py` enforces (1)–(2), (6)–(11) at the code level: `load_detection_result` / `load_operational_result` reject the wrong file kind (`STATUS_NOT_DETECTION` / `STATUS_NOT_OPERATIONAL`); `is_selectable_operational` requires `STATUS_OK`, rejects embedded statuses containing `synthetic`/`placeholder`/`pending`/`training`, and requires non-null `hazard_recall`, `false_alert_rate`, `operational_alert_score`. `app.py` renders the sklearn comparison, the object-detection comparison, and the operational comparison as **separate** tables. Unit tests in `tests/test_results_loader.py` lock these behaviors.

---

## 10. KPI and winner-selection policy

### 10.1 Alert-level hazard mapping

```text
fire        → hazard
smoke       → hazard
background  → no hazard
```

### 10.2 Metric priorities (decision order)

1. **Hazard Recall** — primary operational decision metric.
2. **False Alert Rate** — main secondary operational metric.
3. **Operational Alert Score** — cost-sensitive ranking summary.
4. **Object-detection Recall and mAP@0.5** — supporting detection-quality metrics.
5. **Inference speed** — supporting metric only when measured under documented and comparable conditions.

### 10.3 Cost weights and score formula (source of truth: `src/evaluation.py`)

```text
Missed hazard / false negative weight = 10
False alert  / false positive weight  = 1
```

```text
weighted_error_cost = 10 × FN + 1 × FP

max_possible_cost   = 10 × total_hazard_cases + 1 × total_background_cases

operational_alert_score = 1 - weighted_error_cost / max_possible_cost
```

Higher Operational Alert Score is better. (Verified against the YOLO11n operational file: `10×154 + 42 = 1582`; `10×2301 + 2005 = 25015`; `1 − 1582/25015 = 0.9368`.)

### 10.4 Final detector-selection hierarchy

1. Require **complete, measured, non-synthetic** operational result files.
2. Prefer **higher Hazard Recall**.
3. When Hazard Recall is effectively tied, prefer **lower False Alert Rate**.
4. Then prefer **higher Operational Alert Score**.
5. Use **detection Recall and mAP@0.5** as supporting evidence.
6. Use **measured inference speed** only as an additional practical consideration.
7. **Do not choose YOLO11s merely because it is the designated primary detector** — selection must follow the measured KPI hierarchy.
8. Keep **YOLO11n as the selected fallback** when YOLO11s has not been measured or does not show a justified improvement.

### 10.5 Interim detector vs final winner

- **Eligible measured detectors:** both **YOLO11n** and **YOLO11s** now have complete measured detection and operational results.
- **Selected detector:** the detector chosen by the KPI hierarchy from those measured results. Today this is **YOLO11s** (it wins on Hazard Recall, False Alert Rate, and Operational Alert Score; see §15).

> **Important:** YOLO11s is selected **only because** its measured detection and operational result files exist and it wins the operational selection rule over YOLO11n. If those measured files were absent, YOLO11s would not be selectable and YOLO11n would remain the interim selected detector.

**Repository evidence (today):** `src/results_loader.select_operational_winner` implements this hierarchy via `_rank_candidates` (sort by `-hazard_recall`, then `false_alert_rate`, then `-operational_alert_score`, then `-detection_recall`, `-map50`, then `inference_ms`). With both detectors measured, the winner resolves to `YOLO11s`. The unit test `test_pending_yolo11s_does_not_beat_measured_yolo11n` still locks the absent-file fallback (a detector with missing result files never wins).

---

## 11. Evaluation metric families

These three families are **complementary, not interchangeable**. They must remain in separate tables and must never be combined into a single ranking.

| Family | Metrics | Purpose | Applies to |
|---|---|---|---|
| **Object-detection evaluation** | mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1 | Measure bounding-box and class-detection quality | YOLO11n, YOLO11s |
| **Operational alert evaluation** | TP, FN, FP, TN, Hazard Recall, False Alert Rate, Alert Precision, Alert F1, Weighted Error Cost, Operational Alert Score | Measure whether the system raises or misses a hazard-level alert | Both families (after hazard reduction) |
| **Approximate fire-location evaluation** | location coverage count, location coverage rate, mean normalized location error, median normalized location error, 3×3 grid hit rate, bottom-center fire-box anchor, best-IoU fire-box matching | Measure approximate image-space fire location when both GT fire and predicted fire boxes exist | Object detectors only |

Location-metric constraints (must be stated wherever location metrics appear):

- Location metrics use **class-1 fire boxes only**.
- **Smoke-only images are not treated as fire epicenters.**
- Outputs are **approximate image-space estimates**.
- These are **not precise geographic coordinates**.

**Repository evidence (today):** `src/evaluation.py` implements the bottom-center anchor (`anchor_x = x_center`, `anchor_y = y_center + height/2`), `best_iou_fire_match` (single best-IoU pair per image), `fire_location_error` (returns `None` when no GT fire or no predicted fire), and the 3×3 grid hit logic. The operational comparison in `app.py` shows location columns as `N/A` for sklearn rows.

---

## 12. Train/test and data-leakage requirements

Acceptance requirements:

- Use the **real D-Fire** data.
- Preserve the documented D-Fire **train / held-out test split** (train 17,221 · test 4,306 · total 21,527).
- Training uses the **training split**.
- Final reported evaluation uses **unseen test data**.
- The test split **must not** fit sklearn feature scalers or models.
- The test split **must not** be used as additional training data for YOLO11n or YOLO11s.
- Repeated hyperparameter decisions made on the held-out test set must be **identified as a leakage risk**.
- Metrics must **identify the evaluated split**.
- **Confidence threshold and image size** must be documented for operational evaluation (current: conf 0.25, imgsz 640).
- YOLO11n and YOLO11s must be evaluated under **equivalent conditions** for a fair final comparison.
- **Training curves alone are not sufficient submission evidence.**
- Validation/test metrics must come from an **actual completed run**.

### 12.1 Accurate note on the YOLO11n split usage

`scripts/YOLO11n_baseline.py` writes `data/dfire_yolo11n.yaml` with `train: train/images` and `val: test/images`, then evaluates with `eval_model.val(..., split="val")`. **Consequence:** the D-Fire **`test` folder is passed to Ultralytics as the `val` split** and used as the evaluation set. There is **no separate third (development/validation) split** in the repository. This must be described accurately: PyroFinder uses D-Fire's two-way train/test split, with the `test` folder mapped to Ultralytics' `val` role for evaluation. Do **not** claim a three-way split exists.

> This two-way arrangement means there is no independent development split for tuning. Any repeated tuning decisions taken against the `test` (mapped to `val`) set are a **leakage risk** and should be disclosed in the results write-up.

---

## 13. Required Streamlit input → output flow

Official flow, translated to PyroFinder:

```text
User uploads an RGB image
→ selected fine-tuned detector runs
→ annotated image is displayed
→ fire/smoke detections are listed
→ confidence scores and detection counts are shown
```

For the final M3 submission the following are **required**:

- the **selected measured detector** must be runnable;
- the detector must use a **fine-tuned D-Fire checkpoint**;
- the app must **not** silently substitute generic pretrained `yolo11n.pt` / `yolo11s.pt`;
- heavy ML libraries and weights must be **loaded lazily**;
- model training must **not** run inside Streamlit;
- a **missing checkpoint** must produce a clear user-facing message;
- an **invalid checkpoint class mapping** must fail clearly;
- outputs must be based on **actual inference**;
- synthetic detection boxes or static output images **do not** satisfy the requirement;
- only `fire` and `smoke` detections are allowed;
- the annotated image must correspond to the **uploaded** image;
- confidence values and counts must come from the **actual inference result**.

Three scopes:

### 13.1 Required M3 functionality

```text
Selected measured winner:
uploaded image → detector → annotated output and detection details
```

### 13.2 Useful comparison functionality

```text
YOLO11n and YOLO11s side by side
```

Useful **only** when both fine-tuned checkpoints and both measured result sets exist.

### 13.3 Optional M4 functionality

Production UX; cloud deployment; AI agents; advanced mapping; full alert operations; camera-stream integration; production monitoring interfaces.

**Repository evidence (today):** `app.py` Inference Demo (`tab_inference`) loads detectors only inside a `@st.cache_resource` function (`load_detector_cached`), called only when the user clicks **Run inference**. `src/inference.py` imports `ultralytics` lazily inside `load_detector`/`run_detection`, raises `FileNotFoundError` for a missing checkpoint (never substituting pretrained weights), and `validate_detector_classes` now requires the class set to equal exactly `{"fire", "smoke"}`. YOLO11s is offered only when `models/yolo11s_dfire_best.pt` exists; side-by-side appears only when both checkpoints exist. `run_detection` measures inference time during the call and returns real fire/smoke counts, max confidence, and the annotated PNG of the uploaded image.

---

## 14. Required GitHub and README evidence

The final repository must provide evidence for:

- sklearn training code (`scripts/dummy_try.py`, `scripts/simple_baselines.py`);
- YOLO11n training code (`scripts/YOLO11n_baseline.py`);
- YOLO11s training or reproducible training workflow (`notebooks/PyroFinder_YOLO11s_Kaggle_Training.ipynb` — present and tracked in Git; the notebook itself commits no result files — measured results are committed separately under `results/`);
- model-evaluation code;
- operational evaluation code (`scripts/evaluate_yolo_alert_metrics.py`, `src/evaluation.py`);
- result JSON / CSV files;
- Streamlit inference code (`app.py`, `src/inference.py`);
- dependency declarations (`requirements.txt`);
- run instructions;
- model-result summary;
- selected-winner explanation;
- clear checkpoint instructions.

Clarifications:

- Large model checkpoints **may remain Git-ignored** (`models/` is local-only by policy).
- If checkpoints are not committed, the **README must state exactly where** the required fine-tuned checkpoint must be placed (e.g. `models/yolo11s_dfire_best.pt`).
- The submitted Streamlit demonstration must still be **runnable in the intended submission environment**.
- A Streamlit interface that **cannot access its selected checkpoint does not prove the model runs**.
- The README **must not** claim a winner that is not supported by measured artifacts.

---

## 15. Current PyroFinder evidence matrix

Allowed status labels only: **Complete**, **Partially complete**, **Pending**, **Blocked**, **Not required for M3**.

Classifications are conservative: a requirement is **Complete** only with credible evidence the work actually occurred (training happened; real result artifacts exist; checkpoint present and inference verified). The existence of a source file, a UI section, a documented path, a Streamlit option, a placeholder row, or a running training notebook does **not** make an item Complete.

| Requirement | Status | Current evidence | Missing evidence or required action |
|---|---|---|---|
| Real sklearn baseline training | Complete | Three result JSONs on full D-Fire (train 17,221 / test 4,306) with class distributions | — |
| DummyClassifier result evidence | Complete | `results/baseline_dummy_classifier.json`: acc 0.47, macro F1 0.21, fire/smoke recall 0.00 | — |
| Logistic Regression result evidence | Complete | `results/baseline_logistic_regression.json` (PROJECT_CONTEXT: acc 0.6078, macro F1 0.6151, fire recall 0.7462, smoke recall 0.6661) | — |
| Random Forest result evidence | Complete | `results/baseline_random_forest.json`: acc 0.8579, macro F1 0.8486, fire recall 0.7973, smoke recall 0.8145 | — |
| Train/test separation (sklearn) | Complete | Result JSONs record train/test class_distribution; D-Fire pre-split used | Confirm scalers/models never fit on test (code review of `scripts/simple_baselines.py`) |
| YOLO11n training | Complete | `results/baseline_yolo11n.json` + `results/results_yolo11n.csv` (Kaggle, T4, 30 epochs, imgsz 640) | — |
| YOLO11n object-detection evaluation | Complete | `baseline_yolo11n.json`: mAP@0.5 0.747, mAP@0.5:0.95 0.4249, P 0.7397, R 0.6825, F1 0.7099 | — |
| YOLO11n per-epoch training results | Complete | `results/results_yolo11n.csv` (per-epoch curves) | — |
| YOLO11n operational alert evaluation | Complete | `results/yolo11n_operational_metrics.json`: TP 2147/FN 154/FP 42/TN 1963, Hazard Recall 0.9331, FAR 0.0209, Score 0.9368 | — |
| YOLO11n per-image prediction evidence | Complete | `results/yolo11n_test_predictions.csv` = 4,306 data rows + header; totals reconcile with JSON confusion counts | — |
| YOLO11n approximate fire-location evaluation | Complete | `location_metrics`: GT fire 1,115; coverage 1,020/1,115 (0.9148); mean 0.01343; median 0.005704; grid hit 0.9559 | — |
| YOLO11s training | Complete | Fine-tuned on Kaggle (T4, 30 epochs, imgsz 640, 2026-06-12); checkpoint produced | — |
| YOLO11s object-detection evaluation | Complete | `results/baseline_yolo11s.json`: mAP@0.5 0.7668, mAP@0.5:0.95 0.4414, P 0.7573, R 0.6967, F1 0.7257 | — |
| YOLO11s operational evaluation | Complete | `results/yolo11s_operational_metrics.json`: TP 2156/FN 145/FP 37/TN 1968, Hazard Recall 0.9370, FAR 0.0185, Score 0.9406 | — |
| YOLO11s per-image prediction evidence | Complete | `results/yolo11s_test_predictions.csv` present (per-image alert outcome + fire-location error table) | — |
| Same-task detector comparison (YOLO11n vs YOLO11s) | Complete | Both detectors measured under equivalent conditions; comparison rendered in `app.py` / `results_loader.py` | — |
| sklearn comparison table | Complete | `app.py` renders per-class metrics, distributions, and a model-comparison sub-tab | — |
| Object-detection comparison table | Complete | `app.py` "Object-detection comparison": YOLO11n and YOLO11s both measured | — |
| Operational comparison table | Complete | `app.py` `_render_operational_alert_metrics`: sklearn + YOLO11n + YOLO11s measured | — |
| Final detector winner selection | Complete | `select_operational_winner` selects YOLO11s from measured, complete files (wins Hazard Recall, FAR, Operational Alert Score) | — |
| Winner justification (written) | Complete | KPI hierarchy documented (§10); YOLO11s vs YOLO11n measured differences cited in PROJECT_CONTEXT §12.5 and README | — |
| Uploaded-image inference UI | Complete | `tab_inference` upload + Run inference button; Streamlit boots cleanly headless | — |
| Fine-tuned YOLO11n inference availability | Complete | `models/yolo11n_dfire_best.pt` present locally; `checkpoint_exists("YOLO11n")` true | Note: weight is Git-ignored; README must state placement |
| Fine-tuned YOLO11s inference availability | Complete (local) | `models/yolo11s_dfire_best.pt` produced by the Kaggle run; weight is Git-ignored | Note: weight is Git-ignored; README must state placement |
| Selected winner running in Streamlit | Complete | Selected detector (YOLO11s) runs end-to-end in the demo when its checkpoint is present; YOLO11n available as fallback | Note: YOLO11s weight is Git-ignored; README must state placement |
| Missing-checkpoint handling | Complete | `load_detector` raises `FileNotFoundError`; UI shows `MISSING_YOLO11S_MESSAGE`; never loads pretrained | — |
| Lazy model loading | Complete | `ultralytics` imported only inside `src/inference.py` functions; `@st.cache_resource` load on click | — |
| Prevention of generic pretrained-weight fallback | Complete | `CHECKPOINTS` maps only to fine-tuned paths; no fallback to `yolo11n.pt`/`yolo11s.pt` | — |
| Class-mapping validation (exactly fire/smoke) | Complete | `validate_detector_classes` requires set == `{"fire","smoke"}`; tests cover only-fire/only-smoke/extra/empty/malformed | — |
| Training code in GitHub | Complete | `scripts/dummy_try.py`, `scripts/simple_baselines.py`, `scripts/YOLO11n_baseline.py` | — |
| Evaluation code in GitHub | Complete | `src/evaluation.py`, `scripts/evaluate_yolo_alert_metrics.py`, `src/results_loader.py` | — |
| Result files in GitHub | Complete | sklearn + YOLO11n JSON/CSV committed under `results/` | — |
| Requirements / dependency evidence | Complete | `requirements.txt` (pandas, streamlit, plotly, scikit-learn, pytest, numpy, Pillow, opencv-python-headless, ultralytics, PyYAML, folium, streamlit-folium, shapely) | — |
| README run instructions | Complete | `README.md` Installation + run commands; data regeneration commands | — |
| README checkpoint instructions | Partially complete | README documents YOLO11n checkpoint as Git-ignored and YOLO11s expected path | Add explicit YOLO11s checkpoint placement for the submission environment when it exists |
| README measured-result summary | Complete | README summarizes sklearn + YOLO11n + measured YOLO11s detection and operational results | — |
| README selected-winner summary | Complete | README states YOLO11s is the measured, selected primary detector over YOLO11n | — |
| Final M3 submission evidence | Complete | sklearn + YOLO11n + YOLO11s fully measured; YOLO11n-vs-YOLO11s comparison and KPI-based winner justification present | — |

---

## 16. Missing or pending evidence

The critical-path YOLO11s items are now **satisfied**:

1. **Measured YOLO11s detection results** — `results/baseline_yolo11s.json` (present).
2. **Measured YOLO11s operational results** — `results/yolo11s_operational_metrics.json` (present).
3. **YOLO11s per-image predictions** — `results/yolo11s_test_predictions.csv` (present).
4. **YOLO11s per-epoch curves** — `results/results_yolo11s.csv` (present).
5. **YOLO11s fine-tuned checkpoint** — `models/yolo11s_dfire_best.pt` (produced; local-only, Git-ignored).
6. **Completed YOLO11n-vs-YOLO11s comparison** under equivalent conditions (done; rendered in `app.py`).
7. **Final KPI-based winner justification** referencing measured differences (done; YOLO11s selected — see §15 and PROJECT_CONTEXT §12.5).
8. **README updates** for YOLO11s measured results, selected winner, and checkpoint placement (done).

Remaining non-YOLO11s build work (alert log, N-frame confirmation, camera map, deployment) is tracked in `PROJECT_CONTEXT.md` §20.

Reproduce the YOLO11s operational/location metrics from the checkpoint (evaluation only, no training):

```bash
python scripts/evaluate_yolo_alert_metrics.py \
  --raw-root "<path-to-D-Fire-root>" \
  --weights "models/yolo11s_dfire_best.pt" \
  --model-name "YOLO11s" --conf 0.25 \
  --output-json "results/yolo11s_operational_metrics.json" \
  --output-csv "results/yolo11s_test_predictions.csv"
```

---

## 17. Synthetic-result and placeholder policy

- Synthetic values may be used **only** for temporary UI development.
- Every synthetic value must be **visibly labeled as synthetic**.
- Synthetic values must **not** be stored as production result files.
- Synthetic values must **not** appear in README result claims.
- Synthetic values must **not** appear in the official comparison table.
- Synthetic values must **not** be used to select a winner.
- Placeholder metrics must **not** be described as measured.
- Any detector row whose measured result file is genuinely absent must show **`Training in progress`** or **`Pending measured results`** (never invented values).
- A model **currently training** is not a completed candidate.
- A **checkpoint alone** is not sufficient evaluation evidence.
- **Training curves alone** are not sufficient winner-selection evidence.
- YOLO11s becomes eligible **only after** real detection **and** operational evaluation outputs exist.
- Result files with **missing required metrics** are incomplete.
- **Malformed or wrong-kind** result files are invalid submission evidence.
- **No number may be described as measured** unless it came from an actual evaluation run.

**Repository evidence (today):** YOLO11s now has measured, non-synthetic result files in `results/` (`baseline_yolo11s.json`, `yolo11s_operational_metrics.json`, `results_yolo11s.csv`, `yolo11s_test_predictions.csv`). `src/results_loader._NON_MEASURED_STATUS_KEYWORDS = ("synthetic", "placeholder", "pending", "training")` still blocks any synthetic/placeholder/pending file from selection; `tests/test_results_loader.py::test_synthetic_status_is_not_selectable` enforces it. The **`Training in progress`** status is now reserved for any detector whose measured result file is genuinely absent.

---

## 18. M3 acceptance criteria

### 18.1 Predictive evidence

- [ ] Real models were trained on D-Fire.
- [ ] At least two suitable **same-task** candidates are available for the final detector comparison.
- [ ] YOLO11n and YOLO11s have **measured** results under comparable conditions.
- [ ] Test-set evidence exists.
- [ ] No synthetic result is used.

### 18.2 Comparison evidence

- [ ] sklearn baselines are compared only with **compatible classification metrics**.
- [ ] YOLO11n and YOLO11s are compared with **compatible detection metrics**.
- [ ] Operational metrics are clearly **separated** from detection metrics.
- [ ] Approximate location metrics are clearly **separated** from both.
- [ ] The winner is selected using the **documented KPI hierarchy**.
- [ ] The written explanation refers to **measured differences**.

### 18.3 Streamlit evidence

- [ ] A user can upload a real RGB image.
- [ ] The selected fine-tuned detector runs.
- [ ] The annotated result appears.
- [ ] Fire and smoke detections are shown.
- [ ] Confidence values and counts are shown.
- [ ] No generic pretrained checkpoint is silently loaded.
- [ ] Missing weights produce a clear message.
- [ ] Training does not occur in Streamlit.

### 18.4 Repository evidence

- [ ] Training and evaluation code are present.
- [ ] Measured result artifacts are present.
- [ ] README explains local execution.
- [ ] README explains checkpoint setup.
- [ ] README summarizes measured results.
- [ ] README names the selected winner and explains why.
- [ ] The repository does not contain fabricated metrics or committed large weights contrary to project policy.

> **Current gate status:** §18.1 and §18.2 are satisfied — YOLO11s is measured and the YOLO11n-vs-YOLO11s comparison is complete (YOLO11s selected). §18.3 runs on the selected detector (YOLO11s) when its checkpoint is present, with YOLO11n as fallback. §18.4 is satisfied for results/README (sklearn + YOLO11n + YOLO11s measured; selected winner documented); model weights remain Git-ignored per policy.

---

## 19. Common submission risks

- Submitting **only the descriptive dashboard** (M2 deliverable) and treating it as M3.
- Comparing **sklearn Macro F1 directly with YOLO11 mAP**.
- Claiming **YOLO11s is the winner before measured results exist**.
- Relying on a **training-in-progress notebook** as if it were a result.
- Selecting a winner from **one measured detector** and calling the comparison complete.
- Using **only training metrics** (no test-set evaluation).
- **Evaluating on data used for training** (leakage); note the two-way split has no separate dev set (§12.1).
- Using **undocumented confidence thresholds** (document conf 0.25, imgsz 640).
- Using **synthetic values** in final tables.
- **Silently using generic pretrained weights** in the inference demo.
- Having a **Streamlit inference UI without an accessible fine-tuned checkpoint**.
- Showing **side-by-side model options when one model cannot run**.
- Reporting **estimated inference speed as measured**.
- Claiming **precise geolocation** from normalized image-space error.
- **Documenting June 23 instead of the official June 16 deadline** (stale-doc risk, §3).
- Leaving **README statements inconsistent** with the actual result files.

---

## 20. Pre-submission verification checklist

- [ ] **Deadline and format:** Tuesday, June 16, 2026; Streamlit submission. No stale June 23 date presented as the M3 deadline.
- [ ] **Repository cleanliness:** no fabricated metrics; no committed large weights; `models/` Git-ignored.
- [ ] **Real result files:** all reported numbers trace to a committed JSON/CSV from an actual run.
- [ ] **YOLO11s completion:** detection + operational + per-image + per-epoch files exist and are measured (or YOLO11s is honestly marked pending and not declared winner).
- [ ] **Candidate comparability:** same-task detectors compared with detection metrics; sklearn compared only with classification metrics.
- [ ] **Train/test separation:** training on train split; final metrics on unseen test; two-way split (test→val) disclosed.
- [ ] **Metric-family separation:** detection, operational, and location metrics in separate tables.
- [ ] **Final winner selection:** chosen by the documented KPI hierarchy from measured, complete files only.
- [ ] **Winner explanation:** written, KPI-based, cites measured differences.
- [ ] **Fine-tuned checkpoint availability:** selected detector's checkpoint reachable in the submission environment.
- [ ] **Successful uploaded-image inference:** real upload → real detector → real output.
- [ ] **Annotated output:** corresponds to the uploaded image.
- [ ] **Confidence and count output:** from the actual inference result.
- [ ] **Missing-weight behavior:** clear user-facing message; no pretrained fallback.
- [ ] **README run instructions** present and correct.
- [ ] **README result summary** matches the committed result files.
- [ ] **README winner summary** present and supported by artifacts.
- [ ] **No synthetic values** anywhere in final tables, README, or selection.
- [ ] **No unsupported claims** (no precise geolocation, no early-warning wording, no fire-spread prediction).
- [ ] **No stale deadline** in the submitted M3 materials.
- [ ] **No incompatible metric ranking** (no sklearn-vs-YOLO direct metric ranking).

---

## 21. Source-of-truth files

| File | Role in this document |
|---|---|
| Official M3 requirements (this milestone) | Course submission rules and authoritative deadline (June 16, 2026). |
| `PROJECT_CONTEXT.md` | Canonical product/ML scope, dataset counts, KPI, current M3 results. |
| `CLAUDE.md` | Repository structure, module responsibilities, current coding-agent status. |
| `README.md` | Public-facing description, run instructions, measured-result summary (sklearn + YOLO11n + YOLO11s). |
| `ASSISTANT_WORKING_RULES.md` | Assistant behavior and accuracy rules. |
| `docs/AI_AGENT_SYSTEM.md` | Agent roles and workflows (supporting; flag stale statements). |
| `src/evaluation.py` | Source of truth for cost weights, operational score formula, and location helpers. |
| `src/results_loader.py` | Result loading, status classification, selectability, and winner selection. |
| `src/inference.py` | Lazy detector loading, fine-tuned-only checkpoints, exact fire/smoke class validation. |
| `app.py` | Streamlit dashboard: separate comparison tables, operational metrics, inference demo. |
| `scripts/simple_baselines.py`, `scripts/YOLO11n_baseline.py`, `scripts/evaluate_yolo_alert_metrics.py` | Training and evaluation code. |
| `results/baseline_*.json`, `results/results_yolo11n.csv`, `results/yolo11n_operational_metrics.json`, `results/yolo11n_test_predictions.csv` | Measured result artifacts. |
| `tests/test_results_loader.py`, `tests/test_inference.py`, `tests/test_evaluation.py` | Tests locking the result-loading, inference-guard, and metric behaviors. |

---

*PyroFinder · Technion Course 016833 · Location-Based Services: Data Science · M3 submission requirements · Deadline: Tuesday, June 16, 2026.*

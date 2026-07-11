# PyroFinder — Project Context

**Canonical source of truth** for product scope, ML, data, and architecture. If another
file conflicts with this one, this file wins (unless the owners update it). For
repo/code status see `CLAUDE.md`; for assistant behavior see `ASSISTANT_WORKING_RULES.md`.

**Status:** M3 complete — sklearn baselines, YOLO11n baseline, and the selected YOLO11s
primary detector are all measured; the Live Ops operator dashboard has shipped.

- **Live app:** https://pyrofinder.streamlit.app/ · **Run:** `streamlit run app.py`
- **Course:** Technion 016833 — Location-Based Services: Data Science.

---

## 1. What PyroFinder is

Real-time `fire`/`smoke` detection that reuses cameras a site already owns — no new
hardware. Pipeline: sample frames → **YOLO11s** detection → multi-frame confirmation →
approximate location → operational alert. It is a monitoring/alerting system built on
detection outputs, not a pure YOLO demo.

**One-liner:** property owners in fire-prone areas suffer delayed fire awareness;
PyroFinder turns their existing cameras into a real-time fire/smoke monitoring layer with
multi-frame confirmation and approximate map-based alerts.

## 2. Problem & audience

Existing wildfire solutions need towers, sensors, drones, or public infrastructure.
PyroFinder fills the gap for individual owners: homeowners, farm/ranch owners,
agricultural facility managers, private landowners. Their cameras are passive — someone
must watch every feed. **Persona — Dani:** a farm owner with boundary cameras; during dry
months a fire can start at a field edge or neighbouring land, and Dani can't watch every
feed, so PyroFinder alerts on a confirmed detection and shows roughly where. Secondary
users (municipalities, fire/rescue) may receive approximate alert info for shared areas.

## 3. Product surfaces

- **Live Ops dashboard — the default landing surface** (`pages/1_Live_Ops.py`). Three
  views: **Setup** (place camera on a map, calibrate image↔map anchors, mark detection
  zones with the AI zone assistant + optional per-zone reference point), **Live**
  (autoplay a camera sequence, per-class smoke/fire detection, N-frame confirmation,
  freeze-on-confirm, operational chat), **History** (filterable event log).
- **Central Control dashboard** — operator setup/history: camera metadata, map reference
  points, image zones, export/import config, incident assistant, weather risk advisory.
- **Operations & Learning dashboard** — internal ML tool: dataset inspection, D-Fire EDA,
  inference demo, model comparison, evaluation metrics, false-alarm review.
- **M2 / M3 / M4 dashboards** — milestone views (problem/EDA, model results, operator flow).
- **Mobile customer app** and **emergency viewer** — future / out of scope this semester.

## 4. What PyroFinder is NOT

- Not an "early warning system"; does not predict physical fire spread.
- No precise geolocation; no automatic image-to-map registration; no true geographic
  spread direction (only apparent in-frame movement).
- No emergency-dispatch integration; the assistant only drafts/recommends — never contacts
  anyone or dispatches automatically.
- No new/dedicated hardware; no live RTSP production streaming required this semester.
- Classes are strictly `fire` and `smoke`. No YOLOv12; no generic "YOLO" wording.

---

## 5. ML problem

- **Task:** two-class object detection. Input: RGB frames resized to 640×640. Output per
  frame: boxes `(x_center, y_center, w, h)` normalized, class `fire`/`smoke`, confidence.
- **Primary model:** Ultralytics **YOLO11s** fine-tuned on D-Fire — chosen for
  near-real-time sampled-frame inference with stronger quality than YOLO11n. **Fallback:**
  **YOLO11n** (lightweight speed baseline; not an equal parallel model).
- **Loss:** Ultralytics YOLO detection loss (box regression + classification + DFL).
- **Primary KPI:** the cost-sensitive **Operational Alert Score** (FN weight 10, FP weight
  1) — at the alert level each image is reduced to hazard detected / not detected, and a
  missed hazard is 10× worse than a false alarm. It already encodes its components
  **Hazard Recall** (FN-driven) and **False Alert Rate** (FP-driven); those are reported as
  diagnostics, not separate ranking tiers. Detection mAP@0.5 / Recall / F1 and inference
  speed are supporting metrics.
- **Split:** D-Fire's provided train/test split.

## 6. Results (measured — no synthetic values anywhere)

All numbers are measured on the **D-Fire test split (4,306 images)**, confidence 0.25 for
the operational evaluation (evaluation only, no training). YOLO11s is the **selected**
detector: it wins the primary KPI and leads on its components and on supporting detection
metrics. Full analysis: `docs/M3_RESULTS_SUMMARY.md`; raw files in `results/`.

**Object detection (boxes/classes — never compared to sklearn accuracy):**

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| YOLO11s (primary) | **0.7668** | 0.4414 | 0.7573 | 0.6967 | 0.7257 |
| YOLO11n (fallback) | 0.7470 | 0.4249 | 0.7397 | 0.6825 | 0.7099 |

YOLO11s per-class: smoke mAP@0.5 0.8222 / fire 0.7115. Training: 30 epochs, image size
640, batch 16, Kaggle Tesla T4.

**Operational alert level (`fire` or `smoke` = hazard; image-level):**

| Model | Hazard Recall | False Alert Rate | Alert F1 | Operational Alert Score |
|---|---:|---:|---:|---:|
| YOLO11s | **0.9370** | 0.0185 | 0.9595 | **0.9406** |
| YOLO11n | 0.9331 | 0.0209 | 0.9563 | 0.9368 |
| Random Forest* | 0.8779 | 0.0828 | — | 0.8810 |
| Logistic Regression* | 0.8057 | 0.5037 | — | 0.7809 |
| DummyClassifier* | 0.0000 | 0.0000 | — | 0.0802 |

\* sklearn baselines are **image-level classifiers** (60-dim colour features, no boxes);
kept only as a reference floor — they can't replace YOLO11s. Detection classification
scores (full D-Fire): Random Forest ≈ 0.858 acc / 0.849 macro-F1, Logistic Regression ≈
0.608 / 0.615, Dummy 0.47 / 0.21.

**Approximate fire-location** (bottom-center anchor of class-1 fire boxes; image-space
only, never precise geolocation): YOLO11s coverage 1,040/1,115, mean error 0.0135, 3×3
grid-hit 0.9644 (YOLO11n 1,020/1,115, 0.0134, 0.9559). Smoke-only images are never treated
as a fire epicenter.

Result files: `results/baseline_{dummy_classifier,logistic_regression,random_forest,yolo11n,yolo11s}.json`,
`results/*_operational_metrics.json`, `results/*_test_predictions.csv`. Reproduce YOLO
operational/location metrics (evaluation only):

```bash
python scripts/evaluate_yolo_alert_metrics.py --raw-root "<D-Fire root>" \
  --weights models/yolo11s_dfire_best.pt --model-name YOLO11s --conf 0.25 \
  --output-json results/yolo11s_operational_metrics.json \
  --output-csv results/yolo11s_test_predictions.csv
```

---

## 7. Data

**D-Fire** (https://github.com/gaia-solutions-on-demand/DFireDataset, CC0). Verified class
mapping: **0 = smoke, 1 = fire** (do not invert). The committed `data/dfire_metadata.csv`
(21,527 rows × 36 cols) lets the app run on a fresh clone with no raw dataset.

| Split / category | Count |
|---|---:|
| Total / Train / Test | 21,527 / 17,221 / 4,306 |
| Background / Smoke-only / Fire-only / Fire+smoke | 9,838 / 5,867 / 1,164 / 4,658 |
| Fire boxes / Smoke boxes | 14,692 / 11,865 |

EDA highlights: background is the largest class (accuracy alone is misleading); fire-only
images have a much higher dark-pixel ratio than smoke-only (~64% vs ~9%); smoke boxes are
~7× larger than fire boxes; in fire+smoke images the smoke centroid sits above the fire
centroid ~95% of the time. Known gaps: few night/indoor/close-range scenes; smoke confused
with cloud/fog/haze/glare; real deployment needs validation beyond D-Fire.

Regenerate metadata: `python scripts/build_dfire_metadata.py --raw-root "<D-Fire root>" --output data/dfire_metadata.csv`.

**Supplementary / validation** (verify + normalize to `fire`/`smoke` before use): Smart
Fire System (supplementary/external), Aerial Rescue OD (robustness; Fire class only,
vehicle/human as negatives), Fire Detection in YOLO Format (small, after class check), FURG
Fire (video/temporal validation). Large raw datasets stay outside Git.

## 8. Detection, tracking, mapping

- **Detection** frame-by-frame with fine-tuned YOLO11s (fire/smoke only). The Live tab uses
  **separate smoke/fire confidence thresholds** (default 0.40 each in `config/live_ops.yaml`).
- **Confirmation:** no alert from a single frame. Live uses one-miss-tolerant confirmation
  (`tracking.is_confirmed_with_tolerance`): the current frame must be positive AND ≥ N of the
  last N+1 frames positive. `N` is configurable (`confirmation_frames`, shipped default 1).
- **Location (approximate):** named image polygon → image quadrant → camera-projected map
  point (reference-point homography). A class-1 fire box is anchored bottom-center. Inside a
  zone with an operator-set reference point, that point is projected; a matched zone without
  one gets no invented point; otherwise the detection anchor is projected. Never precise
  geolocation.
- **Direction:** apparent image-plane movement + wind-driven downwind direction (Open-Meteo).
  Never claimed as true fire spread.
- **Operational context** (`data/live_ops/live_ops_operational_context.{json,md}`): optional
  landmarks, sensitive receptors, and verified authority contacts + a contact policy. Feeds
  incident reasoning and first-message wording only — never zone metadata, never detection
  input; missing degrades gracefully.
- **Incident messaging** is audience-relevant: camera by **name** (no ID), never confidence
  (already confirmed), never temperature/humidity/wind-speed (direction only); approximate
  **coordinates only for field responders** (fire department). It uses a verified contact's
  phone when the context has one, else offers to search (never auto). Drafts/recommends only.

## 9. Mapping strategy

Mapping is an **offline, pre-event setup stage** (operational config, not YOLO training
data). Modes: responsibility zones, named image polygons, image↔map polygon/point linking,
manual/GPS camera location, camera metadata (height/azimuth/FOV/indoor-outdoor),
reference-point mapping. Libraries: Folium, streamlit-folium, Shapely (GeoPandas/PyProj only
if advanced GIS is added later). Automatic image-to-map registration is future work.

---

## 10. Repository structure

```text
app.py                       Streamlit shell (theme, sidebar, dispatch); new sessions open Live Ops
pages/1_Live_Ops.py          Live Ops default landing page

src/
  inference.py               lazy YOLO11n/YOLO11s + run_detection (per-class conf) + hazard/anchor helpers
  tracking.py                N-frame confirmation (strict + one-miss) + apparent direction
  mapping.py                 polygons, quadrant, reference-point homography, zone ref points, downwind
  incident_agent.py          Incident Assistant: context, concise/audience-relevant messages, contacts
  live_ops_config.py         Live Ops config + demo assets + operational-context loader + FOV cone
  live_ops_cache.py          pre-computed per-frame detection cache (redraw boxes, no YOLO in default mode)
  live_ops_agents.py         dual-agent ops chat (Watch weather / Response incident)
  zone_agent.py, agent_schemas.py   text→operational zone records; priority/zone-type/compass helpers
  segmentation_assist.py     local OpenCV GrabCut box→polygon (no weights/network/YOLO/Groq)
  weather.py                 Open-Meteo (no key) + offline mock; fire-weather risk
  llm.py                     Groq helper (lazy); reads GROQ_API_KEY; not the detector
  evaluation.py, results_loader.py   operational metrics + result JSON loading/winner (pure)
  data.py, eda.py, viz.py, model.py, detection.py, alerts.py, ui.py   dataset/EDA/annotation/etc.
  dashboards/                live_ops · central_control · operations_learning · m2(+m2/) · m3(+m3/) · m4 · model_helpers

config/                      live_ops.yaml, live_ops_camera.json
data/                        dfire_metadata.csv · samples/ · live_demo/ (26 frames + cache/detections.json) · live_ops/ (context) · live_events.jsonl (History, git-ignored)
results/                     measured metric JSON/CSV (baselines, YOLO11n, YOLO11s)
models/                      yolo11n_dfire_best.pt, yolo11s_dfire_best.pt  (COMMITTED — gitignore exception)
scripts/                     build_dfire_metadata, dummy_try, simple_baselines, YOLO11n_baseline,
                             evaluate_yolo_alert_metrics, build_live_ops_cache
tests/                       pytest suite (pure helpers incl. test_live_ops_*; no weights required)
docs/                        M2_DATA_EDA, M3_RESULTS_SUMMARY, M3_SUBMISSION_REQUIREMENTS,
                             AI_AGENT_SYSTEM, Literature_review, market survey, design_handoff
```

## 11. Runtime, secrets, and policy

- **Dependencies** (`requirements.txt`): streamlit, pandas, plotly, numpy, Pillow,
  opencv-python-headless, scikit-learn, ultralytics, PyYAML, folium, streamlit-folium,
  streamlit-image-coordinates, shapely, groq, truststore, pytest.
- **Secrets:** `GROQ_API_KEY` is the only optional key (AI zone setup + incident-chat
  wording) — read from `st.secrets`/env, never logged or committed; absent → deterministic
  local fallback, and `groq` imports lazily. Weather (Open-Meteo) needs **no** key.
- **Do not commit:** raw datasets, training runs, secrets, local machine paths, large
  media, caches. **Committed exceptions:** `data/dfire_metadata.csv`, sample/demo assets,
  `results/`, docs, and the two fine-tuned checkpoints in `models/` (so the public app runs
  on a fresh clone).
- **Commands:** `python -m pytest tests` after code changes; `streamlit run app.py` after
  layout changes.

## 12. User stories (acceptance-tested)

1. Confirmed alert — YOLO11s detects fire/smoke above threshold across N frames → dashboard
   shows a confirmed alert with camera, time, class, and approximate location.
2. Operator map — cameras appear as map markers; clicking shows status/metadata.
3. Image polygons — operator draws a named zone; a test detection inside returns the name.
4. Model comparison — dashboard shows detection vs operational metrics (never sklearn
   accuracy for YOLO).
5. False-alarm review — test alerts can be confirmed / rejected / marked false alarm.

Market gap: a low-friction **software** layer for sites that already have cameras (vs
tower/acoustic/drone/satellite/suppression competitors). Don't invent market-size numbers.

## 13. Documentation map

| File | Role |
|---|---|
| `PROJECT_CONTEXT.md` | Product/ML/data/architecture source of truth (this file). |
| `CLAUDE.md` | Coding-agent repo map + current status. |
| `README.md` | External-facing overview. |
| `ASSISTANT_WORKING_RULES.md` | Assistant communication/coding/session rules. |
| `docs/AI_AGENT_SYSTEM.md` | Agent roles, workflows, prompts. |
| `docs/M2_DATA_EDA.md` | D-Fire workflow, class mapping, EDA. |
| `docs/M3_RESULTS_SUMMARY.md` | Detailed M3 result analysis. |
| `docs/Literature_review.md`, `docs/market_survey_wildfire_existing_sensors.md` | Related work / market. |

Update rule: product/model/data/terminology → this file; repo/code status → `CLAUDE.md`;
agent workflows → `docs/AI_AGENT_SYSTEM.md`; assistant behavior → `ASSISTANT_WORKING_RULES.md`;
public claims → `README.md`.

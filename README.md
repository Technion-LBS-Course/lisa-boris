# PyroFinder

**Real-time fire and smoke detection that turns cameras a site already owns into an automated monitoring layer — no new hardware.**

- **Live app:** https://pyrofinder.streamlit.app/
- **Run locally:** `streamlit run app.py`

Course project for Technion 016833 — Location-Based Services: Data Science (team: Lisa & Boris).

---

## The problem

Property owners in fire-prone areas (homes, farms, ranches, agricultural sites) already have security cameras, but those cameras are passive — someone has to watch every feed to notice smoke or fire. PyroFinder watches the feeds automatically: it detects `fire` and `smoke` with a fine-tuned **YOLO11s** detector, confirms detections across several frames to cut false alarms, estimates an **approximate** event location, and raises an alert with operational context.

**Persona — Dani:** a farm owner with boundary cameras. During dry months a fire can start at a field edge or neighbouring land; Dani can't watch every feed, so PyroFinder alerts when fire or smoke is confirmed and shows roughly where.

---

## What it does

The app opens on the **Live Ops** dashboard (`pages/1_Live_Ops.py`), a three-view operator flow:

- **Setup** — place the camera on a map; calibrate image↔map anchors; mark detection zones with an AI zone assistant (describe an area → local GrabCut segmentation proposes a polygon) and optionally attach a per-zone map reference point.
- **Live** — autoplays a demo camera sequence, runs YOLO11s per frame with **separate smoke/fire confidence thresholds**, and freezes on an N-frame-confirmed detection until the operator resolves it **in the ops chat** (confirm / false alarm). The assistant reasons from operational context and drafts messages conversationally.
- **History** — a filterable log of resolved events.

The classic multi-dashboard shell stays available from the sidebar: **M4**, **M3**, **M2**, **Operations & Learning** (dataset/EDA/inference/model comparison), and **Central Control** (camera metadata, mapping, zone setup, incident assistant, risk advisory).

---

## Model results (measured — no synthetic values)

Two-class object detection on the D-Fire test split (4,306 images). YOLO11s is the selected primary detector; YOLO11n is the lightweight speed baseline/fallback.

| Metric | YOLO11s | YOLO11n |
|---|---:|---:|
| mAP@0.5 | **0.7668** | 0.7470 |
| Precision / Recall / F1 | 0.757 / 0.697 / 0.726 | 0.740 / 0.683 / 0.710 |
| Operational Alert Score (primary KPI) | **0.9406** | 0.9368 |
| Hazard Recall / False Alert Rate | 0.9370 / 0.0185 | 0.9331 / 0.0209 |

The primary KPI is the cost-sensitive **Operational Alert Score** (a missed hazard is weighted 10× a false alarm). Full analysis: [`docs/M3_RESULTS_SUMMARY.md`](docs/M3_RESULTS_SUMMARY.md); raw numbers in [`results/`](results/). Classical baselines (Random Forest ≈ 0.85 macro-F1, etc.) are image-level classifiers kept only as a reference — they produce no boxes and never replace YOLO11s.

---

## Data

Primary dataset: **D-Fire** (21,527 images; train 17,221 / test 4,306; CC0). Verified class mapping: **0 = smoke, 1 = fire**. The committed [`data/dfire_metadata.csv`](data/dfire_metadata.csv) lets the app run on a fresh clone with no raw dataset. Details: [`docs/M2_DATA_EDA.md`](docs/M2_DATA_EDA.md) and [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

---

## Install & run

```bash
git clone https://github.com/Technion-LBS-Course/lisa-boris.git
cd lisa-boris
pip install -r requirements.txt
streamlit run app.py
```

The two fine-tuned checkpoints (`models/yolo11s_dfire_best.pt`, `models/yolo11n_dfire_best.pt`) are **committed** (a `.gitignore` exception, ~24 MB total), so inference runs on a fresh clone.

- **`GROQ_API_KEY`** (optional): powers the AI zone assistant and incident-chat wording. Without it, both fall back to deterministic local logic. Set it in `.streamlit/secrets.toml` or the environment; never commit it.
- Weather uses **Open-Meteo** (no key, no signup); if live weather is unavailable it falls back to a labelled offline mock.

Run the tests: `python -m pytest tests`

---

## Repository map

```text
app.py                  Streamlit shell: theme, sidebar, dispatch to dashboards
pages/1_Live_Ops.py     Live Ops default landing page
src/                    pure logic: inference, tracking, mapping, evaluation, agents…
src/dashboards/         one renderer per dashboard (live_ops, central_control, m2–m4, …)
config/                 live_ops.yaml + prepared camera mapping config
data/                   dfire_metadata.csv, samples, live_demo (frames + detection cache),
                        live_ops (operational context)
results/                measured metric JSON/CSV
models/                 committed fine-tuned YOLO11n/YOLO11s checkpoints
docs/                   data/EDA, results summary, agent system, market/literature
tests/                  pytest suite (pure helpers; no weights required)
```

Full module-by-module map: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) and [`CLAUDE.md`](CLAUDE.md).

---

## Scope

All locations are **approximate** (image polygon, quadrant, or camera-projected map point) — never precise geolocation. PyroFinder is **not** an early-warning system, does not predict physical fire spread, and never contacts anyone or dispatches automatically: the incident assistant only **drafts and recommends**. Classes are strictly `fire` and `smoke`.

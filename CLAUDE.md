# PyroFinder — Claude Code Context

Coding-agent context: repo map + current status. **Product/ML source of truth is
`PROJECT_CONTEXT.md`; assistant behavior rules are in `ASSISTANT_WORKING_RULES.md`.**
Don't re-host metric numbers here — point to `PROJECT_CONTEXT.md` §Results and `results/`.

PyroFinder turns a site's existing cameras into automated `fire`/`smoke` monitoring:
YOLO11s detection → multi-frame confirmation → approximate location → operational alert.

## Hard constraints

- **YOLO11s** primary detector, **YOLO11n** lightweight fallback. Name the version — never generic "YOLO", never YOLOv12.
- Classes are strictly `fire` and `smoke`. Don't train or add others.
- No heavy ML imports at module load — `ultralytics`/`torch` load lazily inside `src/inference.py`; `cv2` inside `src/segmentation_assist.py` / `src/live_ops_cache.py`; `groq` inside `src/llm.py`.
- All locations are **approximate** (image polygon / quadrant / camera-projected point) — never precise geolocation, never automatic image-to-map registration.
- Not an "early warning system"; no fire-spread prediction; the incident assistant only drafts/recommends and never contacts or dispatches automatically.

## App shell

- `app.py` — thin shell: page config, theme, sidebar, dispatch. New sessions `switch_page` to the Live Ops page; the classic sidebar shell (M4 / M3 / M2 / Operations & Learning / Central Control) stays reachable.
- `pages/1_Live_Ops.py` — Live Ops default landing page → `src/dashboards/live_ops.render()`.

## Repository map

### `src/` — pure logic (importable without heavy ML unless called)
- `inference.py` — lazy YOLO11n/YOLO11s loading + `run_detection` (optional per-class `conf_by_class`); `top_hazard_detection`, `select_confirmed_event_detection`, `bbox_bottom_center_norm`. Fine-tuned D-Fire checkpoints only.
- `tracking.py` — N-frame confirmation: strict (`is_confirmed_detection`) and one-miss-tolerant (`is_confirmed_with_tolerance`); apparent-direction estimation.
- `mapping.py` — polygon/quadrant helpers, reference-point homography (`estimate_map_position`), per-zone reference-point helpers, downwind geometry. No Streamlit/ML.
- `incident_agent.py` — Incident Assistant (pure). `build_incident_context` → `IncidentContext` (matched/nearest zones, approximate map point, apparent direction, wind/downwind, optional operational context). `format_initial_incident_message`/`initial_incident_message` (concise opener), `incident_reasoning` (why-on-ask), `respond_to_operator` + `build_incident_system_prompt` (per-recipient relevance: camera by name, no confidence/temperature/humidity/wind-speed, coordinates for the fire department only), draft messages, `contact_guidance`/`contact_clause`/`_preferred_contact` (verified-phone-or-offer-to-search), `create_incident_alert`. Optional Groq wording; degrades to deterministic text.
- `live_ops_config.py` — Live Ops config + demo assets: `load_settings` (`config/live_ops.yaml`), `load_camera_config`, `load_reference_frame`, `demo_sequence_items`, `load_operational_context` (optional JSON+MD), `approx_fov_cone`. Import-safe.
- `live_ops_cache.py` — pre-computed per-frame detection cache: `build_sequence_frames`, fingerprint helpers, manifest load/save/`is_valid`, `build` (detection injected — no ML import), `result_from_summary`/`annotate` (redraw boxes with PIL).
- `live_ops_agents.py` — dual-agent ops-chat wrapper: Watch (weather/risk via `src/weather.py`) + Response (incident via `incident_agent`). Never sends anything.
- `zone_agent.py` + `agent_schemas.py` — free-text → operational image-zone records (Groq or local); priority/zone-type/prompt-injection/compass helpers.
- `segmentation_assist.py` — local OpenCV GrabCut box→polygon refinement (no weights, no network, never YOLO/Groq).
- `weather.py` — Open-Meteo (no key) + offline mock; fire-weather risk advisories.
- `llm.py` — Groq helper (lazy `groq`); reads `GROQ_API_KEY` from `st.secrets`/env. Not the detector.
- `evaluation.py` / `results_loader.py` — operational alert metrics + result JSON loading/winner selection (pure stdlib, no ML).
- `data.py`, `eda.py`, `viz.py`, `model.py`, `detection.py`, `alerts.py`, `ui.py` — dataset/EDA/annotation/model-metadata/detection-dataclass/alert-record/UI-theme helpers.

### `src/dashboards/` — one renderer per dashboard
- `live_ops.py` — **default landing.** Setup (place camera / calibrate anchors / zones) · Live (autoplay + confirm) · History. Reuses Central Control `cc_*` state; no module-level ML.
- `central_control.py` — operator dashboard (camera metadata, map reference points, image zones, export/import, incident assistant, risk advisory).
- `operations_learning.py`, `m2_dashboard.py`+`m2/`, `m3_dashboard.py`+`m3/`, `m4_dashboard.py`, `model_helpers.py`.

### Other
- `config/` — `live_ops.yaml` (per-class thresholds, `confirmation_frames`, paths), `live_ops_camera.json` (camera + anchors + zones).
- `data/` — `dfire_metadata.csv`, `samples/`, `live_demo/` (26 committed frames + `cache/detections.json`), `live_ops/` (operational-context JSON+MD), `live_events.jsonl` (History log, git-ignored).
- `models/` — `yolo11n_dfire_best.pt`, `yolo11s_dfire_best.pt` are **committed** (gitignore exception) so a fresh clone runs inference.
- `scripts/` — `build_dfire_metadata.py`, `dummy_try.py`, `simple_baselines.py`, `YOLO11n_baseline.py`, `evaluate_yolo_alert_metrics.py`, `build_live_ops_cache.py`.
- `results/` — measured metric JSON/CSV (see `PROJECT_CONTEXT.md` §Results).
- `tests/` — pure-helper tests incl. `test_live_ops_agents.py`, `test_live_ops_cache.py`, `test_live_ops_config.py`, `test_live_ops_dashboard.py`; no weights required.
- `docs/` — `M2_DATA_EDA.md`, `M3_RESULTS_SUMMARY.md`, `M3_SUBMISSION_REQUIREMENTS.md`, `AI_AGENT_SYSTEM.md`, `Literature_review.md`, market survey, `design_handoff_pyrofinder_streamlit/`.

## Live Ops pipeline (current behavior)

- **Setup 1–3:** place camera on a map (+ approximate FOV cone); calibrate image↔map anchors as clickable cards; mark zones via the chat zone assistant — describe an area, click two corners, local GrabCut proposes a polygon (auto-applied in the same run), draw manually if it's off. Saved zones are clickable cards (rename / priority / delete); after saving, an optional per-zone map reference point can be added.
- **Live:** autoplays the demo sequence; **separate smoke/fire confidence** sliders (default 0.40 each) + `Confirmation frames (N)` (shipped default 1, one-miss tolerance). In **default mode** (sliders at defaults) it serves the committed disk cache (`data/live_demo/cache/detections.json`) and redraws boxes — no YOLO, no flicker; off-default runs YOLO live. On confirmation it freezes on the frame; the operator resolves it in the **ops chat** (confirm / false alarm — no dispatch panel); messages are conversational, audience-relevant, and grounded in the operational context.
- **History:** events from `data/live_events.jsonl` — date/type filter, per-day chart, clip view.

## Data & status

- D-Fire: 21,527 images (train 17,221 / test 4,306); verified mapping **0 = smoke, 1 = fire**. Fresh clone runs on `data/dfire_metadata.csv` alone. Details: `PROJECT_CONTEXT.md` §Data, `docs/M2_DATA_EDA.md`.
- M3 complete: sklearn baselines, YOLO11n baseline/operational, YOLO11s fine-tuned + selected primary (measured; no synthetic values). Numbers: `PROJECT_CONTEXT.md` §Results and `results/`.

## Conventions & commands

Python 3.10+, small testable functions, English names, comments only where the *why* is non-obvious. After code changes: `python -m pytest tests`. After layout changes: `streamlit run app.py`.

# PyroFinder — Live-Ops Dashboard: Implementation Plan

Plan for the new Streamlit dashboard described in `README.md` + `REQUIREMENTS.md`
(design variant **1b**, light tablet). Grounded in the current codebase. **No
existing code is modified** — everything below is additive (new files, reuse of
existing modules read-only), following the exact pattern the M4 dashboard already
uses to reuse Central Control.

---

## 1. Answers to the open questions (`REQUIREMENTS.md §9`)

All five are answered from the codebase — none need to be left open.

### Q1 · Segmentation model + GPU
**There is no SAM/SAM2.** The repo's segmentation is **local, classical OpenCV
GrabCut**, box-prompted, in [`src/segmentation_assist.py`](../../src/segmentation_assist.py)
(`refine_box_to_mask()` → `mask_to_polygon()` via `cv2.findContours` +
`approxPolyDP`). It uses **no model weights, no network, no GPU**, and never calls
YOLO or Groq (`SEG_BACKEND = "opencv-grabcut"`, [segmentation_assist.py:41](../../src/segmentation_assist.py#L41)).
`opencv-python-headless` is already a dependency ([requirements.txt:8](../../requirements.txt#L8)).

→ **Use GrabCut for the Screen-3 "chat-defined polygon".** The 2-click rectangle
becomes the ROI box; `refine_box_to_mask(frame, box_norm)` returns the polygon.
**No GPU needed anywhere** — YOLO runs CPU on Streamlit Cloud today. Do **not**
introduce SAM.

### Q2 · Weather source
**Already integrated, keyless.** [`src/weather.py`](../../src/weather.py) calls
**Open-Meteo** over stdlib `urllib` (no API key), with a deterministic offline
**mock** fallback flagged `is_live=False` ([weather.py:332](../../src/weather.py#L332)).
`fetch_weather(lat, lon)` → `Weather`; `assess_risk(weather, zones)` →
`RiskAdvisory`. → **Do not stub.** The Routine ("Watch") agent is `assess_risk`.

### Q3 · Where dashboards live + component libs
Dashboards are **not** a `pages/` multipage dir today — they are modules in
[`src/dashboards/`](../../src/dashboards/) dispatched by a sidebar `selectbox` in
[`app.py`](../../app.py#L37). Component libs already installed
([requirements.txt](../../requirements.txt)): **folium + streamlit-folium**
(maps, via `st_folium last_clicked`), **streamlit-image-coordinates** (pixel
clicks), **plotly** (charts, themed by `src/ui.apply_chart_theme`), **shapely**,
**opencv-python-headless**, **ultralytics**, **groq**. There is **no
drawable-canvas** — image interaction is click-to-place composited with PIL
(`central_control._composite_image`), not a canvas. → **Reuse those. Add no new UI
component libraries.**

### Q4 · Notification channels for the emergency agent
**On-screen logged actions only — never a real send.** The project's hard rule
(CLAUDE.md "Forbidden": *No emergency dispatch integration*) and
[`src/incident_agent.py`](../../src/incident_agent.py) ("only *drafts and
recommends*… never contacts anyone, never dispatches") define this. → The design's
**"DISPATCH" button must be reframed** as: open the Emergency agent's structured
notification drafts (owner / neighbour / farm worker / fire-dept summary from
`incident_agent.build_drafts`), each with a **"Log as sent"** action that appends
to an on-screen notification log. No Telegram/Slack/webhook. (Wiring a real
channel later is possible but is out of MVP scope and needs explicit sign-off.)

### Q5 · Event storage
**No DB exists.** The established pattern is a **session-state alert log**
(`cc_alert_log`) built from `alerts.create_alert_record` /
`incident_agent.create_incident_alert`, plus **CSV download**
([central_control.py:1957](../../src/dashboards/central_control.py#L1957)). → For
History that survives reruns/sessions, **append confirmed/false events to a new
`data/live_events.jsonl`** (new file, allowed) and read History from it, falling
back to the session log. This follows the existing "records + CSV" pattern without
inventing a DB.

---

## 2. Reuse map — existing code already covers most requirements

| Requirement | Reused code (read-only) |
|---|---|
| Camera config load (§2) | `_import_config_panel` loads `camera_mapping_config.json` → `cc_camera/cc_reference_points/cc_image_zones` ([central_control.py:523](../../src/dashboards/central_control.py#L523)); `mapping.default_camera_metadata`, `validate_camera_metadata` |
| Camera + FOV on map (§2) | `folium` markers as in `_build_and_consume_camera_map` ([central_control.py:658](../../src/dashboards/central_control.py#L658)); FOV cone = new folium `Polygon` from heading+FOV |
| Anchors / homography (§3) | `mapping.compute_homography` (DLT+SVD) + `estimate_map_position` ([mapping.py:313](../../src/mapping.py#L313)) — **the "project's existing utility"**; numbered markers via `_composite_image` + folium |
| Zones render + point-in-zone (§4) | `_composite_image` ([central_control.py:181](../../src/dashboards/central_control.py#L181)); `mapping.point_in_polygon`, `find_zone_for_detection` |
| Chat-defined polygon (§4) | 2 clicks via `_consume_image_click`; ROI box via `_bbox_norm_from_pixels`; `segmentation_assist.refine_box_to_mask`; name/priority via `zone_agent.parse_zone_description` |
| Video + per-frame YOLO (§5) | **New** `cv2.VideoCapture` frame sampler → feeds the **existing** sequence pipeline (`_build_sequence_frames`, `cc_seq*`, `_detect_frame_bytes`, `_load_detector_cached`); `inference.run_detection` |
| N-frame confirmation (§5) | `_process_current_sequence_frame` + `tracking.is_confirmed_with_tolerance` ([central_control.py:2347](../../src/dashboards/central_control.py#L2347)); `inference.select_confirmed_event_detection` |
| Alert → map projection (§5) | `_render_incident_map` + `_incident_map_point` / `_preview_map_point` ([central_control.py:2022](../../src/dashboards/central_control.py#L2022)); `inference.bbox_bottom_center_norm` → `estimate_map_position` |
| Emergency agent (§6) | `incident_agent.build_incident_context / recommend_actions / respond_to_operator / build_drafts / create_incident_alert` |
| Routine agent (§6) | `weather.fetch_weather` + `weather.assess_risk` |
| Playback controls (§5) | `_playback_controls`, `_advance_playback_if_needed`, `_apply_pending_seq_seek` ([central_control.py:2452](../../src/dashboards/central_control.py#L2452)) — Streamlit-only autoplay, one bounded `time.sleep` per rerun |
| Detector loading | `_load_detector_cached` (lazy `ultralytics`), checkpoints present: `models/yolo11s_dfire_best.pt`, `yolo11n_dfire_best.pt` |

**Net-new logic is small:** the light-1b layout/CSS, a video→frames sampler, the
dual-agent chat transcript, the DISPATCH-as-drafts reframe, the read-only
calibration view, and the History persistence file.

---

## 3. Surfacing the dashboard **without editing `app.py`**

`app.py`'s dispatch is a hardcoded `selectbox` — adding a mode there would *modify
existing code*, which is forbidden. Two additive options:

- **Recommended — Streamlit-native `pages/` entry.** Create `pages/` (does not
  exist yet) with one thin file, e.g. `pages/1_🔥_Live_Ops.py`. Streamlit
  auto-adds it to the nav with **zero edits to `app.py`**. That file sets its own
  page config, injects the light-1b theme, and calls the renderer. Trade-off:
  introduces Streamlit's page-nav UI (app.py becomes the "main" page); each page
  is its own script, so the new page injects its own theme instead of inheriting
  `app.py`'s dark video background — which is exactly what variant 1b wants.
- **Alternative — one-line dispatch in `app.py`** (add a mode + `elif`). Cleaner
  UX (stays in the existing sidebar shell) but **modifies existing code** →
  needs your explicit OK. Not chosen by default.

Either way the **logic lives in `src/dashboards/live_ops.py`** (same folder,
conventions, and lazy-import discipline as every other dashboard), so it is unit-
and smoke-testable like the rest.

---

## 4. New files (all additive)

```
pages/1_🔥_Live_Ops.py            thin entry: st.set_page_config + light-theme CSS + live_ops.render()
src/dashboards/live_ops.py        renderer: top-nav Setup/Live/History; reuses central_control as cc + src/*
src/live_ops_config.py            load prepared config + runtime knobs (§8) into cc_* state; pure (yaml/json), no ML
src/live_ops_agents.py            dual-agent ops-chat orchestration over incident_agent + weather; pure wrappers
config/live_demo_cam04.json       prepared camera_mapping_config.json (camera + >=4 reference_points + image_zones)
config/live_ops.yaml              runtime knobs: video path, DETECTION_INTERVAL_SEC, conf, routine interval, contacts
data/live_demo/                   prepared pre-recorded video (or an image-sequence folder) — Git-ignored if large
data/live_events.jsonl            appended event history (created on first confirmed/false event)
tests/test_live_ops_config.py     unit tests: config load/validate, knobs defaults — no ML, no network
tests/test_live_ops_agents.py     unit tests: agent-labelled routing, offline fallbacks — no ML, no network
tests/test_live_ops_dashboard.py  smoke: import-safe, render present, no ultralytics/torch/groq/cv2 at import
```

**Prepared setup file = one `camera_mapping_config.json`** in the exact shape the
existing Export tab already produces (`build_camera_mapping_config`,
[mapping.py:281](../../src/mapping.py#L281)). Author it once via the current
Central Control UI (place camera, add ≥4 reference points, draw the preloaded
zones, Export), drop it in `config/`. This satisfies "reuse the project's existing
config format" and makes §2/§3/§4 preload a call to the existing loader. Keep
`vertices_norm` in it (resolution-independent) so zone matching survives the
frame-resize step.

---

## 5. Session-state model

Reuse the `cc_*` keys verbatim (so setup done here is visible in Central Control /
M4 and vice-versa): `cc_camera`, `cc_reference_points`, `cc_image_zones`,
`cc_uploaded_image`, `cc_image_size`, `cc_seq*`, `cc_seq_det`,
`cc_incident_*`, `cc_alert_log`, `cc_risk_*`. Call `cc._init_state()` first (M4
does exactly this, [m4_dashboard.py:27](../../src/dashboards/m4_dashboard.py#L27)).

Add a few new keys namespaced `lo_*`:
`lo_tab` (Setup/Live/History), `lo_step` (1/2/3), `lo_config_loaded` (bool),
`lo_ops_chat` (`[{agent, role, content}]`), `lo_notify_log` (list),
`lo_routine_last` (UTC), `lo_hist_filter`, `lo_hist_from`, `lo_hist_to`,
`lo_active_alert` (dict|None for the banner).

---

## 6. Screen-by-screen build

Top bar + segmented nav **Setup · Live · History** (variant-1b top bar copy;
[PyroFinder.dc.html:364-378](PyroFinder.dc.html)). Use `st.segmented_control`
keyed on `lo_tab` (survives reruns — the exact reason M4 chose it over `st.tabs`,
[m4_dashboard.py:36](../../src/dashboards/m4_dashboard.py#L36)). Setup shows an
inner 3-step pill row keyed on `lo_step`.

### Setup / Step 1 — Place camera (preloaded, read-only)
- On entry, `live_ops_config.load_demo_config()` populates `cc_*` from
  `config/live_demo_cam04.json` (reuses `_import_config_panel`'s merge logic).
- Instruction banner (`#fbeee7` bg / `#8a3413` text). Map (~55%): folium camera
  `Marker` + an FOV **cone** = a folium `Polygon` of 3 points computed from camera
  `lat/lon` + `heading` + `FOV` (small new helper; `mapping.downwind_arrow_endpoint`
  shows the bearing-offset math to copy). Label chip "CAM-04 · placed ✓".
- Row: reference-frame preview (first sampled frame) + "Camera data" card (Camera
  ID, Name, Resolution, FOV, Status — key left / mono value right) + **Next →**.
- No manual placement — read-only visualization (README §2).

### Setup / Step 2 — Calibrate anchors (preloaded, read-only)
- Render `cc_reference_points` (≥4) as **numbered** markers on the frame via
  `_composite_image` (extend point labels to show index) and matching numbered
  folium markers on the map. Horizontal chip row "1 Water tower ✓ …".
- Compute + cache the homography with `mapping.compute_homography` (the existing
  DLT utility — **answers "cv2.findHomography or existing utility"**); it feeds
  Live's projection. Footer: ← Back / SAVE (green outline) / Next →. SAVE just
  re-exports the config (read-only demo).

### Setup / Step 3 — Zones + AI chat (preloaded + ONE chat-defined polygon)
- Camera workspace (~55%) with preloaded polygons drawn by `_composite_image`
  (priority colors already implemented: HIGH=ember, MED=amber). Label chips
  "Z1 · Dry brush · HIGH" / "Z2 · Access road · MED".
- "Zone assistant" chat panel. **Demo flow (§4):**
  1. Two clicks on the frame (`_consume_image_click`) → opposite corners; draw the
     rectangle immediately (`_composite_image(pending_vertices=...)`).
  2. Free-text prompt ("mark the dry brush inside the box") → `st.chat_input`.
  3. `segmentation_assist.refine_box_to_mask(frame, box_from_two_points)` → polygon
     (GrabCut, on click only). `zone_agent.parse_zone_description(text)` → name /
     priority / object_to_find.
  4. Show proposed polygon + quick-reply chips **Looks right ✓ / Edit outline /
     Change priority** (mirror `_render_seg_candidate` accept/fallback).
  5. On confirm, register into `cc_image_zones` (reuse `_commit_zone` shape:
     `vertices_px` + `vertices_norm` + priority + `object_to_find`).
- Footer: ← Back / **Finish — Go live ✓** (green) → sets `lo_tab="Live"`.

### Live — video + YOLO + alert + map + ops chat
- **Frames:** new `live_ops_config.extract_video_frames(path, DETECTION_INTERVAL_SEC)`
  using `cv2.VideoCapture` (sample 1 frame / N seconds) → `cc._build_sequence_frames`
  → `cc_seq`. (An image-sequence folder is the drop-in alternative; both land in the
  same state, so all downstream reuse is identical. `TESTING/sequance` — the
  sequence default — does not exist in-repo, so ship `data/live_demo/`.)
- **Per-frame detection + confirmation:** call `cc._process_current_sequence_frame(N)`
  — reuses cached per-frame YOLO (`cc_seq_det`), N-of-(N+1) one-miss tolerance, and
  builds the incident once via the guard `cc_incident_confirmed_idx`.
- **Layout (top→bottom, variant 1b):**
  - **Alert banner** (full width, solid `#d9481f`, pulsing dot) shown when an
    incident is confirmed OR `lo_active_alert` set: "SMOKE detected — {zone} · {time}",
    **DISPATCH** (see §7) + **Dismiss** (clears `lo_active_alert`, keeps the log).
  - **Video area (~50%)**: latest sampled frame with YOLO boxes
    (`_render_detection_overlay`) + "LIVE · CAM-04 · Ridge East" chip; box label
    "SMOKE 87%" comes from the overlay/`top_hazard_detection`.
  - **Bottom 50/50**: left = incident map (`_render_incident_map`, pulsing marker +
    downwind arrow already implemented); right = **Ops chat** (§8).
- Playback: `_playback_controls` + `_advance_playback_if_needed` +
  `_apply_pending_seq_seek` (autoplay to simulate "live"; a real video can also just
  advance by wall-clock). Confidence threshold + `DETECTION_INTERVAL_SEC` exposed in
  a settings expander (REQUIREMENTS §5/§8).

### History — event log + per-day chart
- Read events from `data/live_events.jsonl` (fallback to `cc_alert_log`). Date-range
  inputs (`st.date_input`) + filter chips All/Smoke/Fire/False alarms
  (`st.segmented_control`, active = solid red).
- "Detections per day" **stacked bar** (red confirmed / gray `#e4ded5` false) —
  plotly `go.Bar` stacked + `ui.apply_chart_theme`, or `st.bar_chart`. Filters drive
  both chart and list.
- Scrollable event list: status dot + bold desc + mono meta (`time · camera ·
  confidence`) + type badge (SMOKE/FIRE `#d9481f`, verified `#d97e1f`, FALSE
  `#b9b2a6`) + **Clip** button → shows the saved frame (persist the annotated PNG /
  frame index with each event).

---

## 7. DISPATCH & notifications (compliance-critical)

DISPATCH does **not** contact anyone (Q4). It opens the Emergency agent's
notification panel:
- `incident_agent.build_drafts(ctx)` → owner / neighbour / farm worker /
  fire-dept summary (worker messages carry zone+task, never coordinates — enforced
  in `incident_agent`).
- Each draft gets **"Log as sent (demo)"** → appends `{audience, text, ts}` to
  `lo_notify_log` and posts a Response-agent line into the ops chat. **Nothing
  leaves the app.** Copy must stay "operator relays this" (never "dispatched").
- Confirm alert / Mark false alarm reuse `incident_agent.create_incident_alert` +
  append to `cc_alert_log` **and** `data/live_events.jsonl`.

---

## 8. Ops chat — two agents, one window (§6)

`src/live_ops_agents.py`, one transcript `lo_ops_chat = [{agent, role, content}]`,
rendered with `st.chat_message`; the agent name/icon (`Watch` 🌤 / `Response` 🚨)
prefixes each bubble.

- **Routine "Watch"** — periodic risk status. `fetch_weather(cam lat/lon)` +
  `assess_risk(zones)` → post level + factors + downwind. Manual **Refresh** plus an
  elapsed-time check against `routine_interval` on each autoplay rerun (Streamlit has
  no safe background scheduler — same caveat as the Risk Advisory tab,
  [central_control.py:2693](../../src/dashboards/central_control.py#L2693)).
- **Emergency "Response"** — activates on confirmation / DISPATCH: posts
  `incident_agent.incident_narrative` + `recommend_actions`; drives drafts (§7).
- **Free-text** operator questions → `incident_agent.respond_to_operator(ctx, msg,
  history)` (Groq when `GROQ_API_KEY` in `.streamlit/secrets.toml`, else the
  deterministic responder — both already implemented and offline-safe).

---

## 9. Config block (§8) — `config/live_ops.yaml`

```yaml
camera_config: config/live_demo_cam04.json   # camera + reference_points(anchors) + image_zones
video_path: data/live_demo/ridge_east.mp4    # or a frames/ folder
detection_interval_sec: 2.0
confidence_threshold: 0.25
routine_report_interval_min: 30
contacts:                                    # on-screen demo log only (never sent)
  - {name: Owner, channel: log}
  - {name: Fire dept, channel: log}
```
Loaded by `live_ops_config.load()`; every value overridable in a Live "Settings"
expander. Secrets stay in `.streamlit/secrets.toml` (present) — never in this file.

---

## 10. Theme & fidelity

- **Fidelity target (README):** high for layout/hierarchy/copy, medium for styling.
- Inject **scoped light-1b CSS** from the page entry via `st.markdown(...,
  unsafe_allow_html=True)` (same mechanism as `ui.inject_pyrofinder_theme`, but the
  light palette): background `#faf9f7`, cards `#fff`, accent `#d9481f`, banner tint
  `#fbeee7`/`#8a3413`, success `#1e7a43`, radii 14px cards / 16px chips. Optional
  Public Sans + IBM Plex Mono via Google Fonts `@import` (fall back to theme
  defaults per README).
- **Caveats to flag to reviewers:** (a) global `.streamlit/config.toml` is dark;
  the light look is page-scoped CSS, so some Streamlit chrome may not fully
  re-tint — acceptable at medium styling fidelity. (b) `st_folium` renders its own
  tiles; the "map land/roads" placeholder palette isn't reproduced — real tiles
  replace placeholders per README "Assets". (c) `st.chat_message` bubbles differ
  from the mock's tailed bubbles — approximate with light CSS.

---

## 11. Testing (match existing conventions)

- **Smoke** (`test_live_ops_dashboard.py`): module imports; `render` callable; assert
  `ultralytics`/`torch`/`groq`/`cv2` **not** in `sys.modules` after import — mirror
  [test_dashboards_smoke.py:109](../../tests/test_dashboards_smoke.py#L109) and
  keep all heavy imports lazy/inside functions.
- **Unit** (`test_live_ops_config.py`): config load, missing-file + malformed
  handling, knob defaults, video/frames path selection — temp files only, no ML.
- **Unit** (`test_live_ops_agents.py`): Routine vs Emergency labelling, offline
  fallbacks (monkeypatch `weather`/`incident_agent`), transcript shape — no network.
- Reuse existing suites unchanged for the borrowed helpers (tracking, inference,
  incident_agent, weather, segmentation_assist, mapping).

---

## 12. Risks & decisions (with recommendations)

1. **Surfacing** (§3): default to a `pages/` entry (no `app.py` edit). If you'd
   rather keep the single sidebar shell, approve a one-line `app.py` dispatch edit.
   **Rec: `pages/`.**
2. **Video vs image-sequence** (§5): a real `.mp4` via `cv2.VideoCapture` best
   matches REQUIREMENTS; the existing image-sequence path is the lower-risk
   fallback and shares 100% of downstream code. **Rec: ship both loaders; demo with
   whichever asset you have.**
3. **DISPATCH semantics** (§7): must be logged-only. **Rec: confirmed — no real
   channel in MVP.**
4. **History persistence** (§7/Q5): new `data/live_events.jsonl`. **Rec: JSONL +
   session-log fallback + CSV download.**
5. **Prepared config authoring**: build `config/live_demo_cam04.json` via the
   current Central Control Export so shapes/keys match exactly and zone
   `vertices_norm` are present. **Rec: do this first — it unblocks Setup S1–S3.**
6. **Large assets**: keep the demo video out of Git (per repo policy) unless small;
   document the path. The two YOLO checkpoints are already committed and present.

---

## 13. Build sequence

1. Author `config/live_demo_cam04.json` (+ `config/live_ops.yaml`, `data/live_demo/`).
2. `src/live_ops_config.py` + tests (load/validate/knobs; video→frames sampler).
3. `pages/1_🔥_Live_Ops.py` + `src/dashboards/live_ops.py` skeleton (nav, light CSS,
   `cc._init_state()`), smoke test green.
4. **Setup** S1–S2 (preload + read-only map/anchors/homography).
5. **Setup** S3 (zones render + 2-click→GrabCut→confirm chat).
6. **Live** (frames → per-frame YOLO + N-frame confirm → banner + overlay + map).
7. `src/live_ops_agents.py` dual-agent ops chat + DISPATCH-as-drafts + notify log.
8. **History** (JSONL persistence, filters, per-day stacked chart, Clip).
9. Light-1b CSS polish + fidelity review against `PyroFinder.dc.html` variant 1b.
```

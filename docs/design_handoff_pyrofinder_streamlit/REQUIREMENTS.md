# PyroFinder — Functional Requirements (Streamlit dashboard)

Target: a NEW Streamlit dashboard added to the existing project (do not modify existing code — read-only reuse of existing modules, incl. the already-built YOLO model wrapper).

## 1. General
- All components must be REAL (no dummy widgets): real config loading, real video, real YOLO inference, real agent calls.
- All UI text in English.
- Design reference: `PyroFinder.dc.html`, variant **1b** (light theme). See README.md for layout & tokens.

## 2. Setup / Step 1 — Camera (preloaded)
- On load, read a **prepared camera config file** (path configurable; suggest `config/camera_cam04.yaml` or reuse the project's existing config format). It contains: camera id, name, snapshot/reference image path, stream/video source, resolution, FOV, heading, geo position (lat/lon), elevation.
- Display: reference image, camera-data card, and the camera position + FOV cone on the map.
- No manual placement in the demo — the screen is a read-only visualization of the config.

## 3. Setup / Step 2 — Anchor calibration (preloaded)
- Read the **prepared reference-points file**: ≥4 pairs of (image px,py) ↔ (map lat,lon), each with a human-readable name.
- Display the numbered anchors on both the camera image and the map, plus the named chip list.
- The homography / image→geo mapping derived from these pairs is used later to project detections onto the map (Live tab). Compute it with OpenCV (`cv2.findHomography` or the project's existing utility if one exists).

## 4. Setup / Step 3 — Zones (preloaded + ONE chat-defined polygon)
- Load existing polygons from a prepared zones file (id, name, priority, pixel-coordinates polygon) and render them over the camera image.
- **Demo flow — add one polygon via chat**:
  1. User clicks **2 points** on the camera image → these are opposite corners of a rectangle ROI (render the rectangle immediately). Suggested widget: `streamlit-image-coordinates` or `streamlit-drawable-canvas`.
  2. User types in the chat what to detect/segment inside that rectangle (e.g. "mark the dry brush inside the box").
  3. The zone-assistant agent runs **segmentation restricted to the ROI** (use the project's existing segmentation capability; otherwise SAM/SAM2 with the ROI box as prompt — confirm model choice with the team).
  4. The resulting mask is converted to a polygon (e.g. `cv2.findContours` + `approxPolyDP`), rendered as a proposed zone, and the agent asks for confirmation, name, and priority (quick replies: Looks right ✓ / Edit outline / Change priority).
  5. On confirmation the polygon is registered in session state (and optionally appended to the zones file).

## 5. Live tab — video + YOLO
- Video source: a **pre-recorded file from a project folder** (path configurable).
- Frame sampling: run the **existing YOLO model** on frames extracted every `N` seconds (configurable constant, e.g. `DETECTION_INTERVAL_SEC`; expose in a sidebar/settings expander).
- Display: the playing video (or the latest sampled frame) with YOLO bounding boxes + class + confidence drawn on detections.
- Alert logic: when a detection ≥ confidence threshold intersects a registered zone → raise an active alert: red banner (zone name, time), bounding box highlight, and a **map marker** at the geo position obtained by projecting the detection's bottom-center point through the calibration homography (§3).
- Banner actions: **Dismiss** (clear alert) and **DISPATCH** (hand off to the emergency agent, §6).

## 6. Ops chat — two agents in one window
One chat panel on the Live tab, shared by two agents (label each message with the agent's name/icon):

- **Routine agent** ("Watch"): posts **periodic** status reports (configurable interval) assessing fire risk from **meteorological data** (existing weather source/API in the project, or a stub interface to be wired) combined with the camera location — e.g. temperature, humidity, wind speed/direction, resulting risk level.
- **Emergency agent** ("Response"): activated on a real alert (or DISPATCH). It reports the event (what, where — geo from the projection, when, confidence), **recommends actions**, and can **communicate methodically with rescue/security forces and team members** — implement as structured, logged notification actions (e.g. send message / call webhook / notify contact list), each proposed to the operator for confirmation before sending in the demo.
- Both agents should use the project's existing agent/LLM infrastructure. Chat history persists in session state; user can also ask free-text questions about the scene.

## 7. History tab
- Persist every detection/alert event (timestamp, type, camera, zone, confidence, clip/frame reference, resolved-as) to the project's existing storage (file/DB — follow existing patterns).
- UI: date-range filter, type filter chips (All / Smoke / Fire / False alarms), detections-per-day stacked bar chart (confirmed vs false), scrollable event list with a "Clip" action that shows the saved frame/clip.

## 8. Configuration summary (all paths/values in one config block)
- camera config file path; anchors file path; zones file path
- video file path; `DETECTION_INTERVAL_SEC`; confidence threshold
- routine-agent report interval; weather data source
- contact list / notification channels for the emergency agent

## 9. Open questions for planning (resolve with the team in Claude Code)
1. Which segmentation model is available in the repo (SAM/SAM2/other), and is a GPU available for the demo?
2. Existing weather data source — API already integrated, or stub it?
3. Where do existing dashboards live (multipage `pages/` dir?) and what component libs are already in use (folium? plotly? drawable-canvas?)
4. Notification channels for the emergency agent in the demo: on-screen log only, or a real channel (Telegram/Slack/webhook)?
5. Event storage: existing DB/file format to append to?

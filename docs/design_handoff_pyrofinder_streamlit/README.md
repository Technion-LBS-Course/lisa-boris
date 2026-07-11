# Handoff: PyroFinder — Fire & Smoke Detection Dashboard (Streamlit)

## Overview
PyroFinder is a real-time fire and smoke detection application that works with existing cameras. This handoff covers a **new Streamlit dashboard** (added alongside the project's existing Streamlit dashboards) implementing a 3-part flow:

1. **Setup** — camera placement on a map, anchor-point calibration (camera↔map mapping), and detection-zone (polygon) management with an AI chat assistant.
2. **Live** — monitoring of a (pre-recorded) video with YOLO-based smoke/fire detection on sampled frames, an alert banner, a map projection of detections, and a dual-agent ops chat.
3. **History** — date-filtered event log with a detections-per-day chart.

## About the Design Files
The file `PyroFinder.dc.html` in this bundle is a **design reference created in HTML** — a prototype showing intended look and behavior, NOT production code to copy. Your task is to **recreate this design inside the existing Streamlit project**, using its established patterns, components, and libraries (e.g. `streamlit`, `st.tabs`, `streamlit-drawable-canvas` / `st-folium` / plotly, or whatever the codebase already uses). Follow the codebase's existing conventions first; use this document for layout, hierarchy, copy, and visual direction.

**The selected design variant is `1b` — the light tablet-style version** (the second frame in the HTML file). Ignore variant `1a` (the dark one).

## Fidelity
**High-fidelity for layout/hierarchy/copy, medium-fidelity for styling.** Streamlit cannot reproduce pixel-perfect custom HTML; recreate the structure, flow, and information hierarchy faithfully, and approximate the visual style (light theme, flame-red accent `#d9481f`) via Streamlit theming/config and light CSS where the project already permits it.

## Screens / Views

The app has a top-level navigation with 3 tabs: **Setup · Live · History**. Setup contains an internal 3-step flow: **1. Place → 2. Calibrate → 3. Zones**.

### Screen 1 — Setup / Step 1: Place camera
- **Purpose**: show the camera's location, direction and field-of-view cone on a map, alongside its live/preview frame and metadata.
- **Layout**: instruction banner on top (light red-tinted `#fbeee7`, dark-red text `#8a3413`); large map area (~55% height); below it a row: camera frame preview (left, flexible) + "Camera data" card (right, ~250px).
- **Camera data card fields**: Camera ID, Name, Resolution, FOV, Status — key on the left (muted), monospace value on the right, thin divider rows, primary button "Next →" at the bottom.
- **Map**: camera marker (red dot, white ring) + orange-tinted FOV cone polygon + label chip "CAM-04 · placed ✓".
- **IMPORTANT (functional)**: in the real app this screen is **preloaded from a prepared config file** — no manual placement needed for the demo. See REQUIREMENTS.md §2.

### Screen 2 — Setup / Step 2: Calibrate anchors
- **Purpose**: show the ≥4 anchor-point pairs that map camera-image pixels to map/geo coordinates.
- **Layout**: instruction banner; camera frame with numbered anchor markers (red circles, white numbers); map below with the matching numbered markers; a horizontal chip row listing anchors by name (e.g. "1 Water tower ✓", "2 Road junction ✓"…); footer buttons: "← Back", "SAVE" (green outline `#1e7a43`), "Next →" (red primary).
- **IMPORTANT (functional)**: anchors are also **preloaded from the prepared calibration file**. The UI displays them read-only for the demo. See REQUIREMENTS.md §3.

### Screen 3 — Setup / Step 3: Zones + AI chat
- **Purpose**: display registered detection polygons on the camera frame, and let the user add ONE new polygon during the demo via chat-driven segmentation.
- **Layout**: camera frame workspace on top (~55%) with polygons overlaid (dashed outline, tinted fill, label chip "Z1 · Dry brush · HIGH" red / "Z2 · Access road · MED" green); chat panel below ("Zone assistant" header, message bubbles, quick-reply chips "Looks right ✓ / Edit outline / Change priority", input row); footer "← Back" / "Finish — Go live ✓" (green).
- **Chat bubbles**: assistant left-aligned (bg `#f0ede8`), user right-aligned (bg `#fbeee7`, text `#8a3413`), radius 10px with a 3px "tail" corner.
- **IMPORTANT (functional)**: the demo flow — user clicks 2 points defining a rectangle ROI on the frame, then types what to detect; the agent runs segmentation inside that rectangle and registers the polygon. See REQUIREMENTS.md §4.

### Screen 4 — Live
- **Purpose**: monitor the video with YOLO detections, alerting, map projection, and the dual-agent ops chat.
- **Layout** (top to bottom): full-width alert banner when active (solid red `#d9481f`, white text, pulsing dot, "DISPATCH" white button + "Dismiss" ghost button); large video area (~50%) with "LIVE · CAM-04 · Ridge East" chip and, on detection, a red bounding box + label "SMOKE 87%"; bottom row split 50/50 — map (detection projected as pulsing red dot inside the FOV cone) + "Ops chat" panel.
- **IMPORTANT (functional)**: video is a **pre-recorded file from a folder**; YOLO inference runs on frames sampled at a configurable interval. Chat is fed by two agents. See REQUIREMENTS.md §5–6.

### Screen 5 — History
- **Purpose**: browse past events.
- **Layout**: title + date-range pickers (from → to); filter chips row (All / Smoke / Fire / False alarms — active chip solid red, others white outline); "Detections per day" stacked bar chart (red = confirmed, gray `#e4ded5` = false alarm); scrollable event list — each row: colored status dot, description (bold) + meta line (timestamp · camera · confidence, monospace, muted), type badge, "Clip" button.
- Event type colors: SMOKE/FIRE alert `#d9481f`, verified/warn `#d97e1f`, FALSE `#b9b2a6`.

## Interactions & Behavior
- Tab navigation Setup / Live / History (segmented control in the top bar; active tab white pill with red text).
- Setup steps navigable via pill buttons and Next/Back.
- Zone chat: 2-click rectangle ROI → free-text prompt → agent proposes polygon → quick-reply confirm/edit/priority → polygon registered and rendered.
- Live: alert banner + bounding box appear when a detection crosses the confidence threshold; map dot pulses; Dismiss hides the banner; Dispatch triggers the emergency agent flow.
- History: date range + type filters filter both chart and list.

## State Management (Streamlit)
Use `st.session_state` for: current tab/step, loaded camera config, anchors, zones list (incl. the chat-added polygon), chat transcripts (setup assistant + ops chat), ROI click points, active alert object, detection log, history filters. Long-running work (video frame loop, YOLO inference, agents) should follow the project's existing patterns (e.g. `st.fragment` with `run_every`, threads + queue, or the existing pipeline runner).

## Design Tokens (variant 1b)
- **Colors**: background `#faf9f7`; surface/cards `#ffffff`; borders `rgba(0,0,0,.08)`; text `#1c1b18`; muted text `rgba(0,0,0,.45)`; accent (primary) `#d9481f`; accent hover `#c23e18`; accent tint bg `#fbeee7`; accent dark text `#8a3413`; success green `#1e7a43`; success tint `#eaf5ee`; warn orange `#d97e1f`; neutral gray `#b9b2a6`; chart gray `#e4ded5`; map land `#e7ebe2`; map roads `#f7f4ee`.
- **Typography**: UI — Public Sans (400/500/600/700); data/labels/timestamps — IBM Plex Mono (400–600). Body 12–13px, section titles 13–16px bold, chips/labels 10–11px.
- **Radius**: cards 14px; buttons 8–10px; chips/pills 16px; small chips 5–6px.
- **Spacing**: page padding 18px; card gaps 14px; card inner padding 14–16px.
- **Buttons**: primary solid red, white text, 46px tall; secondary white with 1px border; success solid/outline green.

## Assets
- No external images. Map areas and camera frames in the prototype are **placeholders** — in the real app they are replaced by real map tiles (or a static map image) and real video frames.
- Fonts: Google Fonts — Public Sans, IBM Plex Mono (optional in Streamlit; fall back to theme defaults if the project doesn't inject fonts).

## Files
- `PyroFinder.dc.html` + `support.js` — the interactive design reference (open in a browser; use variant **1b**, the light 834px-wide frame; click through tabs/steps).
- `REQUIREMENTS.md` — functional requirements & integration spec (READ THIS with the README).

## Constraint
**Do not modify or replace any existing code in the repository.** Add the new dashboard as a new page/module alongside the existing ones, importing existing utilities read-only.

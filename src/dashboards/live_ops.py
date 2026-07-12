"""Live Ops Dashboard — Setup · Live · History (design variant 1b, light).

Added alongside the existing dashboards; reuses the Central Control implementation
(``central_control`` helpers + shared ``cc_*`` session state) and the pure ``src/*``
modules read-only. Three views:

* **Setup** — preloaded from the prepared camera mapping config: place the camera +
  approximate field of view on the map; calibrate anchors (add / rename / delete /
  pick on image + map); mark detection zones with the framed **zone assistant** —
  box an area with two clicks → segment it → name + priority (all in the chat), with
  a manual "draw the contour" fallback if the segmentation is not good enough.
* **Live** — autoplays a demo sequence, runs YOLO11s per frame (per-class
  confidence: smoke and fire have separate thresholds), and on an
  N-frame-confirmed detection freezes on that frame (keeping the box) until the
  operator resolves it **inside the ops chat** (confirm / false alarm). Any
  message drafting or contact suggestion happens conversationally in the chat,
  grounded in the incident + operational context — no separate dispatch panel.
* **History** — date/type-filtered event log persisted to ``data/live_events.jsonl``.

No module-level ML imports: ``ultralytics`` loads lazily inside ``src.inference``;
``cv2`` only inside ``src.live_ops_config`` / ``src.segmentation_assist``;
``groq`` only lazily inside ``src.llm``.
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.dashboards import central_control as cc, map_tiles
from src import (
    incident_agent,
    live_ops_agents as agents,
    live_ops_config as lo_cfg,
    mapping,
    segmentation_assist as seg,
    tracking,
    weather,
    zone_agent,
)
from src.agent_schemas import PRIORITY_LABELS, priority_label_to_int

_ACCENT = "#d9481f"
_PRIORITY_DOT = {"high": "#d9481f", "medium": "#d97e1f", "low": "#1e7a43"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _settings() -> dict:
    if "lo_settings" not in st.session_state:
        st.session_state.lo_settings = lo_cfg.load_settings()
    return st.session_state.lo_settings


# ── Session state + one-time config load ──────────────────────────────────────


def _init_lo_state() -> None:
    defaults = {
        "lo_tab": "Setup",
        "lo_step": 1,
        "lo_config_loaded": False,
        "lo_operational_context": None,     # optional structured operational context (JSON)
        "lo_operational_context_md": None,  # optional human-readable operational context (MD)
        "lo_ops_chat": [],          # Live ops chat [{agent, role, content}]
        "lo_zone_chat": [],         # Setup zone-assistant chat [{role, content}]
        "lo_zone_pts": [],          # clicked points [[x, y], ...] (pixels)
        "lo_zone_mode": "box",      # "box" (2-point → segment) | "draw" (manual contour)
        "lo_zone_poly": None,       # proposed polygon (normalized [{x, y}]) or None
        "lo_zone_name": "",         # zone name — set only via the chat (Groq parse)
        "lo_zone_priority": "medium",  # zone priority — set only via the chat (Groq parse)
        "lo_selected_zone": None,   # zone_id of the zone card selected for edit/delete
        "lo_zone_ref_prompt": None,     # zone_id awaiting a Yes/No reference-point decision
        "lo_zone_ref_picking": None,    # zone_id whose reference point is being picked
        "lo_zone_ref_pending_pt": None,  # clicked reference-point pixel awaiting Save
        "lo_selected_anchor": None,  # point_id of the anchor card selected for edit/delete
        "lo_routine_last": None,
        "lo_seq_det": {},
        "lo_seq_det_conf": None,
        "lo_seq_frames_hash": "",     # content fingerprint of the loaded source frames
        "lo_disk_cache": None,        # pre-computed per-frame detections (default mode only)
        "lo_disk_cache_fp": None,     # fingerprint the loaded disk cache was prepared for
        "lo_disk_model": None,        # detector name the disk cache was built with
        "lo_confirmed_idx": None,
        "lo_active_alert": None,
        "lo_suppress_until_clear": False,
        "lo_autoplay_armed": False,
        "lo_seq_loaded_from": None,
        "lo_show_clip": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _load_config_once() -> None:
    if st.session_state.lo_config_loaded:
        return
    settings = _settings()
    try:
        cfg = lo_cfg.load_camera_config(settings["camera_config"])
    except Exception as exc:  # noqa: BLE001 — surface, never crash the page
        st.warning(f"Could not load camera config ({settings['camera_config']}): {exc}")
        cfg = {"camera": {}, "reference_points": [], "image_zones": []}

    base = mapping.default_camera_metadata()
    st.session_state.cc_camera = {**base, **(cfg["camera"] or {})}
    st.session_state.cc_reference_points = cfg["reference_points"]
    st.session_state.cc_image_zones = cfg["image_zones"]

    ref = lo_cfg.load_reference_frame(settings["reference_frame"])
    if ref:
        st.session_state.cc_uploaded_image = ref
        try:
            from PIL import Image
            st.session_state.cc_image_size = Image.open(io.BytesIO(ref)).size
        except Exception:
            pass

    # Optional operational context (landmarks / receptors / contact policy) used only
    # for Live incident reasoning + first-message wording. Missing files degrade to None.
    oc = lo_cfg.load_operational_context(settings)
    st.session_state.lo_operational_context = oc.get("json")
    st.session_state.lo_operational_context_md = oc.get("md")

    st.session_state.lo_config_loaded = True


# ── Groq: make the key available regardless of launch directory ───────────────


def _ensure_groq_env() -> None:
    """Load GROQ_API_KEY into the environment so the Groq client works no matter
    which directory the app was launched from (``st.secrets`` only auto-loads from
    the entrypoint dir / home). The key is never logged, printed, or committed.
    """
    import os

    if os.environ.get("GROQ_API_KEY"):
        return
    try:
        from src import llm
        if llm.api_key_present():  # st.secrets already exposes it
            return
    except Exception:
        pass
    try:
        path = lo_cfg.REPO_ROOT / ".streamlit" / "secrets.toml"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("GROQ_API_KEY") and "=" in stripped:
                    value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        os.environ["GROQ_API_KEY"] = value
                    break
    except Exception:
        pass


# ── Shared UI bits ─────────────────────────────────────────────────────────────


def _top_bar() -> None:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:2px'>"
        f"<div style='width:28px;height:28px;border-radius:8px;background:{_ACCENT};color:#fff;"
        "display:grid;place-items:center;font-weight:700;font-size:15px'>P</div>"
        "<span style='font-weight:700;font-size:20px;letter-spacing:-.01em'>PyroFinder</span></div>",
        unsafe_allow_html=True,
    )


def _step_title(title: str, body: str = "") -> None:
    extra = f" {body}" if body else ""
    st.markdown(
        f"<div style='padding:12px 16px;background:#fbeee7;border-radius:12px;color:#8a3413;"
        f"margin-bottom:10px'><strong>{title}</strong>{extra}</div>",
        unsafe_allow_html=True,
    )


def _status_pill(label: str, name: str, tone: str = "live") -> None:
    dot = "#d9481f" if tone == "alert" else "#1e7a43"
    st.markdown(
        f"<span style='display:inline-block;padding:4px 11px;border-radius:16px;background:#fff;"
        f"border:1px solid rgba(0,0,0,.1);font:600 11px \"IBM Plex Mono\",monospace'>"
        f"<span style='color:{dot}'>●</span> {label} · {name}</span>",
        unsafe_allow_html=True,
    )


# ── Geo maps (camera + approximate FOV cone + reference points + incident) ─────


def _base_map(center, zoom=13):
    import folium

    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    map_tiles.add_satellite_basemap(m)
    return m


def _add_camera_and_fov(m) -> None:
    import folium

    cam = st.session_state.cc_camera
    lat, lon = cam.get("latitude"), cam.get("longitude")
    if lat is None or lon is None:
        return
    cone = lo_cfg.approx_fov_cone(cam, st.session_state.cc_reference_points)
    if cone:
        folium.Polygon(cone, color=_ACCENT, weight=1, fill=True, fill_color=_ACCENT,
                       fill_opacity=0.12, tooltip="Approximate field of view").add_to(m)
    label = cam.get("camera_name") or cam.get("camera_id") or "Camera"
    folium.Marker([float(lat), float(lon)],
                  icon=folium.Icon(color="red", icon="camera", prefix="fa"),
                  tooltip=f"{label} · placed").add_to(m)


def _add_reference_points(m) -> None:
    import folium

    for p in st.session_state.cc_reference_points:
        if not p.get("enabled", True) or p.get("map_lat") is None:
            continue
        folium.CircleMarker([float(p["map_lat"]), float(p["map_lon"])], radius=6,
                            color="#8a3413", weight=2, fill=True, fill_color="#fbeee7",
                            fill_opacity=1.0, tooltip=f"Anchor {p.get('point_name', '')}").add_to(m)


def _map_center(fallback=(38.84, -121.31)):
    cam = st.session_state.cc_camera
    if cam.get("latitude") is not None and cam.get("longitude") is not None:
        return [float(cam["latitude"]), float(cam["longitude"])]
    return list(fallback)


def _render_setup_map(key: str, height: int = 380) -> None:
    try:
        import folium  # noqa: F401
        from streamlit_folium import st_folium
    except ImportError:
        st.info("Map requires `folium` and `streamlit-folium`.")
        return
    m = _base_map(_map_center(), 13)
    _add_camera_and_fov(m)
    _add_reference_points(m)
    map_tiles.add_layer_control_once(m)
    st_folium(m, key=key, height=height, use_container_width=True, returned_objects=[])


def _render_live_map(ctx, preview_point=None, height: int = 360) -> None:
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.info("Map requires `folium` and `streamlit-folium`.")
        return

    incident = None
    if ctx is not None and ctx.approximate_lat is not None and ctx.approximate_lon is not None:
        incident = (ctx.approximate_lat, ctx.approximate_lon)
    elif preview_point is not None:
        incident = preview_point

    center = list(incident) if incident else _map_center()
    m = _base_map(center, 14)
    _add_camera_and_fov(m)
    _add_reference_points(m)

    if incident is not None and ctx is not None and ctx.approximate_lat is not None:
        folium.Marker(list(incident), icon=folium.Icon(color="orange", icon="fire", prefix="fa"),
                      tooltip="Approximate incident point").add_to(m)
        if ctx.wind_direction_deg is not None:
            end = mapping.downwind_arrow_endpoint(incident[0], incident[1], ctx.wind_direction_deg)
            folium.PolyLine([list(incident), list(end)], color="black", weight=3, opacity=0.9,
                            tooltip=f"Downwind risk: {ctx.downwind_risk_direction}").add_to(m)
    elif incident is not None:
        folium.Marker(list(incident),
                      icon=folium.Icon(color="beige", icon="exclamation-triangle", prefix="fa"),
                      tooltip="Detection observed — unconfirmed").add_to(m)

    map_tiles.add_layer_control_once(m)
    st_folium(m, key="lo_live_map", height=height, use_container_width=True, returned_objects=[])


# ── SETUP · Step 1 (place camera) ─────────────────────────────────────────────


def _camera_data_card() -> None:
    cam = st.session_state.cc_camera
    w, h = st.session_state.cc_image_size
    lat, lon = cam.get("latitude"), cam.get("longitude")
    rows = [
        ("Name", str(cam.get("camera_name", "—"))),
        ("Resolution", f"{w}×{h}"),
        ("Location", f"{lat:.4f}, {lon:.4f}" if lat is not None else "—"),
        ("Status", "Configured ✓"),
    ]
    body = (
        "<div style='background:#fff;border:1px solid rgba(0,0,0,.08);border-radius:14px;"
        "padding:6px 14px 10px'>"
        "<div style='font-weight:700;padding:9px 0 4px'>Camera data</div>"
    )
    for key, value in rows:
        body += (
            "<div style='display:flex;justify-content:space-between;gap:12px;padding:7px 0;"
            "border-bottom:1px solid rgba(0,0,0,.05)'>"
            f"<span style='color:rgba(0,0,0,.45);font-size:12px'>{key}</span>"
            f"<span style='font-family:\"IBM Plex Mono\",monospace;font-size:12px;text-align:right'>{value}</span>"
            "</div>"
        )
    body += "</div>"
    st.markdown(body, unsafe_allow_html=True)


def _setup_step1() -> None:
    _step_title("Step 1 · Place the camera on the map.")
    _render_setup_map("lo_setup_map1", height=380)
    col_img, col_card = st.columns([2, 1])
    with col_img:
        if st.session_state.cc_uploaded_image:
            st.image(st.session_state.cc_uploaded_image, use_container_width=True)
        else:
            st.info("Reference frame not found — check `reference_frame` in config/live_ops.yaml.")
    with col_card:
        _camera_data_card()
        if st.button("Next →", type="primary", use_container_width=True, key="lo_s1_next"):
            st.session_state.lo_pending_step = 2
            st.rerun()


# ── SETUP · Step 2 (calibrate anchors: add / rename / delete / pick) ──────────


def _anchor_cards() -> None:
    """Existing anchors as clickable cards — clicking a card selects it (highlighted)
    for editing/deleting below. No dropdown involved.
    """
    refs = st.session_state.cc_reference_points
    if not refs:
        st.caption("No anchors yet — pick an image point and the matching map point, name it, and add it.")
        return
    selected = st.session_state.get("lo_selected_anchor")
    if selected is not None and not any(p["point_id"] == selected for p in refs):
        st.session_state.lo_selected_anchor = None
        selected = None
    per_row = 4
    for start in range(0, len(refs), per_row):
        row = refs[start:start + per_row]
        cols = st.columns(len(row))
        for i, (col, p) in enumerate(zip(cols, row), start=start + 1):
            with col:
                is_sel = p["point_id"] == selected
                label = f"{i} · {p.get('point_name') or p['point_id']} ✓"
                if st.button(label, key=f"lo_anchor_card_{p['point_id']}",
                             type="primary" if is_sel else "secondary",
                             use_container_width=True):
                    st.session_state.lo_selected_anchor = None if is_sel else p["point_id"]
                    st.rerun()


def _anchor_edit_delete() -> None:
    refs = st.session_state.cc_reference_points
    target = next((p for p in refs if p["point_id"] == st.session_state.get("lo_selected_anchor")), None)
    if target is None:
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        new_name = st.text_input("Anchor name", value=target.get("point_name", ""), key="lo_ref_rename")
    with c2:
        st.write("")
        if st.button("Save name", key="lo_ref_savename", use_container_width=True):
            target["point_name"] = new_name.strip()
            st.rerun()
    with c3:
        st.write("")
        if st.button("Delete", key="lo_ref_del", use_container_width=True):
            st.session_state.cc_reference_points = [p for p in refs if p["point_id"] != target["point_id"]]
            st.session_state.lo_selected_anchor = None
            st.rerun()


def _setup_step2() -> None:
    _step_title("Step 2 · Calibrate anchors")
    # The image fills its column at its natural 16:9 ratio (rounded, responsive);
    # size the map to that same height so the two equal-width columns line up.
    col_img, col_map = st.columns(2)
    with col_img:
        comp = cc._composite_image(
            st.session_state.cc_uploaded_image,
            reference_points=st.session_state.cc_reference_points,
            pending_img_pt=st.session_state.get("cc_pending_ref_img"),
        )
        click = cc._consume_image_click(comp, key="lo_ref_img")
        if click:
            st.session_state.cc_pending_ref_img = click
            st.rerun()
        pend = st.session_state.get("cc_pending_ref_img")
        st.caption(f"Image point: ({pend[0]:.0f}, {pend[1]:.0f})" if pend
                   else "Click the matching points on the map and on the image")
    with col_map:
        latlon = cc._build_and_consume_ref_map(height=350)
        if latlon:
            st.session_state.cc_pending_ref_map = latlon
            st.rerun()
        pm = st.session_state.get("cc_pending_ref_map")
        if pm:
            st.caption(f"Map point: {pm[0]:.5f}, {pm[1]:.5f}")

    a1, a2 = st.columns([3, 1])
    with a1:
        st.text_input("New anchor name", key="lo_ref_name", placeholder="e.g. Water tower",
                      label_visibility="collapsed")
    with a2:
        if st.button("Add anchor", type="primary", use_container_width=True, key="lo_ref_add"):
            cc._add_reference_point(st.session_state.get("lo_ref_name", ""), "")

    _anchor_cards()
    _anchor_edit_delete()

    b1, _, b2 = st.columns([1, 4, 1])
    with b1:
        if st.button("← Back", key="lo_s2_back", use_container_width=True):
            st.session_state.lo_pending_step = 1
            st.rerun()
    with b2:
        if st.button("Next →", type="primary", key="lo_s2_next", use_container_width=True):
            st.session_state.lo_pending_step = 3
            st.rerun()


# ── SETUP · Step 3 (zones: box → segment → name/priority; manual-draw fallback) ─


def _zone_say(text: str, role: str = "assistant") -> None:
    st.session_state.lo_zone_chat.append({"role": role, "content": text})


def _zone_reset() -> None:
    st.session_state.lo_zone_pts = []
    st.session_state.lo_zone_poly = None
    st.session_state.lo_zone_mode = "box"


def _zone_overlay_vertices():
    """Pixel vertices to preview on the frame for the current zone-in-progress."""
    pts = st.session_state.lo_zone_pts
    poly = st.session_state.lo_zone_poly
    w, h = st.session_state.cc_image_size
    if poly:
        return [list(v) for v in seg.polygon_to_pixel_vertices(poly, w, h)]
    if st.session_state.lo_zone_mode == "box" and len(pts) == 2:
        (x0, y0), (x1, y1) = pts[0], pts[1]
        return [[min(x0, x1), min(y0, y1)], [max(x0, x1), min(y0, y1)],
                [max(x0, x1), max(y0, y1)], [min(x0, x1), max(y0, y1)]]
    return [list(p) for p in pts] if pts else None


def _zone_on_click(pt) -> None:
    pts = st.session_state.lo_zone_pts
    if st.session_state.lo_zone_mode == "box":
        st.session_state.lo_zone_pts = [list(pt)] if len(pts) >= 2 else pts + [list(pt)]
        st.session_state.lo_zone_poly = None  # box changed → invalidate any segmentation
    else:  # draw the contour, point by point
        st.session_state.lo_zone_pts = pts + [list(pt)]
        st.session_state.lo_zone_poly = None


def _zone_segment() -> None:
    pts = st.session_state.lo_zone_pts
    if len(pts) < 2:
        _zone_say("Click two opposite corners on the image first, then I'll segment the box.")
        return
    w, h = st.session_state.cc_image_size
    (x0, y0), (x1, y1) = pts[0], pts[1]
    box = {"x_min": min(x0, x1) / w, "y_min": min(y0, y1) / h,
           "x_max": max(x0, x1) / w, "y_max": max(y0, y1) / h}
    try:
        box = seg.validate_roi_box(box)
    except ValueError:
        _zone_say("That box is too small — click two corners further apart.")
        return
    with st.spinner("Segmenting inside the box…"):
        result = seg.refine_box_to_mask(st.session_state.cc_uploaded_image, box)
    if result.get("ok") and result.get("polygon"):
        st.session_state.lo_zone_poly = result["polygon"]
        _zone_say("Here's the outline from the box. Keep it, or draw the contour yourself.")
    else:
        st.session_state.lo_zone_poly = seg.polygon_from_box_fallback(box)
        _zone_say("Couldn't isolate a clean shape, so I kept the box outline. "
                  "Keep it, or draw the contour yourself.")


def _zone_save() -> None:
    poly = st.session_state.lo_zone_poly
    w, h = st.session_state.cc_image_size
    if poly is None and st.session_state.lo_zone_mode == "draw" and len(st.session_state.lo_zone_pts) >= 3:
        poly = [{"x": p[0] / w, "y": p[1] / h} for p in st.session_state.lo_zone_pts]
    if poly is None:
        _zone_say("Nothing to save yet — segment the box or draw at least 3 contour points.")
        return
    name = (st.session_state.get("lo_zone_name") or "").strip()
    if not name:
        _zone_say("Tell me the zone name and priority first — type it below, "
                  "e.g. 'dry brush on the left, high priority'.")
        return
    label = st.session_state.get("lo_zone_priority", "medium")
    verts = [list(v) for v in seg.polygon_to_pixel_vertices(poly, w, h)]
    zone_id = str(uuid.uuid4())[:8]
    st.session_state.cc_image_zones.append({
        "zone_id": zone_id,
        "zone_name": name,
        "zone_type": "custom",
        "alert_label": name,
        "priority": priority_label_to_int(label),
        "priority_label": label,
        "object_to_find": name,
        "requires_user_confirmation": True,
        "vertices_px": verts,
        "vertices_norm": mapping.normalize_polygon_vertices(verts, w, h),
        "zone_ref_point_px": None,
        "zone_ref_point_norm": None,
        "enabled": True,
        "notes": "added via zone assistant",
        "polygon_status": zone_agent.POLYGON_DRAWN,
    })
    _zone_reset()
    st.session_state.lo_zone_name = ""
    st.session_state.lo_zone_priority = "medium"
    # Ask whether to attach an optional per-zone reference point (the map-reporting
    # point). Handled by _zone_ref_point_row() until the operator answers.
    st.session_state.lo_zone_ref_prompt = zone_id
    _zone_say(f"Saved zone '{name}' ({label} priority). "
              "Do you want to add a reference point to this zone?")


def _zone_describe(text: str) -> None:
    """Parse a free-text description into a name + priority (Groq, else local).

    The name and priority for a zone are set ONLY through this conversation — there
    are no manual name/priority controls.
    """
    parse = zone_agent.parse_zone_description(text, cc._ZONE_TYPES)
    info = parse.zones[0] if parse.zones else {}
    name = info.get("zone_name") or ""
    label = info.get("priority_label") if info.get("priority_label") in PRIORITY_LABELS else "medium"
    if name:
        st.session_state.lo_zone_name = name
        st.session_state.lo_zone_priority = label
    _zone_say(text, role="user")
    if name:
        if len(st.session_state.lo_zone_pts) == 2:
            _zone_say(f"Got it — '{name}' ({label} priority). Segmenting the box now…")
        else:
            _zone_say(f"Got it — '{name}' ({label} priority). Click two corners on the "
                      "image to box the area — I'll segment it automatically.")
    else:
        _zone_say("I didn't catch a name — tell me what to monitor and its priority, "
                  "e.g. 'dry brush on the left, high priority'.")


def _zone_assistant_panel() -> None:
    with st.container(border=True):
        st.markdown("**✦ Zone assistant**")
        if not st.session_state.lo_zone_chat:
            _zone_say("Tell me what to monitor and its priority (e.g. 'dry brush on the left, "
                      "high priority'). Then click two corners on the image to box it — I'll "
                      "segment it, and you can draw the contour yourself if it's off.")
        # Scrolling conversation.
        with st.container(height=300):
            for msg in st.session_state.lo_zone_chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        _zone_action_row()

        with st.form("lo_zone_desc_form", clear_on_submit=True):
            desc = st.text_input("Message the zone assistant", label_visibility="collapsed",
                                 placeholder="e.g. dry brush on the left ridge, high priority")
            if st.form_submit_button("Send ↑") and desc.strip():
                _zone_describe(desc.strip())
                st.rerun()


def _zone_ref_save() -> None:
    """Write the picked reference point onto the zone (image-space; projected via
    the shared calibration when a detection later falls inside the zone)."""
    zone_id = st.session_state.get("lo_zone_ref_picking")
    pend = st.session_state.get("lo_zone_ref_pending_pt")
    if not zone_id or not pend:
        return
    w, h = st.session_state.cc_image_size
    name = ""
    for z in st.session_state.cc_image_zones:
        if z.get("zone_id") == zone_id:
            z["zone_ref_point_px"] = [float(pend[0]), float(pend[1])]
            z["zone_ref_point_norm"] = [pend[0] / w if w else 0.0, pend[1] / h if h else 0.0]
            name = z.get("zone_name", "")
            break
    st.session_state.lo_zone_ref_picking = None
    st.session_state.lo_zone_ref_pending_pt = None
    _zone_say(f"Reference point saved for '{name}'. Incidents detected in this zone will "
              "be reported from that point on the map.")


def _zone_ref_cancel() -> None:
    st.session_state.lo_zone_ref_picking = None
    st.session_state.lo_zone_ref_pending_pt = None
    _zone_say("No reference point set — this zone will use the shared image-to-map "
              "calibration, like detections outside any zone.")


def _zone_ref_point_row() -> None:
    """Optional per-zone reference point, asked after a zone is saved.

    Yes → the operator clicks the map-reporting point on the image (stored as
    ``zone_ref_point_px`` / ``zone_ref_point_norm`` — existing fields, no new zone
    metadata). No → the zone keeps no reference point and future detections inside
    it are located from the shared image reference points, exactly like detections
    outside any zone. Optional per zone; never forced.
    """
    if st.session_state.get("lo_zone_ref_picking"):
        pend = st.session_state.get("lo_zone_ref_pending_pt")
        st.caption(
            f"Reference point at ({pend[0]:.0f}, {pend[1]:.0f}) — save it or click again."
            if pend else "Click the reference location on the image for this zone."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save reference point ✓", type="primary", key="lo_zone_ref_save",
                         disabled=not pend, use_container_width=True):
                _zone_ref_save(); st.rerun()
        with c2:
            if st.button("Cancel", key="lo_zone_ref_cancel", use_container_width=True):
                _zone_ref_cancel(); st.rerun()
        return

    st.caption("Add a reference point to this zone?")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Yes", type="primary", key="lo_zone_ref_yes", use_container_width=True):
            st.session_state.lo_zone_ref_picking = st.session_state.lo_zone_ref_prompt
            st.session_state.lo_zone_ref_prompt = None
            st.session_state.lo_zone_ref_pending_pt = None
            _zone_say("Click the reference location on the image — incidents detected in "
                      "this zone will be reported from that point.")
            st.rerun()
    with c2:
        if st.button("No", key="lo_zone_ref_no", use_container_width=True):
            st.session_state.lo_zone_ref_prompt = None
            _zone_say("No reference point set — incidents in this zone will use the shared "
                      "image-to-map calibration, like detections outside any zone.")
            st.rerun()


def _zone_maybe_autosegment() -> bool:
    """Run box→segmentation as soon as a 2-point box and a chat name both exist.

    Called at the top of ``_setup_step3`` — BEFORE the frame is composited — so the
    proposed polygon replaces the rectangle in the SAME run (previously it only
    appeared on the next interaction, because the image was drawn before the
    segmentation ran). Returns True when it produced a new polygon.
    """
    if st.session_state.get("lo_zone_ref_prompt") or st.session_state.get("lo_zone_ref_picking"):
        return False
    if (st.session_state.lo_zone_mode == "box"
            and st.session_state.lo_zone_poly is None
            and len(st.session_state.lo_zone_pts) == 2
            and (st.session_state.get("lo_zone_name") or "").strip()):
        _zone_segment()
        return st.session_state.lo_zone_poly is not None
    return False


def _zone_action_row() -> None:
    """Segmentation-first zone flow — all inside the chat frame.

    The box→segmentation step runs in :func:`_zone_maybe_autosegment` before the
    frame is drawn; here we only render Save / Delete / Draw once the proposed
    polygon exists, plus the box/draw guidance captions.
    """
    # After a save, the reference-point prompt/picking takes over the action row.
    if st.session_state.get("lo_zone_ref_prompt") or st.session_state.get("lo_zone_ref_picking"):
        _zone_ref_point_row()
        return

    mode = st.session_state.lo_zone_mode
    pts = st.session_state.lo_zone_pts
    poly = st.session_state.lo_zone_poly
    name = (st.session_state.get("lo_zone_name") or "").strip()

    if poly is not None:
        st.caption("Proposed zone outline:")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Save", type="primary", key="lo_zone_save", use_container_width=True):
                _zone_save(); st.rerun()
        with c2:
            if st.button("Delete", key="lo_zone_delete", use_container_width=True):
                st.session_state.lo_zone_pts = []
                st.session_state.lo_zone_poly = None
                _zone_say("Discarded that outline. Box the area again when you're ready.")
                st.rerun()
        with c3:
            if st.button("Draw", key="lo_zone_draw", use_container_width=True):
                st.session_state.lo_zone_mode = "draw"
                st.session_state.lo_zone_pts = []
                st.session_state.lo_zone_poly = None
                _zone_say("Okay — click points around the area (at least 3), then ‘Save outline’.")
                st.rerun()
        return

    if mode == "box":
        if len(pts) < 2:
            st.caption(f"Click two opposite corners on the image ({len(pts)}/2 picked).")
        elif not name:
            st.caption("Box set — describe the zone in the chat to segment it "
                      "(e.g. 'dry brush on the left, high priority').")
        else:
            st.caption("That box was too small — click two new corners further apart.")
        if pts and st.button("Reset", key="lo_zone_reset_box", use_container_width=True):
            _zone_reset(); st.rerun()
    else:  # draw
        st.caption(f"Click points around the area ({len(pts)} picked; need ≥3).")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Save outline ✓", type="primary", key="lo_zone_save_draw",
                         disabled=len(pts) < 3, use_container_width=True):
                _zone_save(); st.rerun()
        with c2:
            if st.button("Undo point", key="lo_zone_undo", disabled=not pts, use_container_width=True):
                st.session_state.lo_zone_pts = pts[:-1]; st.rerun()
        with c3:
            if st.button("Back to box", key="lo_zone_backbox", use_container_width=True):
                _zone_reset(); st.rerun()


def _zone_list() -> None:
    """Existing zones as clickable cards — clicking a card selects it (highlighted)
    for rename / priority change / delete below. Mirrors the anchor card list in
    Step 2; no dropdown involved.
    """
    zones = st.session_state.cc_image_zones
    if not zones:
        return
    st.markdown("**Zones**")
    selected = st.session_state.get("lo_selected_zone")
    if selected is not None and not any(z["zone_id"] == selected for z in zones):
        st.session_state.lo_selected_zone = None
        selected = None
    per_row = 3
    for start in range(0, len(zones), per_row):
        row = zones[start:start + per_row]
        cols = st.columns(len(row))
        for col, z in zip(cols, row):
            with col:
                is_sel = z["zone_id"] == selected
                label = f"● {z.get('zone_name', '')} · {(z.get('priority_label') or 'medium').upper()}"
                if st.button(label, key=f"lo_zone_card_{z['zone_id']}",
                             type="primary" if is_sel else "secondary",
                             use_container_width=True):
                    st.session_state.lo_selected_zone = None if is_sel else z["zone_id"]
                    st.rerun()
    _zone_edit_delete()


def _zone_edit_delete() -> None:
    """Rename / change priority / delete the selected zone.

    Edits only fields that already exist on the zone record (zone_name, its
    alert_label/object_to_find mirror, and priority) — no new zone metadata.
    """
    zones = st.session_state.cc_image_zones
    target = next((z for z in zones if z["zone_id"] == st.session_state.get("lo_selected_zone")), None)
    if target is None:
        return
    zid = target["zone_id"]
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        new_name = st.text_input("Zone name", value=target.get("zone_name", ""),
                                 key=f"lo_zone_rename_{zid}")
    with c2:
        cur = target.get("priority_label") if target.get("priority_label") in PRIORITY_LABELS else "medium"
        new_priority = st.selectbox("Priority", PRIORITY_LABELS,
                                    index=PRIORITY_LABELS.index(cur), key=f"lo_zone_priority_{zid}")
    with c3:
        st.write("")
        if st.button("Save", key=f"lo_zone_saveedit_{zid}", use_container_width=True):
            name = new_name.strip()
            if name:
                target["zone_name"] = name
                target["alert_label"] = name
                target["object_to_find"] = name
            target["priority_label"] = new_priority
            target["priority"] = priority_label_to_int(new_priority)
            st.rerun()
    if st.button("Delete zone", key=f"lo_zone_delsel_{zid}", use_container_width=True):
        st.session_state.cc_image_zones = [z for z in zones if z["zone_id"] != zid]
        st.session_state.lo_selected_zone = None
        st.rerun()


def _setup_step3() -> None:
    _step_title("Step 3 · Mark detection zones.")
    # Segment BEFORE compositing so the polygon replaces the box in this same run.
    _zone_maybe_autosegment()
    left, right = st.columns([1.3, 1])
    with left:
        picking = bool(st.session_state.get("lo_zone_ref_picking"))
        pending_ref = st.session_state.get("lo_zone_ref_pending_pt")
        comp = cc._composite_image(
            st.session_state.cc_uploaded_image,
            zones=st.session_state.cc_image_zones,
            pending_vertices=None if picking else _zone_overlay_vertices(),
            pending_zone_ref_pt=(tuple(pending_ref) if picking and pending_ref else None),
        )
        click = cc._consume_image_click(comp, key="lo_zone_img")
        if click:
            if picking:
                st.session_state.lo_zone_ref_pending_pt = list(click)
            else:
                _zone_on_click(click)
            st.rerun()
        if picking:
            st.caption("Click the reference location on the image for the saved zone.")
        if not cc.IMG_CLICK_AVAILABLE:
            st.info("Interactive clicking unavailable (streamlit-image-coordinates not installed).")
        _zone_list()
    with right:
        _zone_assistant_panel()

    b1, _, b2 = st.columns([1, 4, 1.4])
    with b1:
        if st.button("← Back", key="lo_s3_back"):
            st.session_state.lo_pending_step = 2
            st.rerun()
    with b2:
        if st.button("Finish — Go live ✓", type="primary", key="lo_s3_finish",
                     use_container_width=True):
            st.session_state.lo_pending_tab = "Live"
            st.rerun()


def _setup_section() -> None:
    if "lo_pending_step" in st.session_state:
        st.session_state.lo_step = st.session_state.pop("lo_pending_step")
    step = st.segmented_control(
        "Setup step", [1, 2, 3],
        format_func=lambda i: f"{i} · {['Place camera', 'Calibrate', 'Zones'][i - 1]}",
        key="lo_step", label_visibility="collapsed",
    ) or 1
    if step == 1:
        _setup_step1()
    elif step == 2:
        _setup_step2()
    else:
        _setup_step3()


# ── LIVE ──────────────────────────────────────────────────────────────────────


def _ensure_sequence_loaded() -> None:
    if st.session_state.get("cc_seq"):
        return
    from src import live_ops_cache

    items, source = lo_cfg.demo_sequence_items(_settings())
    if not items:
        return
    frames = cc._build_sequence_frames(items)
    if not frames:
        return
    st.session_state.cc_seq = frames
    st.session_state.cc_seq_idx = 0
    st.session_state.cc_seq_playing = False
    st.session_state.lo_seq_loaded_from = source
    st.session_state.lo_seq_det = {}
    st.session_state.lo_seq_frames_hash = live_ops_cache.frames_fingerprint(items)
    st.session_state.lo_disk_cache = None
    st.session_state.lo_disk_cache_fp = None
    st.session_state.lo_autoplay_armed = False


def _default_thresholds(settings: dict) -> dict:
    legacy = float(settings.get("confidence_threshold", 0.50))
    return {
        "smoke": round(float(settings.get("smoke_confidence_threshold", legacy)), 4),
        "fire": round(float(settings.get("fire_confidence_threshold", legacy)), 4),
    }


def _is_default_mode(conf_by_class: dict, settings: dict) -> bool:
    """True when the sidebar thresholds equal the configured defaults.

    Only in default mode do we serve the pre-computed disk cache; once the operator
    moves a slider off-default we run the detector live on the chosen thresholds.
    """
    d = _default_thresholds(settings)
    return (round(float(conf_by_class.get("smoke", -1)), 4) == d["smoke"]
            and round(float(conf_by_class.get("fire", -1)), 4) == d["fire"])


def _prepare_disk_cache(conf_by_class: dict, settings: dict) -> None:
    """In default mode, load (or one-time rebuild) the pre-computed detection cache.

    The cache is fingerprinted by frames + default thresholds + model, so it stays
    valid on a fresh clone and rebuilds automatically when the demo frames or the
    default thresholds change. Off-default (slider moved) → no cache, live YOLO.
    """
    from src import inference, live_ops_cache as lc

    if not _is_default_mode(conf_by_class, settings):
        st.session_state.lo_disk_cache = None
        return
    frames = st.session_state.get("cc_seq") or []
    model_name = "YOLO11s" if inference.checkpoint_exists("YOLO11s") else (
        "YOLO11n" if inference.checkpoint_exists("YOLO11n") else None)
    if not frames or model_name is None:
        st.session_state.lo_disk_cache = None
        return
    d = _default_thresholds(settings)
    fp = lc.build_fingerprint(
        st.session_state.get("lo_seq_frames_hash", ""), d["smoke"], d["fire"],
        model_name, len(frames))
    if st.session_state.get("lo_disk_cache_fp") == fp and st.session_state.get("lo_disk_cache"):
        return  # already prepared this session
    manifest = lc.load_manifest()
    if lc.is_valid(manifest, fp):
        st.session_state.lo_disk_cache = manifest["frames"]
    else:
        from PIL import Image
        with st.spinner("Preparing detection cache (one-time for these frames/settings)…"):
            try:
                model = cc._load_detector_cached(model_name)

                def _detect(frame_bytes):
                    return inference.run_detection(
                        model, Image.open(io.BytesIO(frame_bytes)).convert("RGB"),
                        conf=min(d["smoke"], d["fire"]), conf_by_class=d)

                per_frame = lc.build(frames, _detect)
                lc.save_manifest(fp, per_frame)
                st.session_state.lo_disk_cache = per_frame
            except Exception as exc:  # noqa: BLE001 — fall back to live detection
                st.warning(f"Could not build detection cache ({exc}); running live.")
                st.session_state.lo_disk_cache = None
                st.session_state.lo_disk_cache_fp = None
                return
    st.session_state.lo_disk_cache_fp = fp
    st.session_state.lo_disk_model = model_name


def _detect_seq_frame(idx: int, conf_by_class: dict):
    from src import inference, live_ops_cache as lc

    cache = st.session_state.lo_seq_det
    # Cache key is the per-class threshold pair; changing either clears the cache.
    key = (round(float(conf_by_class.get("smoke", 0.0)), 4),
           round(float(conf_by_class.get("fire", 0.0)), 4))
    if st.session_state.lo_seq_det_conf != key:
        cache.clear()
        st.session_state.lo_seq_det_conf = key
    if idx in cache:
        return cache[idx]
    seq = st.session_state.get("cc_seq") or []
    if not (0 <= idx < len(seq)):
        return None

    # Default mode: serve the pre-computed detections (redraw boxes; no YOLO run).
    disk = st.session_state.get("lo_disk_cache")
    if disk is not None and idx < len(disk):
        result = lc.result_from_summary(
            disk[idx], seq[idx]["bytes"],
            st.session_state.get("lo_disk_model") or "YOLO11s")
        cache[idx] = result
        return result

    name = "YOLO11s" if inference.checkpoint_exists("YOLO11s") else (
        "YOLO11n" if inference.checkpoint_exists("YOLO11n") else None)
    if name is None:
        return None
    from PIL import Image
    img = Image.open(io.BytesIO(seq[idx]["bytes"])).convert("RGB")
    try:
        model = cc._load_detector_cached(name)
        result = inference.run_detection(
            model, img, conf=min(conf_by_class.values()), conf_by_class=conf_by_class)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Detection failed: {exc}")
        return None
    result["model_name"] = name
    cache[idx] = result
    return result


def _window_results(idx: int, n_req: int, conf_by_class: dict) -> list:
    start = max(0, idx - n_req)
    return [_detect_seq_frame(i, conf_by_class) for i in range(start, idx + 1)]


def _maybe_confirm(idx: int, window: list, n_req: int) -> None:
    from src import inference

    if st.session_state.lo_confirmed_idx is not None:
        return
    bools = [(inference.top_hazard_detection(r) is not None) if r else False for r in window]
    if st.session_state.get("lo_suppress_until_clear"):
        if not (bools and bools[-1]):
            st.session_state.lo_suppress_until_clear = False
        return
    if not tracking.is_confirmed_with_tolerance(bools, n_req):
        return
    focus = inference.select_confirmed_event_detection(window)
    if focus is None:
        return
    anchor = inference.bbox_bottom_center_norm(focus["bbox_norm"])
    wx = cc._weather_for_incident(anchor)
    ctx = incident_agent.build_incident_context(
        camera=st.session_state.cc_camera,
        image_zones=st.session_state.cc_image_zones,
        reference_points=st.session_state.cc_reference_points,
        detected_class=focus["class"],
        confidence=float(focus["confidence"]),
        centroid_norm=anchor,
        weather=wx,
        timestamp=_now_iso(),
        operational_context=st.session_state.get("lo_operational_context"),
        operational_context_md=st.session_state.get("lo_operational_context_md"),
    )
    st.session_state.cc_incident_ctx = ctx
    st.session_state.cc_incident_weather = wx
    st.session_state.lo_confirmed_idx = idx
    st.session_state.lo_active_alert = {
        "zone": ctx.matched_zone or "monitored area",
        "detected_class": ctx.detected_class,
        "confidence": ctx.confidence,
        "ts": ctx.timestamp,
        "frame_idx": idx,
    }
    st.session_state.cc_seq_playing = False  # freeze on the verified detection frame
    st.session_state.lo_ops_chat.append(
        {"agent": agents.RESPONSE, "role": "assistant",
         "content": agents.emergency_open_text(ctx)})
    # Confirmation happens inside the frame fragment; rerun the WHOLE app so the
    # app-scope status bar + ops chat pick up the new incident (opener + red bar).
    st.rerun(scope="app")


def _maybe_routine_report(force: bool = False) -> None:
    settings = _settings()
    interval = float(settings.get("routine_report_interval_min", 30))
    last = st.session_state.lo_routine_last
    due = force or last is None
    if not due and last is not None:
        try:
            elapsed_min = (datetime.now(timezone.utc)
                           - datetime.fromisoformat(last)).total_seconds() / 60.0
            due = elapsed_min >= interval
        except (ValueError, TypeError):
            due = False
    if not due:
        return
    cam = st.session_state.cc_camera
    wx = weather.fetch_weather(cam.get("latitude"), cam.get("longitude"))
    adv = weather.assess_risk(wx, st.session_state.cc_image_zones)
    st.session_state.lo_ops_chat.append(
        {"agent": agents.WATCH, "role": "assistant",
         "content": agents.routine_status_text(wx, adv)})
    st.session_state.lo_routine_last = _now_iso()


def _render_alert_status() -> None:
    """Always render a fixed-height status bar so the layout never shifts.

    When an incident is active it shows the red alert bar; otherwise it shows a
    calm neutral 'monitoring' bar of the SAME height. Previously the bar appeared
    only during an alert, so the whole screen jumped down (then back up) as the
    alert came and went.
    """
    a = st.session_state.lo_active_alert
    base = ("display:flex;align-items:center;gap:12px;padding:12px 16px;"
            "border-radius:12px;margin:4px 0 8px;min-height:46px;box-sizing:border-box")
    if a:
        cls = str(a["detected_class"]).upper()
        st.markdown(
            f"<div style='{base};background:{_ACCENT};color:#fff'>"
            "<div style='width:10px;height:10px;border-radius:50%;background:#fff'></div>"
            f"<span style='font-weight:700'>{cls} detected — {a['zone']}</span>"
            "<span style='margin-left:auto;font:600 11px \"IBM Plex Mono\",monospace;opacity:.85'>"
            "resolve it in the ops chat →</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='{base};background:rgba(30,122,67,.10);"
            "border:1px solid rgba(30,122,67,.25);color:#1e7a43'>"
            "<div style='width:10px;height:10px;border-radius:50%;background:#1e7a43'></div>"
            "<span style='font-weight:600'>Monitoring — no active incident</span></div>",
            unsafe_allow_html=True,
        )


def _resolve_incident(status: str) -> None:
    ctx = st.session_state.cc_incident_ctx
    if ctx is not None and status in ("confirmed", "false_alarm"):
        alert = incident_agent.create_incident_alert(ctx, status=status)
        st.session_state.cc_alert_log.append(alert)
        _append_event(alert, st.session_state.lo_active_alert)
        st.session_state.lo_ops_chat.append(
            {"agent": agents.RESPONSE, "role": "assistant",
             "content": f"Recorded as **{status.replace('_', ' ')}**. See the History tab. "
                        "Monitoring will resume."})
    st.session_state.cc_incident_ctx = None
    st.session_state.lo_active_alert = None
    st.session_state.lo_confirmed_idx = None
    st.session_state.lo_suppress_until_clear = True
    n = len(st.session_state.get("cc_seq") or [])
    if n:
        st.session_state.cc_seq_pending_seek = min(st.session_state.get("cc_seq_idx", 0) + 1, n - 1)
        st.session_state.cc_seq_playing = st.session_state.get("cc_seq_idx", 0) + 1 < n
    st.rerun()


def _render_live_frame(det, seq, idx, alerting: bool) -> None:
    cam_name = st.session_state.cc_camera.get("camera_name") \
        or st.session_state.cc_camera.get("camera_id", "CAM")
    _status_pill("ALERT" if alerting else "LIVE", cam_name, tone="alert" if alerting else "live")
    if det is not None:
        st.image(det["annotated_png"], use_container_width=True)
    elif seq:
        st.image(seq[idx]["bytes"], use_container_width=True)


def _incident_actions_in_chat(ctx) -> None:
    """Confirm / false-alarm — inside the chat frame.

    There is no separate 'dispatch' panel: any message drafting or contact
    suggestion happens conversationally in the ops chat, where the Response agent
    replies to what the operator asks, grounded in the incident + operational
    context (wind direction, fire location, landmarks, contact policy). PyroFinder
    never contacts anyone automatically.
    """
    if ctx is None or st.session_state.lo_active_alert is None:
        return
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Confirm ✓", type="primary", key="lo_act_confirm", use_container_width=True):
            st.session_state.lo_ops_chat.append(
                {"agent": None, "role": "user", "content": "Confirm the alert."})
            _resolve_incident("confirmed")
    with c2:
        if st.button("False alarm", key="lo_act_false", use_container_width=True):
            st.session_state.lo_ops_chat.append(
                {"agent": None, "role": "user", "content": "Mark as a false alarm."})
            _resolve_incident("false_alarm")


def _ops_chat(ctx) -> None:
    with st.container(border=True):
        st.markdown("**Ops chat**")
        with st.container(height=380):  # scrolling conversation
            for msg in st.session_state.lo_ops_chat:
                role = "assistant" if msg["role"] == "assistant" else "user"
                with st.chat_message(role):
                    agent = msg.get("agent")
                    prefix = (f"**{agents.AGENT_ICON.get(agent, '')} {agent}** · "
                              if (role == "assistant" and agent) else "")
                    st.markdown(prefix + msg["content"])

        _incident_actions_in_chat(ctx)

        if st.button("🌤️ Refresh risk (Watch)", key="lo_watch_refresh", use_container_width=True):
            _maybe_routine_report(force=True)
            st.rerun()
        with st.form("lo_chat_form", clear_on_submit=True):
            msg = st.text_input("Ask about the scene…", label_visibility="collapsed",
                                placeholder="Ask about the scene…")
            sent = st.form_submit_button("Send ↑", use_container_width=True)
        if sent and msg.strip():
            st.session_state.lo_ops_chat.append({"agent": None, "role": "user", "content": msg.strip()})
            if ctx is not None:
                history = [{"role": m["role"], "content": m["content"]}
                           for m in st.session_state.lo_ops_chat if m["role"] in ("user", "assistant")]
                with st.spinner("Response is thinking…"):
                    reply = agents.agent_reply(ctx, msg.strip(), history)
                st.session_state.lo_ops_chat.append(
                    {"agent": agents.RESPONSE, "role": "assistant", "content": reply})
            else:
                st.session_state.lo_ops_chat.append(
                    {"agent": agents.WATCH, "role": "assistant",
                     "content": "No confirmed incident yet — I'll post as soon as an N-frame "
                                "confirmation triggers. Use ‘Refresh risk’ for a weather status."})
            st.rerun()


def _live_sidebar_settings(settings: dict) -> tuple[dict, int]:
    """Return ``({"smoke": thr, "fire": thr}, n_req)`` — per-class confidence.

    Also owns the playback-speed control: it lives in the sidebar (rendered every
    run) so it never resets when a detection toggles the main-area layout, and it
    feeds ``cc_seq_frame_delay_ms`` which the shared autoplay advance reads.
    """
    legacy = float(settings.get("confidence_threshold", 0.50))
    with st.sidebar:
        st.markdown("### Detection settings")
        smoke_default = int(round(float(settings.get("smoke_confidence_threshold", legacy)) * 100))
        fire_default = int(round(float(settings.get("fire_confidence_threshold", legacy)) * 100))
        smoke_pct = st.slider("Smoke confidence (%)", 5, 95, smoke_default, 5, key="lo_conf_pct_smoke")
        fire_pct = st.slider("Fire confidence (%)", 5, 95, fire_default, 5, key="lo_conf_pct_fire")
        n_req = st.number_input("Confirmation frames (N)", 1, 10,
                                int(settings.get("confirmation_frames", 2)), 1, key="lo_confirm_n")
        st.markdown("### Playback")
        speed_ms = st.slider("Playback speed (ms/frame)", 150, 3000, 1500, 50,
                             key="lo_playback_speed_ms",
                             help="Delay between auto-advanced frames while playing.")
    # Feed the shared autoplay advance (self-gates on cc_seq_playing).
    st.session_state.cc_seq_frame_delay_ms = int(speed_ms)
    return {"smoke": smoke_pct / 100.0, "fire": fire_pct / 100.0}, int(n_req)


def _live_playback_controls(n: int) -> None:
    """Prev / Play / Pause / Stop / Next — always visible, even during an alert.

    Replaces the frame slider with step buttons so ``cc_seq_idx`` is a plain,
    stable session variable: the slider was a widget-bound key that Streamlit
    garbage-collected whenever it wasn't rendered (during an alert), which made the
    frame jump on resolve/chat reruns. Manual steps queue ``cc_seq_pending_seek``
    (applied by ``cc._apply_pending_seq_seek`` at the top of the next run) and pause
    playback. On a confirmed detection the frame is simply paused here, not hidden.
    """
    idx = int(st.session_state.get("cc_seq_idx", 0))
    playing = bool(st.session_state.get("cc_seq_playing"))
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("◀ Prev", key="lo_pb_prev", use_container_width=True, disabled=idx <= 0):
            st.session_state.cc_seq_playing = False
            st.session_state.cc_seq_pending_seek = idx - 1
            st.rerun()
    with c2:
        if st.button("▶ Play", key="lo_pb_play", use_container_width=True,
                     disabled=playing or idx >= n - 1):
            st.session_state.cc_seq_playing = True
            st.rerun()
    with c3:
        if st.button("⏸ Pause", key="lo_pb_pause", use_container_width=True, disabled=not playing):
            st.session_state.cc_seq_playing = False
            st.rerun()
    with c4:
        if st.button("⏹ Stop", key="lo_pb_stop", use_container_width=True):
            st.session_state.cc_seq_playing = False
            st.session_state.cc_seq_pending_seek = 0
            st.rerun()
    with c5:
        if st.button("Next ▶", key="lo_pb_next", use_container_width=True, disabled=idx >= n - 1):
            st.session_state.cc_seq_playing = False
            st.session_state.cc_seq_pending_seek = idx + 1
            st.rerun()
    st.caption(f"Frame {idx + 1} / {n}" + (" · playing" if playing else ""))


def _live_section() -> None:
    _ensure_sequence_loaded()
    seq = st.session_state.get("cc_seq") or []
    if not seq:
        st.warning("No demo frames found. Add frames to `data/live_demo/frames` or set "
                   "`video_path` in `config/live_ops.yaml`.")
        return

    # Sidebar + cache prep run on FULL reruns only (they own widgets / a one-time build).
    conf_by_class, n_req = _live_sidebar_settings(_settings())
    _prepare_disk_cache(conf_by_class, _settings())

    if not st.session_state.lo_ops_chat:
        _maybe_routine_report(force=True)
    if not st.session_state.lo_autoplay_armed and st.session_state.lo_active_alert is None:
        st.session_state.cc_seq_playing = True
        st.session_state.lo_autoplay_armed = True

    # Status bar + ops chat render at APP scope so autoplay (which reruns only the
    # frame fragment below) never re-renders them — that is what previously left a
    # ghost/duplicate chat input on screen. A confirmed detection triggers one app
    # rerun (see _maybe_confirm) so they refresh.
    _render_alert_status()
    left, right = st.columns([1.15, 1])
    with left:
        _live_stage(conf_by_class, n_req)
    with right:
        _ops_chat(st.session_state.cc_incident_ctx)


@st.fragment
def _live_stage(conf_by_class: dict, n_req: int) -> None:
    """Auto-advancing frame + playback controls + map, isolated in a fragment.

    Autoplay advances by sleeping then calling ``st.rerun()``; inside a fragment that
    rerun is fragment-scoped, so only this block re-renders every frame — the ops
    chat and status bar (rendered by the caller, at app scope) stay put and no longer
    ghost. NOT the detection cache recomputing (default mode runs no YOLO).
    """
    cc._apply_pending_seq_seek()
    seq = st.session_state.get("cc_seq") or []
    n = len(seq)
    if n == 0:
        return

    idx = min(int(st.session_state.get("cc_seq_idx", 0)), n - 1)
    window = _window_results(idx, n_req, conf_by_class)
    det = window[-1] if window else None
    from src import inference
    current_top = inference.top_hazard_detection(det) if det else None
    _maybe_confirm(idx, window, n_req)  # on a new incident: reruns the whole app (scope="app")
    ctx = st.session_state.cc_incident_ctx
    alerting = st.session_state.lo_active_alert is not None

    _render_live_frame(det, seq, idx, alerting)
    _live_playback_controls(n)
    preview = cc._preview_map_point(current_top) if ctx is None else None
    _render_live_map(ctx, preview_point=preview, height=360)

    # Self-gates on cc_seq_playing: a confirmed detection pauses playback (so this is
    # a no-op) but the controls stay usable; chat reruns never advance the frame.
    cc._advance_playback_if_needed(n)


# ── HISTORY ─────────────────────────────────────────────────────────────────


def _events_path():
    return lo_cfg.REPO_ROOT / "data" / "live_events.jsonl"


def _append_event(alert: dict, active) -> None:
    record = dict(alert)
    record["frame_idx"] = (active or {}).get("frame_idx")
    record["resolved_as"] = alert.get("status")
    try:
        with open(_events_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def _load_events() -> list[dict]:
    events: list[dict] = []
    path = _events_path()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    seen = {(e.get("timestamp"), e.get("status")) for e in events}
    for a in st.session_state.get("cc_alert_log", []):
        if (a.get("timestamp"), a.get("status")) not in seen:
            events.append(a)
    return events


def _history_chart(fdf: pd.DataFrame) -> None:
    import plotly.graph_objects as go

    st.markdown("**Detections per day** · <span style='color:#d9481f'>red = confirmed</span>, "
                "<span style='color:#b9b2a6'>gray = false alarm</span>", unsafe_allow_html=True)
    if fdf.empty:
        st.caption("No events in the selected range.")
        return
    kind = fdf["status"].map(lambda s: "false" if s == "false_alarm" else "confirmed")
    piv = fdf.assign(kind=kind).groupby(["date", "kind"]).size().unstack(fill_value=0)
    for col in ("confirmed", "false"):
        if col not in piv:
            piv[col] = 0
    fig = go.Figure()
    fig.add_bar(x=piv.index.astype(str), y=piv["confirmed"], name="Confirmed", marker_color="#d9481f")
    fig.add_bar(x=piv.index.astype(str), y=piv["false"], name="False alarm", marker_color="#b9b2a6")
    fig.update_layout(barmode="stack", height=240, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      font=dict(color="#1c1b18"), margin=dict(l=30, r=10, t=10, b=30),
                      legend=dict(orientation="h", y=1.15, x=0))
    st.plotly_chart(fig, use_container_width=True)


def _history_list(fdf: pd.DataFrame) -> None:
    st.markdown("**Events**")
    if fdf.empty:
        st.caption("No events in the selected range.")
        return
    for i, (_, e) in enumerate(fdf.sort_values("ts", ascending=False).iterrows()):
        cls = str(e.get("detected_class", "")).upper()
        status = e.get("status", "")
        is_false = status == "false_alarm"
        color = "#b9b2a6" if is_false else "#d9481f"
        conf = e.get("confidence")
        try:
            conf_s = f"{float(conf):.0%}"
        except (TypeError, ValueError):
            conf_s = "—"
        zone = e.get("image_polygon_name") or "—"
        with st.container(border=True):
            c1, c2, c3 = st.columns([6, 2, 1])
            with c1:
                st.markdown(f"<span style='color:{color};font-weight:700'>●</span> "
                            f"**{cls} · {zone}**", unsafe_allow_html=True)
                st.caption(f"{e.get('timestamp', '')} · {e.get('camera_id', '')} · {conf_s}")
            with c2:
                badge = "FALSE" if is_false else cls
                st.markdown(f"<span style='background:{color};color:#fff;padding:2px 8px;"
                            f"border-radius:6px;font-size:11px;font-weight:700'>{badge}</span>",
                            unsafe_allow_html=True)
            with c3:
                fidx = e.get("frame_idx")
                if fidx is not None and not (isinstance(fidx, float) and pd.isna(fidx)):
                    if st.button("Clip", key=f"lo_clip_{i}"):
                        st.session_state.lo_show_clip = int(fidx)

    clip = st.session_state.lo_show_clip
    if clip is not None:
        seq = st.session_state.get("cc_seq") or []
        if 0 <= clip < len(seq):
            st.image(seq[clip]["bytes"], caption=f"Saved frame {clip + 1}", use_container_width=True)
        else:
            st.caption("Clip unavailable — open the Live tab to load the sequence.")


def _history_section() -> None:
    st.markdown("### History")
    events = _load_events()
    if not events:
        st.info("No events yet. Resolve an alert on the Live tab to populate the log.")
        return
    df = pd.DataFrame(events)
    df["ts"] = pd.to_datetime(df.get("timestamp"), errors="coerce", utc=True)
    df["date"] = df["ts"].dt.date
    valid = df.dropna(subset=["ts"])
    dmin = valid["date"].min() if not valid.empty else datetime.now(timezone.utc).date()
    dmax = valid["date"].max() if not valid.empty else dmin

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        dfrom = st.date_input("From", value=dmin, key="lo_hist_from")
    with c2:
        dto = st.date_input("To", value=dmax, key="lo_hist_to")
    with c3:
        flt = st.segmented_control("Type", ["All", "Smoke", "Fire", "False alarms"],
                                   default="All", key="lo_hist_filter") or "All"

    mask = df["date"].apply(lambda d: pd.notna(d) and (dfrom <= d <= dto))
    if flt == "Smoke":
        mask &= df.get("detected_class") == "smoke"
    elif flt == "Fire":
        mask &= df.get("detected_class") == "fire"
    elif flt == "False alarms":
        mask &= df.get("status") == "false_alarm"
    fdf = df[mask]

    _history_chart(fdf)
    _history_list(fdf)


# ── Entry ──────────────────────────────────────────────────────────────────


def render() -> None:
    cc._init_state()
    _init_lo_state()
    _ensure_groq_env()
    _load_config_once()

    if "lo_pending_tab" in st.session_state:
        st.session_state.lo_tab = st.session_state.pop("lo_pending_tab")

    _top_bar()
    tab = st.segmented_control("View", ["Setup", "Live", "History"],
                               key="lo_tab", label_visibility="collapsed") or "Setup"
    st.markdown("---")
    if tab == "Setup":
        _setup_section()
    elif tab == "Live":
        _live_section()
    else:
        _history_section()

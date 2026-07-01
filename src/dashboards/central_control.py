"""Central Control Dashboard renderer.

Camera–Map Calibration and Image Zones setup panel.

Interactive point selection:
  * Map coordinates are captured by clicking a folium map (st_folium `last_clicked`).
  * Image pixels are captured by clicking the camera frame
    (streamlit-image-coordinates).

Both interactive paths degrade gracefully to manual numeric entry if the
component is unavailable. No ML imports (no YOLO, torch, ultralytics).
"""
from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.mapping import (
    build_camera_mapping_config,
    default_camera_metadata,
    estimate_horizon_from_image,
    estimate_map_position,
    normalize_polygon_vertices,
    polygon_centroid_norm,
    validate_camera_metadata,
    validate_image_polygon,
    validate_reference_point,
)
from src import incident_agent, weather, zone_agent
from src.agent_schemas import (
    PRIORITY_LABELS,
    compass_label,
    int_to_priority_label,
    priority_label_to_int,
)

# ── Optional interactive components (graceful fallback) ───────────────────────

try:
    from streamlit_image_coordinates import streamlit_image_coordinates as _img_coords
    IMG_CLICK_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional component
    _img_coords = None
    IMG_CLICK_AVAILABLE = False

_ZONE_TYPES = ["barn", "field", "road", "fence", "parking", "forest_edge", "custom"]

_REF_MARKER = "#E4573D"   # ember glow
_ZONE_LINE = "#8CE9FF"    # hud cyan
_PENDING = "#8F8CC7"      # nordic lilac


@st.cache_resource
def _load_detector_cached(model_name: str):
    """Cache the fine-tuned YOLO detector across reruns.

    ``ultralytics`` is imported lazily inside ``src.inference.load_detector``, so
    importing this dashboard module never pulls in heavy ML libraries.
    """
    from src.inference import load_detector

    return load_detector(model_name)


# ── Session state ─────────────────────────────────────────────────────────────


def _init_state() -> None:
    defaults = {
        "cc_camera": default_camera_metadata(),
        "cc_reference_points": [],
        "cc_image_zones": [],
        "cc_uploaded_image": None,
        "cc_image_size": (640, 480),
        "cc_pending_ref_img": None,   # (x_px, y_px)
        "cc_pending_ref_map": None,   # (lat, lon)
        "cc_pending_vertices": [],    # [[x_px, y_px], ...]
        "cc_cam_click_latlon": None,  # (lat, lon)
        "cc_map_estimate": None,      # generated projection result
        "cc_incident_point": None,    # (x_norm, y_norm) — confirmed hazard point
        "cc_incident_ctx": None,      # assembled IncidentContext
        "cc_incident_detection": None,  # last YOLO run_detection result (overlay + boxes)
        "cc_incident_weather": None,  # Weather used for the current incident
        "cc_incident_chat": [],       # operator conversation [{role, content}]
        "cc_alert_log": [],           # created alert records (this session)
        "cc_risk_weather": None,      # last fetched Weather
        "cc_risk_advisory": None,     # last RiskAdvisory
        "cc_risk_time": None,         # last refresh time (UTC datetime)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Click capture helpers ─────────────────────────────────────────────────────


def _image_click_to_natural(value: dict, natural_w: int, natural_h: int) -> tuple[float, float]:
    """Scale a streamlit-image-coordinates click from displayed to natural pixels."""
    disp_w = value.get("width") or natural_w
    disp_h = value.get("height") or natural_h
    nx = value["x"] * natural_w / disp_w if disp_w else value["x"]
    ny = value["y"] * natural_h / disp_h if disp_h else value["y"]
    nx = min(max(nx, 0.0), float(natural_w))
    ny = min(max(ny, 0.0), float(natural_h))
    return nx, ny


def _consume_image_click(pil_img, key: str) -> tuple[float, float] | None:
    """Render a click-capturing image and return a NEW natural-pixel click, else None.

    De-duplicates repeated reruns using the click's unix_time.
    """
    if not IMG_CLICK_AVAILABLE:
        return None
    natural_w, natural_h = pil_img.size
    value = _img_coords(pil_img, key=key, use_column_width="always", cursor="crosshair")
    if not value:
        return None
    seen_key = f"{key}__seen"
    stamp = value.get("unix_time")
    if stamp is not None and st.session_state.get(seen_key) == stamp:
        return None
    st.session_state[seen_key] = stamp
    return _image_click_to_natural(value, natural_w, natural_h)


def _consume_map_click(map_obj, key: str, last_key: str) -> tuple[float, float] | None:
    """Render a folium map and return a NEW (lat, lon) click, else None."""
    try:
        from streamlit_folium import st_folium
    except ImportError:
        st.info("Map requires `folium` and `streamlit-folium`.")
        return None
    state = st_folium(
        map_obj, key=key, height=320, use_container_width=True,
        returned_objects=["last_clicked"],
    )
    clicked = (state or {}).get("last_clicked")
    if not clicked:
        return None
    latlon = (round(clicked["lat"], 6), round(clicked["lng"], 6))
    if st.session_state.get(last_key) == latlon:
        return None
    st.session_state[last_key] = latlon
    return latlon


# ── Image compositing (draw overlays with PIL) ────────────────────────────────


def _composite_image(
    base_bytes: bytes,
    reference_points: list[dict] | None = None,
    zones: list[dict] | None = None,
    pending_vertices: list | None = None,
    pending_img_pt: tuple[float, float] | None = None,
    horizon_y_px: float | None = None,
):
    """Return a PIL image with reference points, zones and pending markers drawn on."""
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(base_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    r = 6

    if horizon_y_px is not None:
        y = horizon_y_px
        w = img.width
        # Dashed line across the width to mark the estimated sky/ground boundary.
        for x0 in range(0, w, 24):
            draw.line([x0, y, min(x0 + 12, w), y], fill=(243, 244, 248, 230), width=2)
        draw.text((6, max(0, y - 14)), "estimated skyline", fill=(243, 244, 248, 230))

    for pt in (reference_points or []):
        if not pt.get("enabled", True):
            continue
        x, y = pt.get("image_x_px", 0), pt.get("image_y_px", 0)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=_REF_MARKER)
        draw.line([x - r * 2, y, x + r * 2, y], fill=_REF_MARKER, width=2)
        draw.line([x, y - r * 2, x, y + r * 2], fill=_REF_MARKER, width=2)
        draw.text((x + r + 2, y - r - 2), pt.get("point_name") or pt.get("point_id", ""), fill=_REF_MARKER)

    for zone in (zones or []):
        if not zone.get("enabled", True):
            continue
        verts = [tuple(v) for v in zone.get("vertices_px", [])]
        if len(verts) >= 3:
            draw.polygon(verts, outline=_ZONE_LINE, fill=(140, 233, 255, 40))
            cx = sum(v[0] for v in verts) / len(verts)
            cy = sum(v[1] for v in verts) / len(verts)
            draw.text((cx, cy), zone.get("zone_name", ""), fill=_ZONE_LINE)

    pv = [tuple(v) for v in (pending_vertices or [])]
    if pv:
        if len(pv) >= 2:
            draw.line(pv, fill=_PENDING, width=3)
        if len(pv) >= 3:
            draw.line([pv[-1], pv[0]], fill=_PENDING, width=2)
        for i, (x, y) in enumerate(pv, start=1):
            draw.ellipse([x - r, y - r, x + r, y + r], fill=_PENDING)
            draw.text((x + r + 2, y - r - 2), str(i), fill=_PENDING)

    if pending_img_pt is not None:
        x, y = pending_img_pt
        draw.line([x - r * 2, y, x + r * 2, y], fill=_PENDING, width=3)
        draw.line([x, y - r * 2, x, y + r * 2], fill=_PENDING, width=3)

    return img


# ── Skyline overlay helper ────────────────────────────────────────────────────


def _horizon_y_px(checkbox_key: str) -> float | None:
    """Optional skyline overlay: returns the estimated sky/ground boundary in pixels.

    Estimate is cached per uploaded image; recomputed only when the frame changes.
    """
    show = st.checkbox("Show estimated skyline", key=checkbox_key, value=False)
    if not show or not st.session_state.cc_uploaded_image:
        return None

    blen = len(st.session_state.cc_uploaded_image)
    if st.session_state.get("cc_horizon_for") != blen:
        try:
            est = estimate_horizon_from_image(st.session_state.cc_uploaded_image)
        except Exception:
            est = None
        st.session_state.cc_horizon = est
        st.session_state.cc_horizon_for = blen

    est = st.session_state.get("cc_horizon")
    if not est:
        st.caption("No clear skyline detected in this frame.")
        return None
    _, h = st.session_state.cc_image_size
    y_px = est["y_norm"] * h
    st.caption(
        f"Estimated skyline at y ≈ {y_px:.0f} px (confidence {est['confidence']:.2f}) — approximate."
    )
    return y_px


# ── Setup status ──────────────────────────────────────────────────────────────


def _setup_status(camera: dict, reference_points: list, image_zones: list) -> dict:
    cam_errors = validate_camera_metadata(camera)
    has_lat_lon = camera.get("latitude") is not None and camera.get("longitude") is not None
    camera_configured = (
        not cam_errors and bool(camera.get("camera_id", "").strip()) and has_lat_lon
    )
    enabled_refs = [p for p in reference_points if p.get("enabled", True)]
    enabled_zones = [z for z in image_zones if z.get("enabled", True)]
    return {
        "camera_configured": camera_configured,
        "refs_ready": len(enabled_refs) >= 4,
        "zones_ready": len(enabled_zones) >= 1,
        "full_ready": camera_configured and len(enabled_refs) >= 4 and len(enabled_zones) >= 1,
        "n_refs": len(enabled_refs),
        "n_zones": len(enabled_zones),
    }


# ── Shared frame uploader ─────────────────────────────────────────────────────


def _frame_uploader() -> None:
    uploaded = st.file_uploader(
        "Camera frame (shared across Reference Points and Image Zones)",
        type=["jpg", "jpeg", "png"],
        key="cc_frame_upload",
    )
    if uploaded is not None:
        img_bytes = uploaded.getvalue()
        if img_bytes != st.session_state.cc_uploaded_image:
            st.session_state.cc_uploaded_image = img_bytes
            try:
                from PIL import Image
                st.session_state.cc_image_size = Image.open(io.BytesIO(img_bytes)).size
            except Exception:
                pass
    if st.session_state.cc_uploaded_image:
        w, h = st.session_state.cc_image_size
        st.caption(f"Frame loaded — {w}x{h} px.")


# ── Import saved configuration ────────────────────────────────────────────────


def _import_config_panel() -> None:
    """Load a previously exported camera_mapping_config.json back into the app.

    Restores camera metadata, reference points, and any saved image zones from a
    config produced by the Export & Generate tab (the inverse of the download).
    """
    with st.expander("Import saved configuration (camera_mapping_config.json)"):
        st.caption(
            "Restore camera metadata and reference points from a config you "
            "exported earlier. Upload the matching camera frame above to draw and "
            "verify zones."
        )
        uploaded = st.file_uploader("Configuration JSON", type=["json"], key="cc_config_import")
        if uploaded is None or not st.button("Load configuration", key="cc_config_import_btn"):
            return
        try:
            data = json.loads(uploaded.getvalue().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            st.error(f"Could not parse JSON: {exc}")
            return
        if not isinstance(data, dict):
            st.error("Config JSON must be an object with a 'camera' key.")
            return
        cam = data.get("camera") or {}
        refs = data.get("reference_points") or []
        zones = data.get("image_zones") or []
        if not (isinstance(cam, dict) and isinstance(refs, list) and isinstance(zones, list)):
            st.error("Config JSON structure is invalid (expected camera, reference_points, image_zones).")
            return
        defaults = default_camera_metadata()
        merged_cam = {**defaults, **{k: cam[k] for k in defaults if k in cam}}
        st.session_state.cc_camera = merged_cam
        st.session_state.cc_reference_points = refs
        if zones:
            st.session_state.cc_image_zones = zones
        st.session_state.cc_cam_click_latlon = None
        errors = validate_camera_metadata(merged_cam)
        st.success(
            f"Loaded camera '{merged_cam.get('camera_id', '')}', {len(refs)} reference "
            f"point(s), {len(zones)} zone(s)."
        )
        if errors:
            st.caption("Camera validation: " + " | ".join(errors))
        st.rerun()


# ── Tab: Camera Metadata ──────────────────────────────────────────────────────


def _tab_camera_metadata() -> None:
    st.subheader("Camera Metadata")
    cam = st.session_state.cc_camera

    col_map, col_form = st.columns([1, 1])

    with col_map:
        st.markdown("**Click the map to set camera location**")
        latlon = _build_and_consume_camera_map()
        if latlon:
            st.session_state.cc_cam_click_latlon = latlon
            st.rerun()
        pending = st.session_state.cc_cam_click_latlon
        if pending:
            st.caption(f"Picked: lat {pending[0]:.6f}, lon {pending[1]:.6f}")
        elif cam.get("latitude") is not None:
            st.caption(f"Current: lat {cam['latitude']:.6f}, lon {cam['longitude']:.6f}")
        else:
            st.caption("No location picked yet.")

    with col_form:
        # Decide which lat/lon to show: a fresh map pick overrides the saved value.
        pick = st.session_state.cc_cam_click_latlon
        cur_lat = pick[0] if pick else cam.get("latitude")
        cur_lon = pick[1] if pick else cam.get("longitude")

        with st.form("cc_camera_form"):
            camera_id = st.text_input("Camera ID *", value=cam.get("camera_id", ""))
            c1, c2 = st.columns(2)
            with c1:
                customer_id = st.text_input("Customer ID", value=cam.get("customer_id", ""))
                site_id = st.text_input("Site ID", value=cam.get("site_id", ""))
            with c2:
                camera_name = st.text_input("Camera Name", value=cam.get("camera_name", ""))
                io_opts = ["outdoor", "indoor", "unknown"]
                io_val = cam.get("indoor_outdoor", "outdoor")
                indoor_outdoor = st.selectbox(
                    "Indoor / Outdoor", io_opts,
                    index=io_opts.index(io_val) if io_val in io_opts else 0,
                )

            st.caption("Latitude / Longitude — set by clicking the map (editable fallback below).")
            l1, l2 = st.columns(2)
            with l1:
                latitude = st.number_input(
                    "Latitude", min_value=-90.0, max_value=90.0,
                    value=float(cur_lat) if cur_lat is not None else 0.0, format="%.6f",
                )
            with l2:
                longitude = st.number_input(
                    "Longitude", min_value=-180.0, max_value=180.0,
                    value=float(cur_lon) if cur_lon is not None else 0.0, format="%.6f",
                )
            height = st.number_input(
                "Camera Height (m)", min_value=0.1, max_value=200.0,
                value=float(cam.get("camera_height_m") or 4.0), format="%.1f",
            )
            notes = st.text_area("Notes", value=cam.get("notes", ""), height=70)
            submitted = st.form_submit_button("Save Camera Metadata")

        if submitted:
            updated = {
                "camera_id": camera_id.strip(),
                "customer_id": customer_id.strip(),
                "site_id": site_id.strip(),
                "camera_name": camera_name.strip(),
                "latitude": latitude,
                "longitude": longitude,
                "camera_height_m": height,
                "indoor_outdoor": indoor_outdoor,
                "notes": notes.strip(),
            }
            errors = validate_camera_metadata(updated)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.session_state.cc_camera = updated
                st.session_state.cc_cam_click_latlon = None
                st.success("Camera metadata saved.")

    errors = validate_camera_metadata(st.session_state.cc_camera)
    if errors:
        st.caption("Validation: " + " | ".join(errors))


def _build_and_consume_camera_map() -> tuple[float, float] | None:
    try:
        import folium
    except ImportError:
        st.info("Map requires `folium` and `streamlit-folium`. Use the manual fields on the right.")
        return None

    cam = st.session_state.cc_camera
    pick = st.session_state.cc_cam_click_latlon
    center = None
    if pick:
        center = [pick[0], pick[1]]
    elif cam.get("latitude") is not None:
        center = [float(cam["latitude"]), float(cam["longitude"])]
    m = folium.Map(location=center or [32.0853, 34.7818], zoom_start=14 if center else 7)
    if center:
        folium.Marker(center, icon=folium.Icon(color="red", icon="camera", prefix="fa")).add_to(m)
    return _consume_map_click(m, key="cc_cam_map", last_key="cc_cam_map_last")


# ── Tab: Map Reference Points ─────────────────────────────────────────────────


def _tab_map_reference_points() -> None:
    st.subheader("Map Reference Points")
    st.caption(
        "Pick a pixel on the camera image, pick the same real-world spot on the map, "
        "then add the pair. At least 4 enabled pairs enable a map estimate."
    )

    if not st.session_state.cc_uploaded_image:
        st.info("Upload a camera frame above to pick image pixels.")
        return

    col_img, col_map = st.columns(2)

    with col_img:
        st.markdown("**1 · Click a pixel on the image**")
        horizon = _horizon_y_px("cc_ref_horizon")
        composite = _composite_image(
            st.session_state.cc_uploaded_image,
            reference_points=st.session_state.cc_reference_points,
            pending_img_pt=st.session_state.cc_pending_ref_img,
            horizon_y_px=horizon,
        )
        click = _consume_image_click(composite, key="cc_ref_img")
        if click:
            st.session_state.cc_pending_ref_img = click
            st.rerun()
        if not IMG_CLICK_AVAILABLE:
            _manual_image_point_inputs()
        pend_img = st.session_state.cc_pending_ref_img
        st.caption(
            f"Image pixel: ({pend_img[0]:.0f}, {pend_img[1]:.0f})" if pend_img
            else "Image pixel: not picked."
        )

    with col_map:
        st.markdown("**2 · Click the matching spot on the map**")
        latlon = _build_and_consume_ref_map()
        if latlon:
            st.session_state.cc_pending_ref_map = latlon
            st.rerun()
        pend_map = st.session_state.cc_pending_ref_map
        st.caption(
            f"Map: lat {pend_map[0]:.6f}, lon {pend_map[1]:.6f}" if pend_map
            else "Map point: not picked."
        )

    st.markdown("**3 · Add the pair**")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        pt_name = st.text_input("Point name", key="cc_ref_name", placeholder="e.g. Gate corner")
    with c2:
        pt_notes = st.text_input("Notes", key="cc_ref_notes")
    with c3:
        st.write("")
        add = st.button("Add Reference Point", use_container_width=True)

    if add:
        _add_reference_point(pt_name, pt_notes)

    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("Clear picked image pixel"):
            st.session_state.cc_pending_ref_img = None
            st.rerun()
    with cc2:
        if st.button("Clear picked map point"):
            st.session_state.cc_pending_ref_map = None
            st.rerun()

    _render_reference_table()


def _manual_image_point_inputs() -> None:
    w, h = st.session_state.cc_image_size
    st.caption("Interactive image clicking unavailable — enter pixel coordinates:")
    mc1, mc2, mc3 = st.columns([2, 2, 1])
    with mc1:
        mx = st.number_input("Image X (px)", min_value=0.0, max_value=float(w), value=0.0, key="cc_man_x")
    with mc2:
        my = st.number_input("Image Y (px)", min_value=0.0, max_value=float(h), value=0.0, key="cc_man_y")
    with mc3:
        st.write("")
        if st.button("Set pixel"):
            st.session_state.cc_pending_ref_img = (mx, my)
            st.rerun()


def _build_and_consume_ref_map() -> tuple[float, float] | None:
    try:
        import folium
    except ImportError:
        st.info("Map requires `folium` and `streamlit-folium`.")
        return None

    cam = st.session_state.cc_camera
    pend = st.session_state.cc_pending_ref_map
    if pend:
        center = [pend[0], pend[1]]
    elif cam.get("latitude") is not None:
        center = [float(cam["latitude"]), float(cam["longitude"])]
    else:
        center = [32.0853, 34.7818]
    m = folium.Map(location=center, zoom_start=15 if cam.get("latitude") is not None else 7)

    if cam.get("latitude") is not None:
        folium.Marker(
            [float(cam["latitude"]), float(cam["longitude"])],
            icon=folium.Icon(color="red", icon="camera", prefix="fa"),
            popup="Camera",
        ).add_to(m)
    for pt in st.session_state.cc_reference_points:
        if pt.get("enabled", True) and pt.get("map_lat") is not None:
            folium.CircleMarker(
                [float(pt["map_lat"]), float(pt["map_lon"])], radius=6,
                color=_REF_MARKER, fill=True, fill_color=_REF_MARKER,
                popup=pt.get("point_name") or pt.get("point_id", ""),
            ).add_to(m)
    if pend:
        folium.CircleMarker(
            [pend[0], pend[1]], radius=7, color=_PENDING, fill=True, fill_color=_PENDING,
            popup="Pending",
        ).add_to(m)
    return _consume_map_click(m, key="cc_ref_map", last_key="cc_ref_map_last")


def _add_reference_point(pt_name: str, pt_notes: str) -> None:
    pend_img = st.session_state.cc_pending_ref_img
    pend_map = st.session_state.cc_pending_ref_map
    if not pend_img or not pend_map:
        st.error("Pick both an image pixel and a map point before adding.")
        return
    w, h = st.session_state.cc_image_size
    pt = {
        "point_id": str(uuid.uuid4())[:8],
        "point_name": pt_name.strip(),
        "map_lat": pend_map[0],
        "map_lon": pend_map[1],
        "image_x_px": pend_img[0],
        "image_y_px": pend_img[1],
        "image_x_norm": pend_img[0] / w if w else 0.0,
        "image_y_norm": pend_img[1] / h if h else 0.0,
        "enabled": True,
        "notes": pt_notes.strip(),
    }
    errors = validate_reference_point(pt, w, h)
    if errors:
        for e in errors:
            st.error(e)
        return
    st.session_state.cc_reference_points.append(pt)
    st.session_state.cc_pending_ref_img = None
    st.session_state.cc_pending_ref_map = None
    st.success(f"Reference point '{pt_name or pt['point_id']}' added.")
    st.rerun()


def _render_reference_table() -> None:
    pts = st.session_state.cc_reference_points
    if not pts:
        st.info("No reference points added yet.")
        return
    st.markdown("**Reference Points**")
    df = pd.DataFrame(pts)[
        ["point_id", "point_name", "map_lat", "map_lon", "image_x_px", "image_y_px", "enabled", "notes"]
    ]
    st.dataframe(df, use_container_width=True)
    sel = st.selectbox("Select point", [""] + [p["point_id"] for p in pts], key="cc_ref_sel")
    if sel:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Toggle enabled", key="cc_ref_toggle"):
                for p in pts:
                    if p["point_id"] == sel:
                        p["enabled"] = not p.get("enabled", True)
                st.rerun()
        with c2:
            if st.button("Delete point", key="cc_ref_del"):
                st.session_state.cc_reference_points = [p for p in pts if p["point_id"] != sel]
                st.rerun()


# ── Tab: Image Zones ──────────────────────────────────────────────────────────


def _tab_image_zones() -> None:
    st.subheader("Image Zones")
    st.caption(
        "Setup: define the named areas to monitor on this camera. Describe them in "
        "text, ask the vision model for approximate ROI boxes, or draw polygons "
        "manually. (Fire/smoke detection itself runs in the Incident Assistant tab.)"
    )

    if "cc_zone_use_ai" not in st.session_state:
        st.session_state.cc_zone_use_ai = True
    if "cc_zone_drafts" not in st.session_state:
        st.session_state.cc_zone_drafts = []
    if "cc_zone_loaded_draft" not in st.session_state:
        st.session_state.cc_zone_loaded_draft = None
    for _msg_key in ("cc_zone_warnings", "cc_zone_clarifications"):
        if _msg_key not in st.session_state:
            st.session_state[_msg_key] = []
    if "cc_zone_source" not in st.session_state:
        st.session_state.cc_zone_source = None

    if not st.session_state.cc_uploaded_image:
        st.info("Upload a camera frame above to define zones.")
        return

    if st.session_state.cc_zone_use_ai:
        _image_zones_ai_panel()
    else:
        _image_zones_manual_panel()

    _render_zone_table()


def _render_zone_frame_preview() -> None:
    """Show the current frame with existing zones drawn (static preview)."""
    w, h = st.session_state.cc_image_size
    n_zones = len([z for z in st.session_state.cc_image_zones if z.get("enabled", True)])
    composite = _composite_image(
        st.session_state.cc_uploaded_image, zones=st.session_state.cc_image_zones
    )
    st.image(
        composite,
        use_container_width=True,
        caption=f"Current frame — {w}×{h} px · {n_zones} zone(s) defined",
    )


def _zone_vertex_editor(img_key: str, horizon_key: str) -> None:
    """Image click-to-vertex editor shared by the manual and AI-assisted panels."""
    horizon = _horizon_y_px(horizon_key)
    composite = _composite_image(
        st.session_state.cc_uploaded_image,
        zones=st.session_state.cc_image_zones,
        pending_vertices=st.session_state.cc_pending_vertices,
        horizon_y_px=horizon,
    )
    click = _consume_image_click(composite, key=img_key)
    if click:
        st.session_state.cc_pending_vertices.append([click[0], click[1]])
        st.rerun()
    if not IMG_CLICK_AVAILABLE:
        _manual_vertex_input()
    n = len(st.session_state.cc_pending_vertices)
    st.caption(f"Vertices picked: {n}" + (" (need ≥3)" if n < 3 else ""))
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Undo last point", key=f"{img_key}_undo", disabled=n == 0):
            st.session_state.cc_pending_vertices.pop()
            st.rerun()
    with b2:
        if st.button("Clear points", key=f"{img_key}_clear", disabled=n == 0):
            st.session_state.cc_pending_vertices = []
            st.rerun()


def _image_zones_manual_panel() -> None:
    if st.button("Switch to AI-assisted", key="cc_zone_to_ai"):
        st.session_state.cc_zone_use_ai = True
        st.rerun()
    col_img, col_form = st.columns([1.15, 1])
    with col_img:
        st.markdown("**Click to add polygon vertices**")
        _zone_vertex_editor("cc_zone_img", "cc_zone_horizon")
    with col_form:
        st.markdown("**Zone details**")
        with st.form("cc_zone_form"):
            zone_name = st.text_input("Zone Name *", placeholder="e.g. East Barn")
            zone_type = st.selectbox("Zone Type", _ZONE_TYPES)
            alert_label = st.text_input("Alert Label", placeholder="e.g. East Barn")
            priority = st.number_input("Priority", min_value=1, max_value=10, value=5, step=1)
            zone_notes = st.text_input("Notes", key="cc_zone_notes")
            save = st.form_submit_button("Save Zone")
        if save:
            _save_zone(zone_name, zone_type, alert_label, int(priority), zone_notes)


def _render_parse_messages() -> None:
    """Show the source, warnings, and clarification requests from the last parse."""
    source = st.session_state.get("cc_zone_source")
    if source == "local":
        st.caption("Structured with the built-in parser (no GROQ_API_KEY configured).")
    elif source == "groq":
        st.caption("Structured with Groq.")
    for warning in st.session_state.get("cc_zone_warnings") or []:
        st.warning(warning)
    for clarification in st.session_state.get("cc_zone_clarifications") or []:
        st.info(clarification)


def _clear_ai_drafts() -> None:
    st.session_state.cc_zone_drafts = []
    st.session_state.cc_pending_vertices = []
    st.session_state.cc_zone_loaded_draft = None
    st.session_state.cc_zone_warnings = []
    st.session_state.cc_zone_clarifications = []
    st.session_state.cc_zone_source = None
    st.rerun()


def _image_zones_ai_panel() -> None:
    left_col, right_col = st.columns([1.15, 1])
    with left_col:
        _render_zone_frame_preview()
    with right_col:
        st.markdown("**Describe the areas to monitor**")
        st.caption("One area per line — what to monitor and a priority.")
        desc = st.text_area(
            "Areas to monitor",
            key="cc_zone_ai_text",
            placeholder=(
                'the hay storage area, called "Hay Storage", high priority\n'
                'left hill named "giva ktana", low priority\n'
                'the right forest edge, named "East Grove", medium'
            ),
            height=150,
            label_visibility="collapsed",
        )
        if st.button("Structure from text", use_container_width=True,
                     help="Turns your text into named zones + priority — you draw the shapes. Works without a Groq key."):
            _generate_zones_from_text(desc)
        if st.button("Detect on image (AI vision)", type="primary", use_container_width=True,
                     help="A Groq vision model proposes approximate ROI boxes you then verify."):
            _detect_zones_from_image(desc)
        if st.button("Switch to manual drawing", use_container_width=True, key="cc_zone_to_manual"):
            st.session_state.cc_zone_use_ai = False
            st.rerun()
        if st.session_state.cc_zone_drafts and st.button("Clear AI drafts", use_container_width=True):
            _clear_ai_drafts()

    _render_parse_messages()

    drafts = st.session_state.cc_zone_drafts
    if not drafts:
        st.caption(
            "No zone drafts yet. “Structure from text” turns your description into named "
            "zones with a low/medium/high priority (accept them as pending and draw later, "
            "or draw each polygon now); “Detect on image” overlays approximate ROI boxes "
            "from a Groq vision model that you verify."
        )
        return
    _place_draft_zones(drafts)


def _generate_zones_from_text(description: str) -> None:
    if not description or not description.strip():
        st.warning("Describe at least one area first.")
        return
    with st.spinner("Structuring your zones…"):
        # parse_zone_description uses Groq when a key exists and otherwise falls
        # back to the deterministic local parser — it never raises for a missing key.
        result = zone_agent.parse_zone_description(description, _ZONE_TYPES)

    st.session_state.cc_zone_warnings = result.warnings
    st.session_state.cc_zone_clarifications = result.clarifications
    st.session_state.cc_zone_source = result.source

    if not result.zones:
        st.session_state.cc_zone_drafts = []
        if not (result.warnings or result.clarifications):
            st.session_state.cc_zone_clarifications = [
                'No areas found. Try, e.g. the hay storage area, called "Hay Storage", high priority.'
            ]
        return
    for zone in result.zones:
        zone["draft_id"] = str(uuid.uuid4())[:8]
    st.session_state.cc_zone_drafts = result.zones
    st.session_state.cc_pending_vertices = []
    st.session_state.cc_zone_loaded_draft = None
    st.rerun()


def _frame_for_vision(max_side: int = 1024):
    """Downscale the uploaded frame and JPEG-encode it for the vision model."""
    from PIL import Image

    img = Image.open(io.BytesIO(st.session_state.cc_uploaded_image)).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h)) if max(w, h) else 1.0
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


def _box_norm_to_vertices(box, w: int, h: int) -> list:
    """Convert a normalized [x0,y0,x1,y1] box to pixel polygon vertices."""
    x0, y0, x1, y1 = box
    return [[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h], [x0 * w, y1 * h]]


def _detect_zones_from_image(description: str) -> None:
    if not description or not description.strip():
        st.warning("Describe at least one area first.")
        return
    if not st.session_state.cc_uploaded_image:
        st.warning("Upload a camera frame first.")
        return
    # Pre-flight checks so the AI-vision button explains exactly what is missing
    # (no silent fallback — the operator explicitly asked for vision).
    from src import llm
    if not llm.groq_available():
        st.error(
            "Groq vision is unavailable because the groq package is not installed. "
            "Run pip install -r requirements.txt and restart the app."
        )
        return
    if not llm.api_key_present():
        st.error(
            "Groq vision requires GROQ_API_KEY. Add it to Streamlit secrets or "
            "environment variables."
        )
        return
    try:
        img_bytes, mime = _frame_for_vision()
        with st.spinner("Asking the Groq vision model to locate your areas…"):
            zones = llm.detect_zone_boxes(img_bytes, description, _ZONE_TYPES, mime=mime)
    except Exception as exc:
        st.error(f"Groq vision request failed: {exc}")
        return
    if not zones:
        st.warning("The vision model returned no zones. Try rephrasing your description.")
        return
    w, h = st.session_state.cc_image_size
    n_boxes = 0
    for z in zones:
        z["draft_id"] = str(uuid.uuid4())[:8]
        box = z.get("box_norm")
        if box:
            z["vertices_px"] = _box_norm_to_vertices(box, w, h)
            n_boxes += 1
    # Vision drafts are their own source of truth — clear any text-parse messages.
    st.session_state.cc_zone_warnings = []
    st.session_state.cc_zone_clarifications = []
    st.session_state.cc_zone_source = None
    st.session_state.cc_zone_drafts = zones
    st.session_state.cc_pending_vertices = []
    st.session_state.cc_zone_loaded_draft = None
    st.success(
        f"Vision model proposed {n_boxes} ROI box(es) across {len(zones)} zone(s). "
        "These are APPROXIMATE — select each below to see and verify it."
    )
    st.rerun()


def _as_op_zone(draft: dict) -> dict:
    """Normalize a draft (text or vision) into an operational zone record."""
    try:
        priority_int = int(draft.get("priority", 5))
    except (TypeError, ValueError):
        priority_int = 5
    label = draft.get("priority_label") or int_to_priority_label(priority_int)
    return {
        "zone_name": draft.get("zone_name", ""),
        "zone_type": draft.get("zone_type", "custom"),
        "alert_label": draft.get("alert_label", "") or draft.get("zone_name", ""),
        "priority_label": label,
        "priority": priority_label_to_int(label),
        "object_to_find": draft.get("object_to_find", ""),
        "notes": draft.get("notes", ""),
        "requires_user_confirmation": bool(draft.get("requires_user_confirmation", True)),
    }


def _accept_pending_drafts(drafts: list[dict]) -> None:
    """Accept all drafts into the zone table as pending (no polygon drawn yet)."""
    for draft in drafts:
        entry = zone_agent.build_zone_table_entry(_as_op_zone(draft), str(uuid.uuid4())[:8])
        st.session_state.cc_image_zones.append(entry)
    count = len(drafts)
    st.session_state.cc_zone_drafts = []
    st.session_state.cc_pending_vertices = []
    st.session_state.cc_zone_loaded_draft = None
    st.success(
        f"Accepted {count} zone(s) as pending. Draw each polygon later — until then "
        "they carry no image geometry and won't match detections."
    )
    st.rerun()


def _place_draft_zones(drafts: list[dict]) -> None:
    st.markdown("**2 · Draft zones** — targets and priorities from your description")
    st.dataframe(
        pd.DataFrame([
            {
                "#": i + 1,
                "zone_name": d.get("zone_name", ""),
                "object_to_find": d.get("object_to_find", ""),
                "zone_type": d.get("zone_type", ""),
                "priority": d.get("priority_label") or int_to_priority_label(int(d.get("priority", 5))),
                "AI ROI": "estimate" if d.get("vertices_px") else "—",
            }
            for i, d in enumerate(drafts)
        ]),
        use_container_width=True,
    )
    st.caption(
        "Every draft needs your confirmation before use. Accept them all as pending "
        "and draw polygons later, or verify one on the image below and save it with "
        "its polygon."
    )
    if st.button(f"Accept all {len(drafts)} zone(s) as pending (draw polygons later)",
                 use_container_width=True, key="cc_zone_accept_pending"):
        _accept_pending_drafts(drafts)

    st.markdown("**3 · Verify each zone on the image, then save it**")
    options = [
        f"{i + 1}. {d.get('zone_name', '')} "
        f"({d.get('priority_label') or int_to_priority_label(int(d.get('priority', 5)))})"
        for i, d in enumerate(drafts)
    ]
    sel = st.selectbox("Zone to place", options, key="cc_zone_draft_sel")
    active = drafts[options.index(sel)]
    did = active["draft_id"]

    # When the selected draft changes, load its AI box (if any) into the editor so
    # the model's estimated ROI is drawn on the image; empty for text-only drafts.
    if st.session_state.cc_zone_loaded_draft != did:
        st.session_state.cc_pending_vertices = [list(v) for v in active.get("vertices_px", [])]
        st.session_state.cc_zone_loaded_draft = did
        st.rerun()

    has_box = bool(active.get("vertices_px"))
    col_img, col_form = st.columns([1, 1])
    with col_img:
        st.markdown(f"{'Verify the AI estimate for' if has_box else 'Click vertices for'} "
                    f"**{active.get('zone_name', '')}**")
        if has_box:
            st.caption("Dashed outline = approximate AI estimate — adjust before saving.")
        _zone_vertex_editor("cc_zone_ai_img", "cc_zone_ai_horizon")
        if has_box and st.button("Reset to AI estimate", key=f"cc_ai_reset_{did}"):
            st.session_state.cc_pending_vertices = [list(v) for v in active["vertices_px"]]
            st.rerun()
    with col_form:
        st.markdown("**Confirm details**")
        object_to_find = st.text_input(
            "Object to find", value=active.get("object_to_find", ""), key=f"cc_ai_obj_{did}",
            help="What to monitor in this zone. A zone target — not a detector class.",
        )
        name = st.text_input("Zone Name *", value=active.get("zone_name", ""), key=f"cc_ai_name_{did}")
        ztype = st.selectbox(
            "Zone Type", _ZONE_TYPES,
            index=(
                _ZONE_TYPES.index(active["zone_type"])
                if active.get("zone_type") in _ZONE_TYPES else _ZONE_TYPES.index("custom")
            ),
            key=f"cc_ai_type_{did}",
        )
        alert_label = st.text_input("Alert Label", value=active.get("alert_label", ""), key=f"cc_ai_label_{did}")
        current_label = active.get("priority_label") or int_to_priority_label(int(active.get("priority", 5)))
        priority_label = st.selectbox(
            "Priority", list(PRIORITY_LABELS),
            index=list(PRIORITY_LABELS).index(current_label) if current_label in PRIORITY_LABELS else 1,
            key=f"cc_ai_prio_{did}",
        )
        notes = st.text_input("Notes", value=active.get("notes", ""), key=f"cc_ai_notes_{did}")
        if st.button("Save this zone", type="primary", key=f"cc_ai_save_{did}"):
            extra = {
                "object_to_find": object_to_find,
                "priority_label": priority_label,
                "requires_user_confirmation": True,
            }
            if _commit_zone(name, ztype, alert_label, priority_label_to_int(priority_label), notes, extra=extra):
                st.session_state.cc_zone_drafts = [
                    d for d in st.session_state.cc_zone_drafts if d["draft_id"] != did
                ]
                st.session_state.cc_pending_vertices = []
                st.success(f"Zone '{name}' saved.")
                st.rerun()


def _manual_vertex_input() -> None:
    st.caption("Interactive image clicking unavailable — paste vertices as JSON [[x,y],...]:")
    txt = st.text_area("Vertices JSON", key="cc_zone_json",
                       placeholder="[[120, 240], [330, 250], [350, 420], [110, 410]]", height=80)
    if st.button("Set vertices"):
        try:
            verts = json.loads(txt)
            if not isinstance(verts, list):
                raise ValueError("expected a list")
            st.session_state.cc_pending_vertices = [[float(v[0]), float(v[1])] for v in verts]
            st.rerun()
        except (json.JSONDecodeError, ValueError, TypeError, IndexError) as exc:
            st.error(f"Invalid vertices JSON: {exc}")


def _commit_zone(zone_name, zone_type, alert_label, priority, zone_notes, extra=None) -> bool:
    """Validate the pending vertices + details and append a zone.

    Returns True on success (caller handles messaging / rerun); False if the
    zone name is missing or the polygon fails validation. ``extra`` carries the
    operational fields (object_to_find, priority_label, requires_user_confirmation)
    when a zone comes from the AI-assisted flow; manual zones pass ``None``.
    """
    verts = list(st.session_state.cc_pending_vertices)
    w, h = st.session_state.cc_image_size
    if not zone_name.strip():
        st.error("Zone Name is required.")
        return False
    extra = extra or {}
    try:
        priority_int = int(priority)
    except (TypeError, ValueError):
        priority_int = 5
    zone = {
        "zone_id": str(uuid.uuid4())[:8],
        "zone_name": zone_name.strip(),
        "zone_type": zone_type,
        "alert_label": alert_label.strip() or zone_name.strip(),
        "priority": priority_int,
        "priority_label": extra.get("priority_label") or int_to_priority_label(priority_int),
        "object_to_find": str(extra.get("object_to_find", "")).strip(),
        "requires_user_confirmation": bool(extra.get("requires_user_confirmation", False)),
        "vertices_px": verts,
        "vertices_norm": [],
        "enabled": True,
        "notes": zone_notes.strip(),
        "polygon_status": zone_agent.POLYGON_DRAWN,
    }
    errors = validate_image_polygon(zone, w, h)
    if errors:
        for e in errors:
            st.error(e)
        return False
    zone["vertices_norm"] = normalize_polygon_vertices(verts, w, h)
    st.session_state.cc_image_zones.append(zone)
    st.session_state.cc_pending_vertices = []
    return True


def _save_zone(zone_name, zone_type, alert_label, priority, zone_notes) -> None:
    if _commit_zone(zone_name, zone_type, alert_label, priority, zone_notes):
        st.success(f"Zone '{zone_name}' added.")
        st.rerun()


def _render_zone_table() -> None:
    zones = st.session_state.cc_image_zones
    if not zones:
        st.info("No zones defined yet.")
        return
    st.markdown("**Image Zones**")
    df = pd.DataFrame([
        {
            "zone_id": z.get("zone_id", ""), "zone_name": z.get("zone_name", ""),
            "object_to_find": z.get("object_to_find", ""),
            "zone_type": z.get("zone_type", ""),
            "alert_label": z.get("alert_label") or z.get("zone_name", ""),
            "priority": z.get("priority_label") or int_to_priority_label(int(z.get("priority", 5))),
            "polygon": (
                "pending" if z.get("polygon_status") == zone_agent.POLYGON_PENDING
                or len(z.get("vertices_px", [])) < 3 else "drawn"
            ),
            "vertices": len(z.get("vertices_px", [])),
            "enabled": z.get("enabled", True), "notes": z.get("notes", ""),
        }
        for z in zones
    ])
    st.dataframe(df, use_container_width=True)
    sel = st.selectbox("Select zone", [""] + [z["zone_id"] for z in zones], key="cc_zone_sel")
    if sel:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Toggle enabled", key="cc_zone_toggle"):
                for z in zones:
                    if z["zone_id"] == sel:
                        z["enabled"] = not z.get("enabled", True)
                st.rerun()
        with c2:
            if st.button("Load into editor", key="cc_zone_load"):
                for z in zones:
                    if z["zone_id"] == sel:
                        st.session_state.cc_pending_vertices = [list(v) for v in z["vertices_px"]]
                st.rerun()
        with c3:
            if st.button("Delete zone", key="cc_zone_del"):
                st.session_state.cc_image_zones = [z for z in zones if z["zone_id"] != sel]
                st.rerun()


# ── Tab: Export & Generate ────────────────────────────────────────────────────


def _tab_export() -> None:
    st.subheader("Export & Generate")

    st.markdown("**Generate map estimate from reference points**")
    st.caption(
        "Uses at least 4 reference point pairs to estimate approximate map positions "
        "for each image zone (locally planar assumption). Estimates are approximate "
        "and depend on your reference points."
    )
    enabled_refs = [p for p in st.session_state.cc_reference_points if p.get("enabled", True)]
    if st.button("Generate Map Estimate", disabled=len(enabled_refs) < 4):
        _generate_map_estimate(enabled_refs)
    if len(enabled_refs) < 4:
        st.caption(f"{len(enabled_refs)} / 4 reference points — add more to enable generation.")

    if st.session_state.cc_map_estimate:
        _render_estimate()

    st.markdown("---")
    config = build_camera_mapping_config(
        st.session_state.cc_camera,
        st.session_state.cc_reference_points,
        st.session_state.cc_image_zones,
    )
    config_json = json.dumps(config, indent=2, default=str)
    st.markdown("**Config JSON preview**")
    st.code(config_json, language="json")
    st.download_button("Download Full Config (JSON)", config_json,
                       file_name="camera_mapping_config.json", mime="application/json")

    if st.session_state.cc_reference_points:
        st.download_button(
            "Download Reference Points (CSV)",
            pd.DataFrame(st.session_state.cc_reference_points).to_csv(index=False),
            file_name="reference_points.csv", mime="text/csv",
        )
    if st.session_state.cc_image_zones:
        rows = [
            {k: v for k, v in z.items() if k not in ("vertices_px", "vertices_norm")}
            for z in st.session_state.cc_image_zones
        ]
        st.download_button(
            "Download Image Zones (CSV)", pd.DataFrame(rows).to_csv(index=False),
            file_name="image_zones.csv", mime="text/csv",
        )


def _generate_map_estimate(enabled_refs: list[dict]) -> None:
    estimates = []
    for z in st.session_state.cc_image_zones:
        if not z.get("enabled", True):
            continue
        centroid = polygon_centroid_norm([tuple(v) for v in z.get("vertices_norm", [])])
        if centroid is None:
            continue
        latlon = estimate_map_position(enabled_refs, centroid)
        verts_latlon = []
        for vn in z.get("vertices_norm", []):
            proj = estimate_map_position(enabled_refs, tuple(vn))
            if proj:
                verts_latlon.append([proj[0], proj[1]])
        if latlon:
            estimates.append({
                "zone_id": z["zone_id"], "zone_name": z["zone_name"],
                "est_lat": latlon[0], "est_lon": latlon[1],
                "vertices_latlon": verts_latlon,
            })
    st.session_state.cc_map_estimate = {"zones": estimates}
    if not estimates:
        st.warning("No zones with valid vertices to project.")
    else:
        st.success(f"Generated approximate map positions for {len(estimates)} zone(s).")
    st.rerun()


def _render_estimate() -> None:
    est = st.session_state.cc_map_estimate
    zones = est.get("zones", [])
    if not zones:
        return
    st.dataframe(
        pd.DataFrame([{"zone_name": z["zone_name"], "est_lat": z["est_lat"], "est_lon": z["est_lon"]} for z in zones]),
        use_container_width=True,
    )
    try:
        import folium
        from streamlit_folium import st_folium
        cam = st.session_state.cc_camera
        center = [zones[0]["est_lat"], zones[0]["est_lon"]]
        m = folium.Map(location=center, zoom_start=16)
        if cam.get("latitude") is not None:
            folium.Marker([float(cam["latitude"]), float(cam["longitude"])],
                          icon=folium.Icon(color="red", icon="camera", prefix="fa"),
                          popup="Camera").add_to(m)
        for p in st.session_state.cc_reference_points:
            if p.get("enabled", True) and p.get("map_lat") is not None:
                folium.CircleMarker([float(p["map_lat"]), float(p["map_lon"])], radius=5,
                                    color=_REF_MARKER, fill=True).add_to(m)
        for z in zones:
            if len(z["vertices_latlon"]) >= 3:
                folium.Polygon(z["vertices_latlon"], color=_ZONE_LINE, fill=True,
                               fill_opacity=0.25, popup=z["zone_name"]).add_to(m)
            folium.Marker([z["est_lat"], z["est_lon"]],
                          icon=folium.Icon(color="blue", icon="fire", prefix="fa"),
                          popup=z["zone_name"]).add_to(m)
        st_folium(m, key="cc_estimate_map", height=360, use_container_width=True,
                  returned_objects=[])
    except ImportError:
        st.info("Install `folium` and `streamlit-folium` to view the estimate on a map.")


# ── Tab: Incident Assistant ───────────────────────────────────────────────────


def _incident_point_picker() -> None:
    """Let the operator place the confirmed hazard point on the frame (or by coords)."""
    if st.session_state.cc_uploaded_image and IMG_CLICK_AVAILABLE:
        w, h = st.session_state.cc_image_size
        cur = st.session_state.cc_incident_point
        cur_px = (cur[0] * w, cur[1] * h) if cur else None
        composite = _composite_image(
            st.session_state.cc_uploaded_image,
            zones=st.session_state.cc_image_zones,
            pending_img_pt=cur_px,
        )
        click = _consume_image_click(composite, key="cc_incident_img")
        if click:
            st.session_state.cc_incident_point = (
                click[0] / w if w else 0.0, click[1] / h if h else 0.0
            )
            st.rerun()
        st.caption("Click the frame where the detector confirmed the hazard.")
        return

    if st.session_state.cc_uploaded_image and not IMG_CLICK_AVAILABLE:
        st.caption("Interactive clicking unavailable — enter the hazard point as normalized coords (0–1).")
    else:
        st.caption("Upload a camera frame above to click the point, or enter normalized coords (0–1).")
    cur = st.session_state.cc_incident_point
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        hx = st.number_input("Hazard X (0–1)", 0.0, 1.0,
                             value=float(cur[0]) if cur else 0.5, step=0.01, key="cc_inc_x")
    with c2:
        hy = st.number_input("Hazard Y (0–1)", 0.0, 1.0,
                             value=float(cur[1]) if cur else 0.5, step=0.01, key="cc_inc_y")
    with c3:
        st.write("")
        if st.button("Set point", key="cc_inc_setpt"):
            st.session_state.cc_incident_point = (hx, hy)
            st.rerun()


def _incident_prev_point() -> tuple[float, float] | None:
    """Optional previous position, used only to estimate apparent image-plane direction."""
    with st.expander("Optional: previous hazard position (for apparent image-plane direction)"):
        if not st.checkbox("Provide a previous position", key="cc_inc_use_prev"):
            return None
        c1, c2 = st.columns(2)
        with c1:
            px = st.number_input("Prev X (0–1)", 0.0, 1.0, 0.5, 0.01, key="cc_inc_px")
        with c2:
            py = st.number_input("Prev Y (0–1)", 0.0, 1.0, 0.5, 0.01, key="cc_inc_py")
        return (px, py)


def _reset_incident_state() -> None:
    st.session_state.cc_incident_ctx = None
    st.session_state.cc_incident_point = None
    st.session_state.cc_incident_detection = None
    st.session_state.cc_incident_weather = None
    st.session_state.cc_incident_chat = []


def _clear_incident() -> None:
    _reset_incident_state()
    st.rerun()


def _log_incident_alert(ctx, status: str) -> None:
    alert = incident_agent.create_incident_alert(ctx, status=status)
    st.session_state.cc_alert_log.append(alert)
    _reset_incident_state()
    st.success(f"Alert recorded with status '{status}'. See the Alert Log below.")
    st.rerun()


def _render_incident_result() -> None:
    ctx = st.session_state.cc_incident_ctx
    if ctx is None:
        return

    st.markdown("**Incident summary**")
    st.table(pd.DataFrame(ctx.display_rows(), columns=["Field", "Value"]))

    st.markdown("**Operational recommendations**")
    for rec in incident_agent.recommend_actions(ctx):
        st.markdown(f"- {rec}")

    st.markdown("**Draft messages** — review and send them yourself; nothing is sent automatically.")
    refine = st.checkbox("Refine wording with AI (optional — needs GROQ_API_KEY)", key="cc_inc_refine")
    drafts = incident_agent.build_drafts(ctx)
    if refine:
        with st.spinner("Refining drafts…"):
            drafts = {aud: incident_agent.polish_message(text, aud) for aud, text in drafts.items()}
    for audience, text in drafts.items():
        with st.expander(f"Draft — {audience}"):
            st.text_area(audience, value=text, height=170, key=f"cc_inc_draft_{audience}")

    st.markdown("**Alert decision**")
    st.caption(
        "Confirm before contacting anyone. PyroFinder never contacts emergency services "
        "or dispatches automatically."
    )
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Confirm alert", type="primary", key="cc_inc_confirm"):
            _log_incident_alert(ctx, "confirmed")
    with a2:
        if st.button("Mark as false alarm", key="cc_inc_false"):
            _log_incident_alert(ctx, "false_alarm")
    with a3:
        if st.button("Clear incident", key="cc_inc_clear"):
            _clear_incident()


def _render_alert_log() -> None:
    log = st.session_state.cc_alert_log
    st.markdown("**Alert log (this session)**")
    if not log:
        st.info("No alerts recorded yet.")
        return
    df = pd.DataFrame([
        {
            "timestamp": a["timestamp"], "camera_id": a["camera_id"],
            "class": a["detected_class"], "confidence": f"{a['confidence']:.0%}",
            "zone": a.get("image_polygon_name"), "location": a["approximate_location"],
            "status": a["status"],
        }
        for a in log
    ])
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download alert log (CSV)", pd.DataFrame(log).to_csv(index=False),
        file_name="alert_log.csv", mime="text/csv",
    )


def _weather_for_incident(centroid_norm):
    """Fetch weather for the incident: event map point → camera lat/lon → none."""
    latlon = estimate_map_position(st.session_state.cc_reference_points, centroid_norm)
    if latlon is None:
        cam = st.session_state.cc_camera
        lat, lon = cam.get("latitude"), cam.get("longitude")
        latlon = (lat, lon) if lat is not None and lon is not None else None
    if latlon is None:
        return None
    try:
        return weather.fetch_weather(latlon[0], latlon[1])
    except Exception:
        return None


def _assess_incident(detected_class, confidence, centroid_norm, prev_centroid_norm) -> None:
    """Build the incident context (+weather) and open the operational conversation."""
    wx = _weather_for_incident(centroid_norm)
    ctx = incident_agent.build_incident_context(
        camera=st.session_state.cc_camera,
        image_zones=st.session_state.cc_image_zones,
        reference_points=st.session_state.cc_reference_points,
        detected_class=detected_class,
        confidence=float(confidence),
        centroid_norm=centroid_norm,
        prev_centroid_norm=prev_centroid_norm,
        weather=wx,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    st.session_state.cc_incident_ctx = ctx
    st.session_state.cc_incident_weather = wx
    st.session_state.cc_incident_point = centroid_norm
    st.session_state.cc_incident_chat = [
        {"role": "assistant", "content": incident_agent.incident_narrative(ctx)}
    ]
    st.rerun()


def _run_yolo_incident() -> None:
    """Run the fine-tuned YOLO11s detector on the current frame (YOLO11n fallback)."""
    if not st.session_state.cc_uploaded_image:
        st.warning("Upload a camera frame above first.")
        return
    from PIL import Image
    from src import inference

    if inference.checkpoint_exists("YOLO11s"):
        model_name = "YOLO11s"
    elif inference.checkpoint_exists("YOLO11n"):
        model_name = "YOLO11n"
    else:
        st.error(inference.MISSING_YOLO11S_MESSAGE)
        return
    try:
        image = Image.open(io.BytesIO(st.session_state.cc_uploaded_image)).convert("RGB")
        with st.spinner(f"Running {model_name} on the current frame…"):
            model = _load_detector_cached(model_name)
            result = inference.run_detection(model, image, conf=0.25)
    except FileNotFoundError:
        st.error(inference.MISSING_YOLO11S_MESSAGE)
        return
    except Exception as exc:
        st.error(f"Detection failed: {exc}")
        return

    result["model_name"] = model_name
    st.session_state.cc_incident_detection = result
    top = inference.top_hazard_detection(result)
    if top is None:
        st.session_state.cc_incident_ctx = None
        st.session_state.cc_incident_chat = []
        st.session_state.cc_incident_weather = None
        st.rerun()
        return
    anchor = inference.bbox_bottom_center_norm(top["bbox_norm"])
    _assess_incident(top["class"], float(top["confidence"]), anchor, None)


def _render_detection_overlay(det: dict) -> None:
    st.image(
        det["annotated_png"], use_container_width=True,
        caption=f"{det.get('model_name', 'YOLO11s')} — {det['total_detections']} detection(s)",
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Fire", det["fire_count"])
    c2.metric("Smoke", det["smoke_count"])
    hc = det.get("max_confidence")
    c3.metric("Top confidence", f"{hc:.0%}" if hc is not None else "—")
    if det["total_detections"] == 0:
        st.info("No fire/smoke detection in this frame.")


def _render_incident_conversation() -> None:
    ctx = st.session_state.cc_incident_ctx
    st.markdown("**Incident conversation**")
    if incident_agent.conversation_uses_llm():
        st.caption("Replies are generated by Groq from the incident context.")
    else:
        st.caption(
            "Groq not configured — replies use the built-in responder. Add GROQ_API_KEY "
            "for a free-form assistant."
        )
    for msg in st.session_state.cc_incident_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    prompt = st.chat_input("e.g. Update the farm workers near this zone")
    if prompt:
        history = list(st.session_state.cc_incident_chat)
        st.session_state.cc_incident_chat.append({"role": "user", "content": prompt})
        with st.spinner("Thinking…"):
            reply = incident_agent.respond_to_operator(ctx, prompt, history=history)
        st.session_state.cc_incident_chat.append({"role": "assistant", "content": reply})
        st.rerun()


def _tab_incident_assistant() -> None:
    st.subheader("Incident Assistant")
    st.caption(
        "Incident workflow: run the YOLO11s fire/smoke detector on the current frame, "
        "match the detection to a mapped zone, pull Open-Meteo weather/wind, and get "
        "recommendations and draft messages. The assistant only drafts and recommends — "
        "it never contacts anyone or dispatches automatically."
    )

    if not st.session_state.cc_camera.get("camera_id", "").strip():
        st.info("Set a Camera ID in the Camera Metadata tab so alerts are attributable.")

    st.markdown("**1 · Detect fire/smoke on the current frame**")
    run = st.button(
        "Run YOLO11s fire/smoke detector", type="primary", key="cc_inc_run_yolo",
        disabled=st.session_state.cc_uploaded_image is None,
    )
    if st.session_state.cc_uploaded_image is None:
        st.caption("Upload a camera frame above to run the detector.")
    if run:
        _run_yolo_incident()

    det = st.session_state.get("cc_incident_detection")
    if det is not None:
        _render_detection_overlay(det)

    # Manual fallback (no nested expander: revealed by a checkbox).
    if st.checkbox("No detector available? Set the hazard point manually", key="cc_inc_manual"):
        col_a, col_b = st.columns(2)
        with col_a:
            manual_class = st.selectbox("Detected class", ["fire", "smoke"], key="cc_inc_class")
        with col_b:
            manual_conf = st.slider("Confidence", 0.0, 1.0, 0.8, 0.05, key="cc_inc_conf")
        _incident_point_picker()
        cur = st.session_state.cc_incident_point
        st.caption(f"Hazard point: ({cur[0]:.3f}, {cur[1]:.3f})" if cur else "Hazard point: not set.")
        prev = _incident_prev_point()
        if st.button("Assess from manual point", disabled=cur is None, key="cc_inc_assess"):
            _assess_incident(manual_class, float(manual_conf), cur, prev)

    if st.session_state.cc_incident_ctx is not None:
        st.markdown("---")
        _render_incident_conversation()
        _render_incident_result()

    st.markdown("---")
    _render_alert_log()


# ── Tab: Risk Advisory ────────────────────────────────────────────────────────

_RISK_BADGE = {"low": "🟢", "moderate": "🟡", "high": "🟠", "extreme": "🔴"}


def _render_risk_result(interval_min: int) -> None:
    wx = st.session_state.cc_risk_weather
    adv = st.session_state.cc_risk_advisory
    last = st.session_state.cc_risk_time
    if wx is None or adv is None:
        st.info("Click **Refresh now** to fetch current weather and generate a risk advisory.")
        return

    if last is not None:
        elapsed_min = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
        if elapsed_min >= interval_min:
            due = " · **a check is due**"
        else:
            due = f" · next check in ~{max(0, interval_min - elapsed_min):.0f} min"
        st.caption(f"Last checked {elapsed_min:.0f} min ago · source: {wx.source}{due}")

    if not wx.is_live:
        st.warning(
            "Live weather is unavailable — showing deterministic fallback/demo data. "
            "Recommendations below are illustrative until a live reading succeeds."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Temperature", f"{wx.temperature_c:.0f} °C" if wx.temperature_c is not None else "—")
    m2.metric("Humidity", f"{wx.relative_humidity:.0f}%" if wx.relative_humidity is not None else "—")
    m3.metric("Wind", f"{wx.wind_speed_kmh:.0f} km/h" if wx.wind_speed_kmh is not None else "—")
    wind_from = compass_label(wx.wind_direction_deg) if wx.wind_direction_deg is not None else "—"
    m4.metric("Wind from", wind_from)

    badge = _RISK_BADGE.get(adv.level, "⚪")
    st.markdown(
        f"**Fire-weather risk: {badge} {adv.level.upper()}** (score {adv.score}) · source: {wx.source}"
    )
    if adv.downwind:
        st.caption(f"Downwind risk direction: {adv.downwind}")
    if adv.factors:
        st.caption("Contributing factors: " + ", ".join(adv.factors))

    st.markdown("**Preventive recommendations**")
    for advisory in adv.advisories:
        st.markdown(f"- {advisory}")


def _tab_risk_advisory() -> None:
    st.subheader("Risk Advisory")
    st.caption(
        "Preventive fire-weather risk advisory based on current weather and your configured "
        "zones. This is advisory guidance — not an early-warning alert, an ignition "
        "prediction, or an emergency dispatch."
    )
    st.caption("Weather context uses Open-Meteo and does not require an API key.")

    cam = st.session_state.cc_camera
    lat, lon = cam.get("latitude"), cam.get("longitude")
    if lat is None or lon is None:
        st.info("Set the camera location (Camera Metadata tab) for site-specific weather. "
                "Using a default location meanwhile.")

    c1, c2 = st.columns([2, 1])
    with c1:
        interval_min = st.number_input(
            "Check interval (minutes)", min_value=5, max_value=720,
            value=int(st.session_state.get("cc_risk_interval_min", 60)), step=5,
            key="cc_risk_interval_min",
            help="How often to re-check. Streamlit has no safe background scheduler here, "
                 "so use Refresh now when a check is due.",
        )
    with c2:
        st.write("")
        st.write("")
        refresh = st.button("Refresh now", type="primary", use_container_width=True, key="cc_risk_refresh")

    if refresh:
        with st.spinner("Fetching current weather…"):
            wx = weather.fetch_weather(lat, lon)
            adv = weather.assess_risk(wx, st.session_state.cc_image_zones)
        st.session_state.cc_risk_weather = wx
        st.session_state.cc_risk_advisory = adv
        st.session_state.cc_risk_time = datetime.now(timezone.utc)
        st.rerun()

    _render_risk_result(int(interval_min))


# ── Main render ───────────────────────────────────────────────────────────────


def render() -> None:
    st.header("Central Control Dashboard")
    st.caption("Configure camera location, image zones, and reference points.")

    _init_state()

    status = _setup_status(
        st.session_state.cc_camera,
        st.session_state.cc_reference_points,
        st.session_state.cc_image_zones,
    )
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.caption(("✓ " if status["camera_configured"] else "○ ") + "Camera configured")
    with sc2:
        st.caption(("✓ " if status["refs_ready"] else "○ ") + f"Reference points ({status['n_refs']} / 4)")
    with sc3:
        st.caption(("✓ " if status["zones_ready"] else "○ ") + f"Image zones ({status['n_zones']})")
    with sc4:
        st.caption(("✓ " if status["full_ready"] else "○ ") + "Full setup ready")

    _frame_uploader()
    _import_config_panel()
    st.markdown("---")

    tab_cam, tab_ref, tab_zones, tab_export, tab_incident, tab_risk = st.tabs(
        ["Camera Metadata", "Map Reference Points", "Image Zones", "Export & Generate",
         "Incident Assistant", "Risk Advisory"]
    )
    with tab_cam:
        _tab_camera_metadata()
    with tab_ref:
        _tab_map_reference_points()
    with tab_zones:
        _tab_image_zones()
    with tab_export:
        _tab_export()
    with tab_incident:
        _tab_incident_assistant()
    with tab_risk:
        _tab_risk_advisory()

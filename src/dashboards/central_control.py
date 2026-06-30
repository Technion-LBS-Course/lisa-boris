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

    if "cc_zone_use_ai" not in st.session_state:
        st.session_state.cc_zone_use_ai = True
    if "cc_zone_drafts" not in st.session_state:
        st.session_state.cc_zone_drafts = []
    if "cc_zone_loaded_draft" not in st.session_state:
        st.session_state.cc_zone_loaded_draft = None
    use_ai = st.session_state.cc_zone_use_ai

    head_l, head_r = st.columns([3, 1])
    with head_l:
        if use_ai:
            st.caption(
                "AI-assisted (default): describe the areas and their priority in plain "
                "text. “Structure from text” names and prioritises them (you draw the "
                "shapes); “Detect on image” asks a vision model for APPROXIMATE ROI "
                "boxes you then verify and adjust."
            )
        else:
            st.caption(
                "Manual: click points on the image to outline a named zone (4 points "
                "recommended), then fill in its details. When a detection falls inside a "
                "zone, the alert can name it, e.g. 'Fire detected in East Barn.'"
            )
    with head_r:
        if use_ai and st.button("Switch to manual drawing", use_container_width=True):
            st.session_state.cc_zone_use_ai = False
            st.rerun()
        if not use_ai and st.button("Switch to AI-assisted", use_container_width=True):
            st.session_state.cc_zone_use_ai = True
            st.rerun()

    if not st.session_state.cc_uploaded_image:
        st.info("Upload a camera frame above to draw zones.")
        return

    if use_ai:
        _image_zones_ai_panel()
    else:
        _image_zones_manual_panel()

    _render_zone_table()


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
    col_img, col_form = st.columns([1, 1])
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


def _image_zones_ai_panel() -> None:
    st.markdown("**1 · Describe the areas — one per line — with a priority**")
    desc = st.text_area(
        "Areas to monitor",
        key="cc_zone_ai_text",
        placeholder=(
            "white building on the left, high priority\n"
            "hill behind the field, medium\n"
            "barn, low"
        ),
        height=110,
    )
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Structure from text", use_container_width=True,
                     help="Names, types and priorities only — you draw the shapes."):
            _generate_zones_from_text(desc)
    with b2:
        if st.button("Detect on image (AI vision)", type="primary", use_container_width=True,
                     help="A vision model proposes approximate ROI boxes you then verify."):
            _detect_zones_from_image(desc)
    with b3:
        if st.session_state.cc_zone_drafts and st.button(
            "Clear AI drafts", use_container_width=True
        ):
            st.session_state.cc_zone_drafts = []
            st.session_state.cc_pending_vertices = []
            st.session_state.cc_zone_loaded_draft = None
            st.rerun()

    drafts = st.session_state.cc_zone_drafts
    if not drafts:
        st.caption(
            "No AI drafts yet. “Structure from text” fills name/type/priority (you draw "
            "each polygon); “Detect on image” overlays approximate ROI boxes from a "
            "vision model that you verify. Prefer drawing yourself? Use “Switch to "
            "manual drawing”."
        )
        return
    _place_draft_zones(drafts)


def _generate_zones_from_text(description: str) -> None:
    if not description or not description.strip():
        st.warning("Describe at least one area first.")
        return
    try:
        from src.llm import extract_zones
    except Exception as exc:  # groq missing / import failure — keep the tab usable
        st.error(f"LLM helper unavailable: {exc}")
        return
    try:
        with st.spinner("Structuring your zones…"):
            zones = extract_zones(description, _ZONE_TYPES)
    except Exception as exc:
        st.error(f"Could not generate zones: {exc}")
        st.caption("Check GROQ_API_KEY in Settings → Secrets, then try again.")
        return
    if not zones:
        st.warning("The model returned no zones. Try rephrasing your description.")
        return
    for z in zones:
        z["draft_id"] = str(uuid.uuid4())[:8]
    st.session_state.cc_zone_drafts = zones
    st.session_state.cc_pending_vertices = []
    st.session_state.cc_zone_loaded_draft = None
    st.success(f"Structured {len(zones)} zone(s). Draw each one on the image below.")
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
    try:
        from src.llm import detect_zone_boxes
    except Exception as exc:  # groq missing / import failure — keep the tab usable
        st.error(f"LLM helper unavailable: {exc}")
        return
    try:
        img_bytes, mime = _frame_for_vision()
        with st.spinner("Asking the vision model to locate your areas…"):
            zones = detect_zone_boxes(img_bytes, description, _ZONE_TYPES, mime=mime)
    except Exception as exc:
        st.error(f"Vision detection failed: {exc}")
        st.caption("Check GROQ_API_KEY, and that the vision model id in src/llm.py is current.")
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
    st.session_state.cc_zone_drafts = zones
    st.session_state.cc_pending_vertices = []
    st.session_state.cc_zone_loaded_draft = None
    st.success(
        f"Vision model proposed {n_boxes} ROI box(es) across {len(zones)} zone(s). "
        "These are APPROXIMATE — select each below to see and verify it."
    )
    st.rerun()


def _place_draft_zones(drafts: list[dict]) -> None:
    st.markdown("**2 · Draft zones** — names and priorities from your description")
    st.dataframe(
        pd.DataFrame([
            {
                "#": i + 1, "zone_name": d["zone_name"], "zone_type": d["zone_type"],
                "priority": d["priority"], "alert_label": d["alert_label"],
                "AI ROI": "estimate" if d.get("vertices_px") else "—",
            }
            for i, d in enumerate(drafts)
        ]),
        use_container_width=True,
    )

    st.markdown("**3 · Verify each zone on the image, then save it**")
    options = [f"{i + 1}. {d['zone_name']} (priority {d['priority']})" for i, d in enumerate(drafts)]
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
                    f"**{active['zone_name']}**")
        if has_box:
            st.caption("Dashed outline = approximate AI estimate — adjust before saving.")
        _zone_vertex_editor("cc_zone_ai_img", "cc_zone_ai_horizon")
        if has_box and st.button("Reset to AI estimate", key=f"cc_ai_reset_{did}"):
            st.session_state.cc_pending_vertices = [list(v) for v in active["vertices_px"]]
            st.rerun()
    with col_form:
        st.markdown("**Confirm details**")
        name = st.text_input("Zone Name *", value=active["zone_name"], key=f"cc_ai_name_{did}")
        ztype = st.selectbox(
            "Zone Type", _ZONE_TYPES,
            index=(
                _ZONE_TYPES.index(active["zone_type"])
                if active["zone_type"] in _ZONE_TYPES else _ZONE_TYPES.index("custom")
            ),
            key=f"cc_ai_type_{did}",
        )
        alert_label = st.text_input("Alert Label", value=active["alert_label"], key=f"cc_ai_label_{did}")
        priority = st.number_input(
            "Priority", min_value=1, max_value=10, value=int(active["priority"]), step=1,
            key=f"cc_ai_prio_{did}",
        )
        notes = st.text_input("Notes", value=active["notes"], key=f"cc_ai_notes_{did}")
        if st.button("Save this zone", type="primary", key=f"cc_ai_save_{did}"):
            if _commit_zone(name, ztype, alert_label, int(priority), notes):
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


def _commit_zone(zone_name, zone_type, alert_label, priority, zone_notes) -> bool:
    """Validate the pending vertices + details and append a zone.

    Returns True on success (caller handles messaging / rerun); False if the
    zone name is missing or the polygon fails validation.
    """
    verts = list(st.session_state.cc_pending_vertices)
    w, h = st.session_state.cc_image_size
    if not zone_name.strip():
        st.error("Zone Name is required.")
        return False
    zone = {
        "zone_id": str(uuid.uuid4())[:8],
        "zone_name": zone_name.strip(),
        "zone_type": zone_type,
        "alert_label": alert_label.strip() or zone_name.strip(),
        "priority": priority,
        "vertices_px": verts,
        "vertices_norm": [],
        "enabled": True,
        "notes": zone_notes.strip(),
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
            "zone_id": z["zone_id"], "zone_name": z["zone_name"], "zone_type": z["zone_type"],
            "alert_label": z["alert_label"], "priority": z["priority"],
            "vertices": len(z["vertices_px"]), "enabled": z["enabled"], "notes": z["notes"],
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
    st.markdown("---")

    tab_cam, tab_ref, tab_zones, tab_export = st.tabs(
        ["Camera Metadata", "Map Reference Points", "Image Zones", "Export & Generate"]
    )
    with tab_cam:
        _tab_camera_metadata()
    with tab_ref:
        _tab_map_reference_points()
    with tab_zones:
        _tab_image_zones()
    with tab_export:
        _tab_export()

"""Tests for src/mapping.py — camera calibration helpers."""

import json

import pytest

from src.mapping import (
    apply_homography,
    bbox_bottom_center_px,
    build_camera_mapping_config,
    compute_homography,
    default_camera_metadata,
    denormalize_image_point,
    downwind_arrow_endpoint,
    downwind_bearing_deg,
    estimate_horizon_y_norm,
    estimate_map_position,
    find_zone_for_detection,
    format_approximate_location,
    generate_zone_map_estimates,
    image_quadrant,
    normalize_image_point,
    normalize_polygon_vertices,
    point_in_polygon,
    polygon_centroid_norm,
    validate_camera_metadata,
    validate_image_polygon,
    validate_reference_point,
    validate_zone_reference_point,
    zone_reference_point_norm,
)


# ── default_camera_metadata ────────────────────────────────────────────────────


def test_default_camera_height_is_4():
    cam = default_camera_metadata()
    assert cam["camera_height_m"] == 4.0


def test_default_camera_metadata_has_required_keys():
    cam = default_camera_metadata()
    for key in [
        "camera_id", "customer_id", "site_id", "camera_name",
        "latitude", "longitude", "camera_height_m", "indoor_outdoor", "notes",
    ]:
        assert key in cam


def test_default_camera_id_is_empty():
    assert default_camera_metadata()["camera_id"] == ""


def test_default_indoor_outdoor_is_outdoor():
    assert default_camera_metadata()["indoor_outdoor"] == "outdoor"


# ── validate_camera_metadata ───────────────────────────────────────────────────


def test_validate_camera_metadata_valid():
    cam = {
        "camera_id": "cam_001",
        "latitude": 32.0,
        "longitude": 34.8,
        "camera_height_m": 4.0,
        "indoor_outdoor": "outdoor",
    }
    assert validate_camera_metadata(cam) == []


def test_validate_camera_metadata_missing_id():
    cam = {
        "camera_id": "",
        "latitude": 32.0,
        "longitude": 34.8,
        "camera_height_m": 4.0,
        "indoor_outdoor": "outdoor",
    }
    errors = validate_camera_metadata(cam)
    assert any("camera_id" in e for e in errors)


def test_validate_camera_metadata_whitespace_id():
    cam = {
        "camera_id": "   ",
        "latitude": 32.0,
        "longitude": 34.8,
        "camera_height_m": 4.0,
        "indoor_outdoor": "outdoor",
    }
    errors = validate_camera_metadata(cam)
    assert any("camera_id" in e for e in errors)


def test_validate_camera_metadata_invalid_lat_too_high():
    cam = {
        "camera_id": "cam_001",
        "latitude": 95.0,
        "longitude": 34.8,
        "camera_height_m": 4.0,
        "indoor_outdoor": "outdoor",
    }
    errors = validate_camera_metadata(cam)
    assert any("latitude" in e for e in errors)


def test_validate_camera_metadata_invalid_lat_too_low():
    cam = {
        "camera_id": "cam_001",
        "latitude": -91.0,
        "longitude": 34.8,
        "camera_height_m": 4.0,
        "indoor_outdoor": "outdoor",
    }
    errors = validate_camera_metadata(cam)
    assert any("latitude" in e for e in errors)


def test_validate_camera_metadata_invalid_lon_too_high():
    cam = {
        "camera_id": "cam_001",
        "latitude": 32.0,
        "longitude": 200.0,
        "camera_height_m": 4.0,
        "indoor_outdoor": "outdoor",
    }
    errors = validate_camera_metadata(cam)
    assert any("longitude" in e for e in errors)


def test_validate_camera_metadata_invalid_lon_too_low():
    cam = {
        "camera_id": "cam_001",
        "latitude": 32.0,
        "longitude": -185.0,
        "camera_height_m": 4.0,
        "indoor_outdoor": "outdoor",
    }
    errors = validate_camera_metadata(cam)
    assert any("longitude" in e for e in errors)


def test_validate_camera_metadata_invalid_height_zero():
    cam = {
        "camera_id": "cam_001",
        "latitude": 32.0,
        "longitude": 34.8,
        "camera_height_m": 0.0,
        "indoor_outdoor": "outdoor",
    }
    errors = validate_camera_metadata(cam)
    assert any("camera_height_m" in e for e in errors)


def test_validate_camera_metadata_invalid_height_negative():
    cam = {
        "camera_id": "cam_001",
        "latitude": 32.0,
        "longitude": 34.8,
        "camera_height_m": -3.0,
        "indoor_outdoor": "outdoor",
    }
    errors = validate_camera_metadata(cam)
    assert any("camera_height_m" in e for e in errors)


def test_validate_camera_metadata_invalid_indoor_outdoor():
    cam = {
        "camera_id": "cam_001",
        "latitude": 32.0,
        "longitude": 34.8,
        "camera_height_m": 4.0,
        "indoor_outdoor": "rooftop",
    }
    errors = validate_camera_metadata(cam)
    assert any("indoor_outdoor" in e for e in errors)


def test_validate_camera_metadata_valid_indoor():
    cam = {
        "camera_id": "cam_002",
        "latitude": 32.0,
        "longitude": 34.8,
        "camera_height_m": 2.5,
        "indoor_outdoor": "indoor",
    }
    assert validate_camera_metadata(cam) == []


def test_validate_camera_metadata_none_lat_lon_is_ok():
    # lat/lon None means not yet configured — only validated when provided
    cam = {
        "camera_id": "cam_001",
        "latitude": None,
        "longitude": None,
        "camera_height_m": 4.0,
        "indoor_outdoor": "outdoor",
    }
    assert validate_camera_metadata(cam) == []


# ── normalize / denormalize ───────────────────────────────────────────────────


def test_normalize_image_point_center():
    xn, yn = normalize_image_point(320, 240, 640, 480)
    assert xn == pytest.approx(0.5)
    assert yn == pytest.approx(0.5)


def test_normalize_image_point_origin():
    xn, yn = normalize_image_point(0, 0, 640, 480)
    assert xn == pytest.approx(0.0)
    assert yn == pytest.approx(0.0)


def test_normalize_image_point_corner():
    xn, yn = normalize_image_point(640, 480, 640, 480)
    assert xn == pytest.approx(1.0)
    assert yn == pytest.approx(1.0)


def test_denormalize_image_point_center():
    xp, yp = denormalize_image_point(0.5, 0.5, 640, 480)
    assert xp == pytest.approx(320.0)
    assert yp == pytest.approx(240.0)


def test_denormalize_image_point_origin():
    xp, yp = denormalize_image_point(0.0, 0.0, 640, 480)
    assert xp == pytest.approx(0.0)
    assert yp == pytest.approx(0.0)


def test_normalize_denormalize_roundtrip():
    x_px, y_px = 123.0, 456.0
    xn, yn = normalize_image_point(x_px, y_px, 640, 480)
    x2, y2 = denormalize_image_point(xn, yn, 640, 480)
    assert x2 == pytest.approx(x_px)
    assert y2 == pytest.approx(y_px)


# ── bbox_bottom_center_px ─────────────────────────────────────────────────────


def test_bbox_bottom_center_px_center():
    x_px, y_px = bbox_bottom_center_px((0.5, 0.5, 0.2, 0.4), 640, 480)
    assert x_px == pytest.approx(320.0)
    assert y_px == pytest.approx(336.0)  # (0.5 + 0.2) * 480


def test_bbox_bottom_center_px_top_left():
    x_px, y_px = bbox_bottom_center_px((0.0, 0.0, 0.0, 0.0), 640, 480)
    assert x_px == pytest.approx(0.0)
    assert y_px == pytest.approx(0.0)


# ── validate_reference_point ──────────────────────────────────────────────────


def test_validate_reference_point_valid():
    pt = {"map_lat": 32.0, "map_lon": 34.8, "image_x_px": 100.0, "image_y_px": 200.0}
    assert validate_reference_point(pt, 640, 480) == []


def test_validate_reference_point_x_too_large():
    pt = {"map_lat": 32.0, "map_lon": 34.8, "image_x_px": 700.0, "image_y_px": 200.0}
    errors = validate_reference_point(pt, 640, 480)
    assert any("image_x_px" in e for e in errors)


def test_validate_reference_point_y_too_large():
    pt = {"map_lat": 32.0, "map_lon": 34.8, "image_x_px": 100.0, "image_y_px": 600.0}
    errors = validate_reference_point(pt, 640, 480)
    assert any("image_y_px" in e for e in errors)


def test_validate_reference_point_bad_lat():
    pt = {"map_lat": 200.0, "map_lon": 34.8, "image_x_px": 100.0, "image_y_px": 100.0}
    errors = validate_reference_point(pt, 640, 480)
    assert any("map_lat" in e for e in errors)


def test_validate_reference_point_bad_lon():
    pt = {"map_lat": 32.0, "map_lon": -200.0, "image_x_px": 100.0, "image_y_px": 100.0}
    errors = validate_reference_point(pt, 640, 480)
    assert any("map_lon" in e for e in errors)


# ── validate_image_polygon ────────────────────────────────────────────────────


def test_validate_image_polygon_valid():
    poly = {"vertices_px": [[10, 10], [100, 10], [100, 100], [10, 100]]}
    assert validate_image_polygon(poly, 640, 480) == []


def test_validate_image_polygon_too_few_vertices():
    poly = {"vertices_px": [[10, 10], [100, 10]]}
    errors = validate_image_polygon(poly, 640, 480)
    assert any("3" in e for e in errors)


def test_validate_image_polygon_one_vertex():
    poly = {"vertices_px": [[10, 10]]}
    errors = validate_image_polygon(poly, 640, 480)
    assert len(errors) > 0


def test_validate_image_polygon_empty_vertices():
    poly = {"vertices_px": []}
    errors = validate_image_polygon(poly, 640, 480)
    assert len(errors) > 0


def test_validate_image_polygon_vertex_x_outside():
    poly = {"vertices_px": [[10, 10], [700, 10], [100, 100]]}
    errors = validate_image_polygon(poly, 640, 480)
    assert len(errors) > 0


def test_validate_image_polygon_vertex_y_outside():
    poly = {"vertices_px": [[10, 10], [100, 10], [100, 600]]}
    errors = validate_image_polygon(poly, 640, 480)
    assert len(errors) > 0


# ── normalize_polygon_vertices ────────────────────────────────────────────────


def test_normalize_polygon_vertices_square():
    verts = [[0, 0], [640, 0], [640, 480], [0, 480]]
    norm = normalize_polygon_vertices(verts, 640, 480)
    assert norm[0] == pytest.approx((0.0, 0.0))
    assert norm[1] == pytest.approx((1.0, 0.0))
    assert norm[2] == pytest.approx((1.0, 1.0))
    assert norm[3] == pytest.approx((0.0, 1.0))


def test_normalize_polygon_vertices_center():
    verts = [[320, 240]]
    norm = normalize_polygon_vertices(verts, 640, 480)
    assert norm[0] == pytest.approx((0.5, 0.5))


# ── point_in_polygon ──────────────────────────────────────────────────────────


def test_point_in_polygon_center_inside_square():
    square = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    assert point_in_polygon(0.5, 0.5, square) is True


def test_point_in_polygon_outside_square():
    square = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    assert point_in_polygon(0.05, 0.05, square) is False


def test_point_in_polygon_far_outside():
    square = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    assert point_in_polygon(0.0, 0.0, square) is False


def test_point_in_polygon_triangle_inside():
    triangle = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    assert point_in_polygon(0.5, 0.3, triangle) is True


def test_point_in_polygon_triangle_outside():
    triangle = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    assert point_in_polygon(0.9, 0.9, triangle) is False


# ── image_quadrant (fallback when no named polygon covers the detection) ───────


def test_image_quadrant_four_corners():
    assert image_quadrant(0.2, 0.2) == "upper-left"
    assert image_quadrant(0.8, 0.2) == "upper-right"
    assert image_quadrant(0.2, 0.8) == "lower-left"
    assert image_quadrant(0.8, 0.8) == "lower-right"


def test_image_quadrant_center_boundary_is_lower_right():
    # The 0.5 boundary is inclusive on the right/lower side (uses >= 0.5).
    assert image_quadrant(0.5, 0.5) == "lower-right"
    assert image_quadrant(0.49, 0.49) == "upper-left"


# ── format_approximate_location ───────────────────────────────────────────────


def test_format_approximate_location_name_and_coords():
    text = format_approximate_location("East Barn", 32.1234, 34.8765)
    assert "East Barn" in text
    assert "32.1234" in text and "34.8765" in text
    assert "Approximate" in text


def test_format_approximate_location_coords_only_no_name():
    # lat/lon present but no polygon name: coordinate-only branch, still flagged
    # approximate and never claiming a precise fix.
    text = format_approximate_location(None, 32.1234, 34.8765)
    assert "32.1234" in text and "34.8765" in text
    assert text.startswith("Approximate location:")
    assert "camera metadata" in text


def test_format_approximate_location_unknown_when_nothing_configured():
    text = format_approximate_location(None, None, None)
    assert "unknown" in text.lower()


# ── find_zone_for_detection ───────────────────────────────────────────────────


def test_find_zone_returns_zone_name():
    zone = {
        "zone_name": "East Barn",
        "alert_label": "East Barn",
        "enabled": True,
        "vertices_norm": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
    }
    assert find_zone_for_detection((0.5, 0.5), [zone]) == "East Barn"


def test_find_zone_returns_none_when_outside():
    zone = {
        "zone_name": "North Corner",
        "alert_label": "North Corner",
        "enabled": True,
        "vertices_norm": [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)],
    }
    assert find_zone_for_detection((0.8, 0.8), [zone]) is None


def test_find_zone_skips_disabled():
    zone = {
        "zone_name": "East Barn",
        "alert_label": "East Barn",
        "enabled": False,
        "vertices_norm": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
    }
    assert find_zone_for_detection((0.5, 0.5), [zone]) is None


def test_find_zone_first_match_wins():
    zone_a = {
        "zone_name": "Zone A",
        "alert_label": "Zone A",
        "enabled": True,
        "vertices_norm": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    }
    zone_b = {
        "zone_name": "Zone B",
        "alert_label": "Zone B",
        "enabled": True,
        "vertices_norm": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    }
    result = find_zone_for_detection((0.5, 0.5), [zone_a, zone_b])
    assert result == "Zone A"


def test_find_zone_empty_zones():
    assert find_zone_for_detection((0.5, 0.5), []) is None


# ── build_camera_mapping_config ───────────────────────────────────────────────


def test_build_camera_mapping_config_is_json_serializable():
    cam = {"camera_id": "cam_001", "latitude": 32.0, "longitude": 34.8}
    refs = [{"point_id": "p1", "map_lat": 32.0, "map_lon": 34.8, "enabled": True}]
    zones = [
        {
            "zone_id": "z1",
            "zone_name": "East Barn",
            "enabled": True,
            "vertices_norm": [(0.1, 0.1), (0.9, 0.9)],
        }
    ]
    config = build_camera_mapping_config(cam, refs, zones)
    dumped = json.dumps(config)
    assert "cam_001" in dumped
    assert "East Barn" in dumped


def test_build_camera_mapping_config_excludes_disabled_refs():
    cam = {"camera_id": "cam_001"}
    refs = [
        {"point_id": "p1", "enabled": True},
        {"point_id": "p2", "enabled": False},
    ]
    config = build_camera_mapping_config(cam, refs, [])
    assert len(config["reference_points"]) == 1
    assert config["reference_points"][0]["point_id"] == "p1"


def test_build_camera_mapping_config_excludes_disabled_zones():
    cam = {"camera_id": "cam_001"}
    zones = [
        {"zone_id": "z1", "zone_name": "Zone A", "enabled": True},
        {"zone_id": "z2", "zone_name": "Zone B", "enabled": False},
    ]
    config = build_camera_mapping_config(cam, [], zones)
    assert len(config["image_zones"]) == 1
    assert config["image_zones"][0]["zone_name"] == "Zone A"


def test_build_camera_mapping_config_returns_dict():
    config = build_camera_mapping_config({}, [], [])
    assert isinstance(config, dict)
    assert "camera" in config
    assert "reference_points" in config
    assert "image_zones" in config


def test_build_camera_mapping_config_rejects_non_serializable_content():
    # A zone carrying a non-JSON value (a set) must raise ValueError rather than
    # produce a config that would later fail to export/import.
    cam = {"camera_id": "cam_001"}
    zones = [{"zone_id": "z1", "enabled": True, "tags": {"east", "barn"}}]
    with pytest.raises(ValueError, match="not JSON-serializable"):
        build_camera_mapping_config(cam, [], zones)


# ── compute_homography / apply_homography ─────────────────────────────────────


def test_compute_homography_too_few_points():
    img = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    mp = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    assert compute_homography(img, mp) is None


def test_compute_homography_mismatched_lengths():
    img = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    mp = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    assert compute_homography(img, mp) is None


def test_compute_and_apply_homography_scaling():
    # Unit square -> square scaled by 2: a valid affine (and thus homography) map.
    img = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    mp = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    h = compute_homography(img, mp)
    assert h is not None
    # Corners map back to the destination corners.
    for (x, y), (u, v) in zip(img, mp):
        pu, pv = apply_homography(h, (x, y))
        assert pu == pytest.approx(u, abs=1e-6)
        assert pv == pytest.approx(v, abs=1e-6)
    # Center maps to (1, 1).
    cu, cv = apply_homography(h, (0.5, 0.5))
    assert cu == pytest.approx(1.0, abs=1e-6)
    assert cv == pytest.approx(1.0, abs=1e-6)


def test_compute_and_apply_homography_translation():
    img = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    mp = [(10.0, 20.0), (11.0, 20.0), (11.0, 21.0), (10.0, 21.0)]
    h = compute_homography(img, mp)
    assert h is not None
    pu, pv = apply_homography(h, (0.5, 0.5))
    assert pu == pytest.approx(10.5, abs=1e-6)
    assert pv == pytest.approx(20.5, abs=1e-6)


def test_apply_homography_returns_none_for_degenerate_denominator():
    # A matrix whose last row is all zeros projects every point to w=0
    # (the point maps "to infinity") -> the projection is undefined, not a
    # silently-wrong finite coordinate.
    degenerate = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    assert apply_homography(degenerate, (0.3, 0.7)) is None


# ── estimate_map_position ─────────────────────────────────────────────────────


def _square_reference_points():
    # image normalized corners -> map (lat, lon). Map is a 2x scaling in lon/lat.
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    maps = [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)]  # (lat, lon)
    pts = []
    for i, ((xn, yn), (lat, lon)) in enumerate(zip(corners, maps)):
        pts.append({
            "point_id": f"p{i}",
            "image_x_norm": xn, "image_y_norm": yn,
            "map_lat": lat, "map_lon": lon,
            "enabled": True,
        })
    return pts


def test_estimate_map_position_center():
    pts = _square_reference_points()
    result = estimate_map_position(pts, (0.5, 0.5))
    assert result is not None
    lat, lon = result
    assert lat == pytest.approx(1.0, abs=1e-6)
    assert lon == pytest.approx(1.0, abs=1e-6)


def test_estimate_map_position_too_few_points():
    pts = _square_reference_points()[:3]
    assert estimate_map_position(pts, (0.5, 0.5)) is None


def test_estimate_map_position_ignores_disabled():
    pts = _square_reference_points()
    pts[0]["enabled"] = False  # only 3 usable -> not enough
    assert estimate_map_position(pts, (0.5, 0.5)) is None


def test_estimate_map_position_skips_malformed_reference_points():
    # A reference point missing a required field is silently skipped, not crashed
    # on. Removing map_lat from one of four points leaves only 3 usable -> None.
    pts = _square_reference_points()
    del pts[0]["map_lat"]
    assert estimate_map_position(pts, (0.5, 0.5)) is None


def test_estimate_map_position_recovers_when_enough_points_remain_after_skip():
    # Five points, one malformed: the four valid (non-collinear) corners still
    # solve the homography, so the malformed point is skipped without losing the fix.
    pts = _square_reference_points()  # 4 valid corners -> center maps to (1, 1)
    pts.append({
        "point_id": "bad", "image_x_norm": "not-a-number", "image_y_norm": 0.5,
        "map_lat": 1.0, "map_lon": 1.0, "enabled": True,
    })
    result = estimate_map_position(pts, (0.5, 0.5))
    assert result is not None
    lat, lon = result
    assert lat == pytest.approx(1.0, abs=1e-6)
    assert lon == pytest.approx(1.0, abs=1e-6)


# ── zone_reference_point_norm ─────────────────────────────────────────────────


def test_zone_reference_point_norm_valid():
    zone = {"zone_ref_point_norm": [0.25, 0.75]}
    assert zone_reference_point_norm(zone) == pytest.approx((0.25, 0.75))


def test_zone_reference_point_norm_missing_or_none():
    assert zone_reference_point_norm({}) is None
    assert zone_reference_point_norm({"zone_ref_point_norm": None}) is None


def test_zone_reference_point_norm_out_of_range():
    assert zone_reference_point_norm({"zone_ref_point_norm": [1.5, 0.5]}) is None
    assert zone_reference_point_norm({"zone_ref_point_norm": [0.5, -0.1]}) is None


def test_zone_reference_point_norm_malformed():
    assert zone_reference_point_norm({"zone_ref_point_norm": [0.5]}) is None
    assert zone_reference_point_norm({"zone_ref_point_norm": "0.5,0.5"}) is None
    assert zone_reference_point_norm({"zone_ref_point_norm": ["a", "b"]}) is None


# ── validate_zone_reference_point ─────────────────────────────────────────────


SQUARE_NORM = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]


def test_validate_zone_ref_ok_inside_polygon():
    zone = {
        "zone_ref_point_px": [320, 240],
        "zone_ref_point_norm": [0.5, 0.5],
        "vertices_norm": SQUARE_NORM,
    }
    assert validate_zone_reference_point(zone, 640, 480) == []


def test_validate_zone_ref_missing_reports_not_set():
    errors = validate_zone_reference_point({"vertices_norm": SQUARE_NORM}, 640, 480)
    assert any("not set" in e for e in errors)


def test_validate_zone_ref_outside_image_bounds():
    zone = {
        "zone_ref_point_px": [700, 240],
        "zone_ref_point_norm": [0.5, 0.5],
        "vertices_norm": SQUARE_NORM,
    }
    errors = validate_zone_reference_point(zone, 640, 480)
    assert any("image bounds" in e for e in errors)


def test_validate_zone_ref_outside_polygon_reuses_point_in_polygon():
    zone = {
        "zone_ref_point_px": [32, 24],
        "zone_ref_point_norm": [0.05, 0.05],
        "vertices_norm": SQUARE_NORM,
    }
    errors = validate_zone_reference_point(zone, 640, 480)
    assert any("outside the zone polygon" in e for e in errors)
    # Consistency with the reused point_in_polygon helper.
    assert point_in_polygon(0.05, 0.05, SQUARE_NORM) is False


def test_validate_zone_ref_skips_polygon_check_under_3_vertices():
    zone = {
        "zone_ref_point_px": [32, 24],
        "zone_ref_point_norm": [0.05, 0.05],
        "vertices_norm": SQUARE_NORM[:2],
    }
    assert validate_zone_reference_point(zone, 640, 480) == []


def test_validate_zone_ref_invalid_norm_coordinates():
    zone = {"zone_ref_point_px": [320, 240], "zone_ref_point_norm": [2.0, 0.5]}
    errors = validate_zone_reference_point(zone, 640, 480)
    assert any("invalid" in e.lower() for e in errors)


# ── generate_zone_map_estimates ───────────────────────────────────────────────


def test_generate_zone_map_estimates_uses_ref_point_not_centroid():
    refs = _square_reference_points()  # lat = 2*y_norm, lon = 2*x_norm
    zone = {
        "zone_id": "z1", "zone_name": "Tree Area", "enabled": True,
        # The polygon centroid is (0.5, 0.5) → would project to (1.0, 1.0). The
        # reference point is deliberately elsewhere to prove it is what's projected.
        "vertices_norm": SQUARE_NORM,
        "zone_ref_point_norm": [0.25, 0.25],
    }
    estimates, skipped = generate_zone_map_estimates([zone], refs)
    assert skipped == 0
    assert len(estimates) == 1
    est = estimates[0]
    assert est["zone_id"] == "z1"
    assert est["zone_name"] == "Tree Area"
    assert est["projection_source"] == "zone_reference_point"
    assert est["zone_ref_point_norm"] == pytest.approx([0.25, 0.25])
    assert est["est_lat"] == pytest.approx(0.5, abs=1e-6)  # ref point, not centroid (1.0)
    assert est["est_lon"] == pytest.approx(0.5, abs=1e-6)
    assert "vertices_latlon" not in est


def test_generate_zone_map_estimates_skips_zone_without_ref_point():
    refs = _square_reference_points()
    zones = [
        {"zone_id": "z1", "zone_name": "A", "enabled": True, "vertices_norm": SQUARE_NORM},
        {"zone_id": "z2", "zone_name": "B", "enabled": True, "zone_ref_point_norm": [0.5, 0.5]},
    ]
    estimates, skipped = generate_zone_map_estimates(zones, refs)
    assert skipped == 1
    assert [e["zone_id"] for e in estimates] == ["z2"]


def test_generate_zone_map_estimates_ignores_disabled_zones():
    refs = _square_reference_points()
    zones = [{"zone_id": "z1", "zone_name": "A", "enabled": False,
              "zone_ref_point_norm": [0.5, 0.5]}]
    estimates, skipped = generate_zone_map_estimates(zones, refs)
    assert estimates == []
    assert skipped == 0


def test_generate_zone_map_estimates_too_few_reference_points():
    refs = _square_reference_points()[:3]
    zones = [{"zone_id": "z1", "zone_name": "A", "enabled": True,
              "zone_ref_point_norm": [0.5, 0.5]}]
    estimates, skipped = generate_zone_map_estimates(zones, refs)
    assert estimates == []
    assert skipped == 1


# ── polygon_centroid_norm ─────────────────────────────────────────────────────


def test_polygon_centroid_norm_square():
    verts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    cx, cy = polygon_centroid_norm(verts)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)


def test_polygon_centroid_norm_empty():
    assert polygon_centroid_norm([]) is None


# ── estimate_horizon_y_norm ───────────────────────────────────────────────────


def test_estimate_horizon_bright_sky_dark_ground():
    # Top half bright (sky), bottom half dark (ground) -> boundary at 0.5.
    rows = [200.0] * 50 + [50.0] * 50
    result = estimate_horizon_y_norm(rows)
    assert result is not None
    y_norm, confidence = result
    assert y_norm == pytest.approx(0.5, abs=0.02)
    assert confidence == pytest.approx(1.0, abs=1e-6)


def test_estimate_horizon_flat_image_returns_none():
    # No brightness variation -> no horizon.
    assert estimate_horizon_y_norm([120.0] * 100) is None


def test_estimate_horizon_too_few_rows():
    assert estimate_horizon_y_norm([10.0, 20.0]) is None


def test_estimate_horizon_gradual_ramp_rejected():
    # A smooth top-to-bottom brightness ramp has no sharp boundary: each step is
    # tiny relative to the total spread -> confidence below threshold -> None.
    rows = [200.0 - i for i in range(100)]
    assert estimate_horizon_y_norm(rows) is None


# ── downwind_bearing_deg / downwind_arrow_endpoint ────────────────────────────


def test_downwind_bearing_is_opposite_wind_from():
    assert downwind_bearing_deg(0.0) == pytest.approx(180.0)
    assert downwind_bearing_deg(90.0) == pytest.approx(270.0)
    assert downwind_bearing_deg(270.0) == pytest.approx(90.0)


def test_downwind_bearing_wraps_below_360():
    assert downwind_bearing_deg(350.0) == pytest.approx(170.0)


def test_downwind_arrow_endpoint_wind_from_north_points_south():
    lat, lon = downwind_arrow_endpoint(32.0, 34.0, wind_from_deg=0.0)
    assert lat < 32.0
    assert lon == pytest.approx(34.0, abs=1e-9)


def test_downwind_arrow_endpoint_wind_from_south_points_north():
    lat, lon = downwind_arrow_endpoint(32.0, 34.0, wind_from_deg=180.0)
    assert lat > 32.0
    assert lon == pytest.approx(34.0, abs=1e-9)

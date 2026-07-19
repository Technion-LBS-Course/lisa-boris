"""Unit tests for src/incident_agent.py (pure — no Streamlit, no ML, no network)."""

import pytest

import src.incident_agent as ia
import src.llm as llm
from src.weather import Weather
from src.incident_agent import (
    build_drafts,
    build_incident_context,
    build_incident_system_prompt,
    compass_label,
    contact_guidance,
    create_incident_alert,
    downwind_direction,
    draft_farm_worker_message,
    draft_owner_message,
    find_nearest_zones,
    format_initial_incident_message,
    incident_narrative,
    incident_reasoning,
    initial_incident_message,
    polish_message,
    recommend_actions,
    respond_to_operator,
    summarize_operational_context,
)

# A small operational-context sample (mirrors the shape of the committed file) —
# hermetic, so these tests never depend on the on-disk operational context file.
OP_CONTEXT = {
    "primary_site_context": {
        "name": "Thunder Valley Casino Resort",
        "operational_relevance": "Populated resort area — treat movement toward it as sensitive.",
    },
    "nearby_operational_landmarks": [
        {"name": "Lincoln Crossing", "sensitivity": "high",
         "recommended_action_hint": "notify the relevant local authority"},
        {"name": "Wetland / slough", "sensitivity": "low_to_medium",
         "recommended_action_hint": "lower priority unless moving toward structures"},
    ],
    "authorities_and_contacts": [
        {"name": "Emergency services", "contact": "911", "usage_rule": "immediate danger"},
        {"name": "Lincoln Fire Department", "contact": "916-645-4040"},
        {"name": "Farm owner / site operator", "contact": None, "source_status": "missing"},
    ],
}

CAMERA = {
    "camera_id": "giloCAM", "camera_name": "gilo", "site_id": "1", "customer_id": "1",
    "latitude": 31.7396, "longitude": 35.1883,
}

# A drawn, enabled, high-priority forest-edge zone covering the image centre,
# with an operator-set zone reference point (the map-reporting point).
CENTER_ZONE = {
    "zone_name": "East Grove", "alert_label": "East Grove", "zone_type": "forest_edge",
    "priority_label": "high", "priority": 9, "enabled": True,
    "vertices_norm": [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
    "zone_ref_point_px": [320, 240],
    "zone_ref_point_norm": [0.5, 0.5],
}

# The same zone without a zone reference point — matched by name, but it must
# never get an invented map point.
CENTER_ZONE_NO_REF = {
    k: v for k, v in CENTER_ZONE.items()
    if k not in ("zone_ref_point_px", "zone_ref_point_norm")
}

# Wind from the W (270°) -> downwind toward the E.
WEATHER = Weather(temperature_c=33, relative_humidity=25, wind_speed_kmh=24,
                  wind_direction_deg=270, source="Open-Meteo", is_live=True)


def _square_reference_points():
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    maps = [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)]  # (lat, lon)
    return [
        {"image_x_norm": xn, "image_y_norm": yn, "map_lat": lat, "map_lon": lon, "enabled": True}
        for (xn, yn), (lat, lon) in zip(corners, maps)
    ]


def _ctx(weather=WEATHER, zones=(CENTER_ZONE,), refs=None, cls="fire", centroid=(0.5, 0.5), prev=None):
    return build_incident_context(
        camera=CAMERA, image_zones=list(zones),
        reference_points=list(refs) if refs is not None else [],
        detected_class=cls, confidence=0.82, centroid_norm=centroid,
        prev_centroid_norm=prev, weather=weather,
        timestamp="2026-07-01T10:00:00+00:00",
    )


# ── compass / downwind ────────────────────────────────────────────────────────


def test_compass_and_downwind():
    assert compass_label(270) == "W"
    assert downwind_direction(270) == "E"  # wind from W blows toward E
    assert downwind_direction(0) == "S"


# ── build_incident_context ────────────────────────────────────────────────────


def test_context_matches_zone_type_and_priority():
    ctx = _ctx()
    assert ctx.matched_zone == "East Grove"
    assert ctx.zone_type == "forest_edge"
    assert ctx.zone_priority_label == "high"
    assert "East Grove" in ctx.location_text
    assert "mapped zone" in ctx.location_text


def test_context_quadrant_fallback_when_no_zone():
    ctx = _ctx(zones=(), centroid=(0.1, 0.1))
    assert ctx.matched_zone is None
    assert "upper-left" in ctx.location_text


def test_context_estimated_map_point_no_geolocation_disclaimer():
    ctx = _ctx(refs=_square_reference_points())
    assert ctx.approximate_lat == pytest.approx(1.0, abs=1e-6)
    assert "approximate map point from zone reference point" in ctx.location_text
    assert "not precise geolocation" not in ctx.location_text.lower()
    assert "image-space" not in ctx.location_text.lower()


def test_context_uses_matched_zone_reference_point_not_detection_anchor():
    # Zone ref point (0.25, 0.25) projects to (0.5, 0.5); the detection point
    # (0.5, 0.5) would project to (1.0, 1.0). The zone reference point must win.
    zone = {**CENTER_ZONE, "zone_ref_point_norm": [0.25, 0.25], "zone_ref_point_px": [160, 120]}
    ctx = _ctx(zones=(zone,), refs=_square_reference_points(), centroid=(0.5, 0.5))
    assert ctx.matched_zone == "East Grove"
    assert ctx.approximate_lat == pytest.approx(0.5, abs=1e-6)
    assert ctx.approximate_lon == pytest.approx(0.5, abs=1e-6)
    assert ctx.map_point_source == "zone_reference_point"
    assert "approximate map point from zone reference point" in ctx.location_text


def test_context_matched_zone_without_ref_point_has_no_map_point():
    ctx = _ctx(zones=(CENTER_ZONE_NO_REF,), refs=_square_reference_points())
    assert ctx.matched_zone == "East Grove"
    # No invented map point — not even a detection-anchor projection.
    assert ctx.approximate_lat is None and ctx.approximate_lon is None
    assert ctx.map_point_source is None
    assert "zone reference point not set" in ctx.location_text
    rows = dict(ctx.display_rows())
    assert "reference point" in rows.get("Estimated map point", "")


def test_context_detection_anchor_fallback_when_no_zone_matched():
    ctx = _ctx(zones=(), refs=_square_reference_points(), centroid=(0.5, 0.5))
    assert ctx.matched_zone is None
    assert ctx.approximate_lat == pytest.approx(1.0, abs=1e-6)
    assert ctx.map_point_source == "detection_anchor"
    assert "estimated location" in ctx.location_text.lower()


def test_context_quadrant_fallback_when_no_zone_and_no_refs():
    ctx = _ctx(zones=(), refs=[], centroid=(0.9, 0.9))
    assert ctx.approximate_lat is None
    assert ctx.map_point_source is None
    assert "lower-right" in ctx.location_text
    assert "camera-frame location" in ctx.location_text


def test_context_matched_zone_with_ref_point_but_too_few_refs():
    # Ref point exists but the homography needs >= 4 reference points — the zone
    # name is kept and no map point is invented.
    ctx = _ctx(refs=_square_reference_points()[:3])
    assert ctx.matched_zone == "East Grove"
    assert ctx.approximate_lat is None
    assert ctx.map_point_source is None
    assert "camera-frame location" in ctx.location_text


def test_context_weather_fields_and_downwind():
    ctx = _ctx()
    assert ctx.temperature_c == 33
    assert ctx.relative_humidity == 25
    assert ctx.wind_speed_kmh == 24
    assert ctx.wind_compass == "W"
    assert ctx.downwind_risk_direction == "E"
    assert ctx.weather_is_live is True
    assert ctx.weather_source == "Open-Meteo"


def test_context_downwind_from_dict_weather():
    ctx = _ctx(weather={"wind_direction_deg": 0})
    assert ctx.downwind_risk_direction == "S"
    assert ctx.wind_compass == "N"


def test_context_missing_weather_does_not_crash():
    ctx = _ctx(weather=None)
    assert ctx.wind_compass is None and ctx.downwind_risk_direction is None
    assert ctx.temperature_c is None
    # recommendations must still work without weather.
    assert recommend_actions(ctx)


def test_context_image_plane_direction():
    ctx = _ctx(centroid=(0.7, 0.5), prev=(0.5, 0.5))
    assert ctx.image_plane_direction == "right"


# ── recommend_actions ─────────────────────────────────────────────────────────


def test_recommendations_fire_high_priority():
    recs = recommend_actions(_ctx())
    text = " ".join(recs).lower()
    assert "on-site verification" in text
    assert "fire department" in text
    assert any("never contacts emergency services" in r.lower() for r in recs)


def test_recommendations_include_wind_when_present():
    recs = " ".join(recommend_actions(_ctx()))
    assert "Wind is from the W" in recs
    assert "toward the E" in recs


def test_recommendations_smoke_verifies_source():
    recs = recommend_actions(_ctx(cls="smoke"))
    assert any("non-fire source" in r for r in recs)
    assert any("dispatches automatically" in r.lower() for r in recs)


def test_recommendations_hot_dry_and_shareable_location():
    # WEATHER is 33°C / RH 25% (temp >= 30 and RH <= 30 → hot/dry line), and the
    # reference points project a map point → the shareable-location line appears.
    recs = recommend_actions(_ctx(refs=_square_reference_points()))
    assert "Hot / dry conditions raise risk — keep water and suppression tools ready." in recs
    assert "An estimated location is available to share with responders." in recs


def test_recommendations_no_shareable_location_without_map_point():
    # No refs → no projected map point → the shareable-location line is absent.
    recs = recommend_actions(_ctx(zones=(), centroid=(0.1, 0.1)))
    assert not any("estimated location is available to share" in r for r in recs)


# ── conversation ──────────────────────────────────────────────────────────────


# ── find_nearest_zones ────────────────────────────────────────────────────────


def test_find_nearest_zones_contains_sorts_first():
    zones = find_nearest_zones((0.5, 0.5), [CENTER_ZONE])
    assert zones and zones[0]["zone_name"] == "East Grove"
    assert zones[0]["contains"] is True
    assert zones[0]["distance_norm"] == 0.0
    assert zones[0]["priority_label"] == "high"


def test_find_nearest_zones_outside_reports_distance():
    zones = find_nearest_zones((0.0, 0.0), [CENTER_ZONE])
    assert zones and zones[0]["contains"] is False
    assert zones[0]["distance_norm"] > 0.0


def test_context_carries_nearest_zones():
    ctx = _ctx(zones=(CENTER_ZONE,), centroid=(0.1, 0.1))
    assert ctx.matched_zone is None
    assert any(z["zone_name"] == "East Grove" for z in ctx.nearest_zones)


# ── initial incident message (concise, structured-context-driven) ─────────────


def test_initial_message_is_concise_and_omits_telemetry():
    ctx = _ctx()  # fire, matched high-priority East Grove, wind from W -> downwind E
    msg = format_initial_incident_message(ctx)
    assert "gilo" in msg                         # camera name
    assert "East Grove" in msg                   # operational 'where'
    assert "fire" in msg.lower()
    assert "drifting east" in msg                # downwind expanded to a word
    assert msg.strip().endswith("?")             # one next-action question
    # No raw telemetry leaks into the opener.
    assert "82%" not in msg and "confidence" not in msg.lower()
    assert "°C" not in msg and "33" not in msg


def test_initial_message_outside_zones_uses_nearest_zone():
    ctx = _ctx(zones=(CENTER_ZONE,), centroid=(0.1, 0.1))
    msg = format_initial_incident_message(ctx)
    assert "outside the marked zones" in msg
    assert "East Grove" in msg


def test_initial_message_smoke_unknown_area_suggests_monitoring():
    # Smoke with no matched zone and no nearby zone (unknown area, no refs) →
    # known_area is False → the opener offers to keep monitoring rather than to
    # prepare a response update. (With a matched zone the code instead asks
    # "Should I prepare a response update?".)
    ctx = _ctx(zones=(), refs=[], cls="smoke", centroid=(0.9, 0.9), weather=None)
    msg = format_initial_incident_message(ctx)
    assert "smoke" in msg.lower()
    assert "Should I keep monitoring, or prepare a notification?" in msg
    assert msg.strip().endswith("?")


def test_initial_message_no_zone_no_refs_uses_quadrant_no_contact_invented():
    ctx = _ctx(zones=(), refs=[], cls="smoke", centroid=(0.9, 0.9), weather=None)
    msg = format_initial_incident_message(ctx)
    assert "frame" in msg.lower()          # quadrant fallback, operational terms
    # never invents a specific contact — only generic wording is allowed
    assert "sheriff" not in msg.lower()


def test_initial_incident_message_deterministic_without_groq(monkeypatch):
    monkeypatch.setattr(ia, "_groq_ready", lambda: False)
    ctx = _ctx()
    assert initial_incident_message(ctx) == format_initial_incident_message(ctx)


def test_initial_incident_message_polished_by_groq(monkeypatch):
    monkeypatch.setattr(ia, "_groq_ready", lambda: True)
    monkeypatch.setattr(llm, "ask", lambda prompt, *a, **k: "Short polished line?")
    assert initial_incident_message(_ctx()) == "Short polished line?"


# ── incident_reasoning (detailed, only shown when the operator asks) ──────────


def test_incident_reasoning_has_context_and_recommendations():
    text = incident_reasoning(_ctx())
    assert "East Grove" in text
    assert "Recommended actions:" in text
    # Detailed reasoning MAY surface telemetry rows the opener omits.
    assert "confidence" in text.lower()


def test_deterministic_reply_why_returns_reasoning(monkeypatch):
    monkeypatch.setattr(ia, "_groq_ready", lambda: False)
    reply = respond_to_operator(_ctx(), "why did you flag this?")
    assert "how i read this incident" in reply.lower()
    assert "Recommended actions:" in reply


# ── operational context (landmarks / receptors / contact policy) ──────────────


def _ctx_with_context(operational_context=OP_CONTEXT, operational_context_md=None,
                      cls="smoke", zones=(CENTER_ZONE,), centroid=(0.5, 0.5), weather=WEATHER):
    return build_incident_context(
        camera=CAMERA, image_zones=list(zones), reference_points=[],
        detected_class=cls, confidence=0.5, centroid_norm=centroid, weather=weather,
        timestamp="2026-07-01T10:00:00+00:00",
        operational_context=operational_context, operational_context_md=operational_context_md,
    )


def test_summarize_operational_context_has_landmarks_and_policy():
    brief = summarize_operational_context(OP_CONTEXT)
    assert "Lincoln Crossing" in brief
    assert "Thunder Valley Casino Resort" in brief
    assert "Contact policy" in brief
    assert "911" in brief and "916-645-4040" in brief  # verified contacts included by default


def test_summarize_operational_context_excludes_contacts_when_asked():
    brief = summarize_operational_context(OP_CONTEXT, include_contacts=False)
    assert "Lincoln Crossing" in brief
    assert "911" not in brief and "916-645-4040" not in brief  # no phone leak into the opener
    assert "Contact policy" in brief


def test_summarize_operational_context_md_fallback_and_empty():
    md = summarize_operational_context(None, "# MD context\nLincoln Crossing area")
    assert md.strip().startswith("# MD context")
    assert summarize_operational_context(None, None) == ""
    assert summarize_operational_context({}, None) == ""


def test_context_stores_operational_context():
    ctx = _ctx_with_context(operational_context_md="# md")
    assert ctx.operational_context is OP_CONTEXT
    assert ctx.operational_context_md == "# md"


def test_context_without_operational_context_is_none():
    ctx = _ctx()  # no operational context supplied
    assert ctx.operational_context is None
    assert ctx.operational_context_md is None
    # Missing context must not break the concise opener.
    assert format_initial_incident_message(ctx).strip().endswith("?")


def test_initial_message_passes_operational_context_to_llm(monkeypatch):
    monkeypatch.setattr(ia, "_groq_ready", lambda: True)
    captured = {}

    def fake_ask(prompt, *a, **k):
        captured["prompt"] = prompt
        return "ThunderValleyWest detected smoke near Lincoln Crossing. Notify the relevant local authority?"

    monkeypatch.setattr(llm, "ask", fake_ask)
    out = initial_incident_message(_ctx_with_context())
    assert out.startswith("ThunderValleyWest")
    # Place names reach the LLM prompt, but verified phone numbers do not (opener brief).
    assert "Lincoln Crossing" in captured["prompt"]
    assert "916-645-4040" not in captured["prompt"] and "911" not in captured["prompt"]


def test_first_message_never_contains_phone_number():
    import re

    msg = format_initial_incident_message(_ctx_with_context(cls="fire"))
    assert not re.search(r"\d{3}[-.\s]?\d{3,4}", msg)  # no phone-like sequence
    assert "916-645-4040" not in msg and "911" not in msg


def test_system_prompt_includes_operational_context_and_contact_rule():
    prompt = build_incident_system_prompt(_ctx_with_context())
    assert "Lincoln Crossing" in prompt
    assert "never invent a contact" in prompt.lower()


def test_incident_reasoning_appends_operational_context():
    text = incident_reasoning(_ctx_with_context())
    assert "Operational context:" in text
    assert "Lincoln Crossing" in text


def test_contact_guidance_lists_verified_contacts():
    text = contact_guidance(_ctx_with_context())
    assert "911" in text and "916-645-4040" in text
    assert "never contacts anyone automatically" in text.lower()


def test_contact_guidance_without_context_offers_search_no_invention():
    text = contact_guidance(_ctx())
    assert "relevant local authority" in text.lower()
    assert "search" in text.lower()
    assert "916-645-4040" not in text  # nothing invented


def test_preferred_contact_prefers_fire_department_over_emergency():
    # OP_CONTEXT has BOTH Emergency services (911) and Lincoln Fire Department
    # (916-645-4040). Default prefer=("fire", "emergency") must rank fire first.
    ctx = _ctx_with_context()
    best = ia._preferred_contact(ctx)
    assert best is not None
    assert best["name"] == "Lincoln Fire Department"
    assert best["contact"] == "916-645-4040"   # the fire-dept number, not 911
    # The fire-department summary surfaces that same preferred number, never 911.
    summary = ia.prepare_fire_department_summary(ctx)
    assert "Suggested contact: Lincoln Fire Department (916-645-4040)" in summary
    assert "911" not in summary


def test_contact_clause_names_verified_phone_then_offers_search():
    # With a verified contact, the clause names the authority + its phone number.
    assert ia.contact_clause(_ctx_with_context()) == "Lincoln Fire Department (916-645-4040)"
    # Without operational context it stays generic and offers to search the web —
    # and never claims to contact anyone automatically.
    fallback = ia.contact_clause(_ctx())
    assert "relevant local authority" in fallback.lower()
    assert "search the web" in fallback.lower()
    assert "916-645-4040" not in fallback          # nothing invented
    assert "automatic" not in fallback.lower()      # never claims auto-contact


def test_deterministic_reply_contact_question_uses_verified_contacts(monkeypatch):
    monkeypatch.setattr(ia, "_groq_ready", lambda: False)
    reply = respond_to_operator(_ctx_with_context(), "who do i call?")
    assert "911" in reply


def test_incident_narrative_mentions_zone_and_wind():
    text = incident_narrative(_ctx())
    assert "East Grove" in text
    assert "Wind is from the W" in text
    assert text.strip().endswith("What would you like to do?")


def test_respond_to_operator_deterministic_worker_when_no_groq(monkeypatch):
    monkeypatch.setattr(ia, "_groq_ready", lambda: False)
    ctx = _ctx()
    reply = respond_to_operator(ctx, "Update the farm workers near East Grove")
    assert reply == draft_farm_worker_message(ctx)
    assert "East Grove" in reply and "Report whether flames are visible" in reply


def test_respond_to_operator_default_offers_menu_when_no_groq(monkeypatch):
    monkeypatch.setattr(ia, "_groq_ready", lambda: False)
    reply = respond_to_operator(_ctx(), "hi")
    assert "farm workers" in reply.lower() and "owner" in reply.lower()


def test_respond_to_operator_uses_llm_when_available(monkeypatch):
    monkeypatch.setattr(ia, "_groq_ready", lambda: True)
    captured = {}

    def fake_chat(messages, **kw):
        captured["messages"] = messages
        return "LLM operational reply."

    monkeypatch.setattr(llm, "chat", fake_chat)
    reply = respond_to_operator(
        _ctx(), "what should I do?", history=[{"role": "assistant", "content": "opening"}]
    )
    assert reply == "LLM operational reply."
    # System prompt carries the incident facts; history + new user message are passed through.
    assert captured["messages"][0]["role"] == "system"
    assert "East Grove" in captured["messages"][0]["content"]
    assert {"role": "assistant", "content": "opening"} in captured["messages"]
    assert captured["messages"][-1] == {"role": "user", "content": "what should I do?"}


def test_respond_to_operator_falls_back_when_llm_errors(monkeypatch):
    monkeypatch.setattr(ia, "_groq_ready", lambda: True)

    def boom(messages, **kw):
        raise RuntimeError("groq down")

    monkeypatch.setattr(llm, "chat", boom)
    ctx = _ctx()
    assert respond_to_operator(ctx, "update the farm workers") == draft_farm_worker_message(ctx)


def test_incident_system_prompt_has_facts_and_guardrails():
    prompt = build_incident_system_prompt(_ctx())
    low = prompt.lower()
    assert "East Grove" in prompt
    assert "cannot send" in low
    assert "fire spread" in low
    # Relevance rules: confirmed (no confidence value), camera by name, coords only
    # for field responders, contacts never invented.
    assert "confirmed" in low
    assert "do not state a confidence value" in low
    assert "coordinates" in low and "field responders" in low
    assert "never invent" in low


def test_incident_system_prompt_omits_camera_id_and_telemetry():
    ctx = _ctx()  # camera_id giloCAM, weather with temp/RH/wind speed
    prompt = build_incident_system_prompt(ctx)
    assert "giloCAM" not in prompt            # camera by NAME only, never the ID
    assert "°C" not in prompt and "RH " not in prompt and "km/h" not in prompt
    assert "gilo" in prompt                    # the camera name is present


# ── drafts: worker (no coords) / owner (weather) / neighbor / fire-dept ────────


def test_worker_draft_has_zone_task_and_no_coordinates():
    ctx = _ctx(refs=_square_reference_points())  # map point exists, but must not leak into worker text
    draft = draft_farm_worker_message(ctx)
    assert draft.startswith("East Grove — high priority:")
    assert "tree line" in draft  # forest_edge task
    assert "Avoid entering the smoky area." in draft
    assert "Report whether flames are visible." in draft
    assert "~" not in draft  # no coordinates
    assert "31." not in draft and "35." not in draft  # no lat/lon leakage


def test_owner_draft_is_relevant_and_omits_telemetry():
    # Owner response update: camera NAME + place + direction + who to notify.
    # No camera ID, no confidence, no coordinates, no raw weather telemetry.
    draft = draft_owner_message(_ctx(refs=_square_reference_points()))
    assert "gilo" in draft and "giloCAM" not in draft
    assert "East Grove" in draft
    assert "confidence" not in draft.lower()
    assert "°C" not in draft and "km/h" not in draft and "RH " not in draft
    assert "~" not in draft  # no coordinates for the owner


def test_owner_draft_uses_verified_contact_phone_when_available():
    ctx = _ctx_with_context()  # OP_CONTEXT has 911 / Lincoln Fire 916-645-4040
    draft = draft_owner_message(ctx)
    assert "916-645-4040" in draft or "911" in draft


def test_owner_draft_offers_search_when_no_verified_contact():
    draft = draft_owner_message(_ctx())  # no operational context
    assert "search the web" in draft.lower()
    assert "916-645-4040" not in draft  # nothing invented


def test_owner_draft_without_weather_does_not_crash():
    draft = draft_owner_message(_ctx(weather=None))
    assert "gilo" in draft


def test_coordinates_only_in_fire_department_draft():
    # Approximate coordinates go ONLY to field responders; never to the owner,
    # neighbour, or worker.
    drafts = build_drafts(_ctx(refs=_square_reference_points()))
    assert "~" in drafts["Fire department summary"]
    assert "~" not in drafts["Property owner"]
    assert "~" not in drafts["Neighbor"]
    assert "~" not in drafts["Farm worker"]


def test_deterministic_affirmative_prepares_owner_update(monkeypatch):
    # "yes" to the opener's response-update question → concise owner update, not a
    # telemetry dump, with a verified contact phone when one is on file.
    monkeypatch.setattr(ia, "_groq_ready", lambda: False)
    reply = respond_to_operator(_ctx_with_context(), "yes")
    assert "confidence" not in reply.lower()
    assert "916-645-4040" in reply or "911" in reply


def test_no_geolocation_disclaimer_in_any_draft():
    drafts = build_drafts(_ctx(refs=_square_reference_points()))
    for text in drafts.values():
        low = text.lower()
        assert "image-space" not in low
        assert "not precise geolocation" not in low
        assert "early warning" not in low


def test_firedept_summary_includes_coords_when_available():
    drafts = build_drafts(_ctx(refs=_square_reference_points()))
    fd = drafts["Fire department summary"]
    assert "coordinates" in fd.lower()   # responders need coordinates
    assert "never contacts emergency services automatically" in fd
    # No confidence / camera-ID / raw telemetry even in the responder summary.
    assert "confidence" not in fd.lower()
    assert "giloCAM" not in fd
    assert "km/h" not in fd and "°C" not in fd


def test_fire_department_summary_coordinate_line_exact_format():
    # The fire department is the one audience allowed approximate coordinates; the
    # zone reference point projects to ~1.00000, 1.00000 with the square refs.
    ctx = _ctx(refs=_square_reference_points())
    summary = ia.prepare_fire_department_summary(ctx)
    assert summary.startswith("Fire-department summary")
    assert "- Approximate coordinates: ~1.00000, 1.00000 (confirm on arrival)" in summary


def test_neighbor_draft_is_short_without_coords():
    draft = build_drafts(_ctx())["Neighbor"]
    assert "~" not in draft
    assert "East Grove" in draft


# ── alert record ──────────────────────────────────────────────────────────────


def test_create_incident_alert_valid():
    alert = create_incident_alert(_ctx(refs=_square_reference_points()), status="confirmed")
    assert alert["status"] == "confirmed"
    assert alert["detected_class"] == "fire"
    assert alert["image_polygon_name"] == "East Grove"
    assert alert["approximate_lat"] == pytest.approx(1.0, abs=1e-6)
    assert alert["geographic_bearing"] is None
    assert alert["camera_id"] == "giloCAM"


def test_create_incident_alert_rejects_non_fire_smoke_class():
    ctx = _ctx(zones=(), cls="person")
    with pytest.raises(ValueError):
        create_incident_alert(ctx)


# ── polish_message (optional Groq wording refine) ─────────────────────────────


def test_polish_message_uses_llm_when_available(monkeypatch):
    monkeypatch.setattr(llm, "ask", lambda prompt, *a, **k: "REFINED TEXT")
    assert polish_message("original", "owner") == "REFINED TEXT"


def test_polish_message_falls_back_on_error(monkeypatch):
    def boom(prompt, *a, **k):
        raise RuntimeError("no key")

    monkeypatch.setattr(llm, "ask", boom)
    assert polish_message("original text", "owner") == "original text"

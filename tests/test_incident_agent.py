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
    create_incident_alert,
    downwind_direction,
    draft_farm_worker_message,
    draft_owner_message,
    incident_narrative,
    polish_message,
    recommend_actions,
    respond_to_operator,
)

CAMERA = {
    "camera_id": "giloCAM", "camera_name": "gilo", "site_id": "1", "customer_id": "1",
    "latitude": 31.7396, "longitude": 35.1883,
}

# A drawn, enabled, high-priority forest-edge zone covering the image centre.
CENTER_ZONE = {
    "zone_name": "East Grove", "alert_label": "East Grove", "zone_type": "forest_edge",
    "priority_label": "high", "priority": 9, "enabled": True,
    "vertices_norm": [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
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
    assert "estimated location" in ctx.location_text.lower()
    assert "not precise geolocation" not in ctx.location_text.lower()
    assert "image-space" not in ctx.location_text.lower()


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


# ── conversation ──────────────────────────────────────────────────────────────


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
    assert "East Grove" in prompt
    assert "confidence" in prompt.lower()
    assert "gps" in prompt.lower()  # worker no-coordinates rule
    assert "cannot send" in prompt.lower()
    assert "fire spread" in prompt.lower()


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


def test_owner_draft_includes_weather_and_camera():
    draft = draft_owner_message(_ctx())
    assert "giloCAM" in draft
    assert "Wind from the W" in draft
    assert "km/h" in draft
    assert "°C" in draft


def test_owner_draft_without_weather_does_not_crash():
    draft = draft_owner_message(_ctx(weather=None))
    assert "giloCAM" in draft


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
    assert "Estimated coordinates" in fd
    assert "does not contact emergency services automatically" in fd


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

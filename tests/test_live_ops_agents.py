"""Unit tests for src/live_ops_agents — dual-agent ops-chat composition.

Pure: no network, no ML. The Response reply is exercised via a monkeypatched
incident_agent so no Groq/network call happens.
"""
from src import incident_agent, live_ops_agents as agents, llm, weather


def _ctx():
    return incident_agent.build_incident_context(
        camera={"camera_id": "CAM-04", "camera_name": "Ridge East"},
        image_zones=[],
        reference_points=[],
        detected_class="smoke",
        confidence=0.87,
        centroid_norm=(0.5, 0.6),
        weather=None,
    )


def test_agent_identities():
    # AGENT_ICON must have an icon for exactly the two agent identities — a missing
    # key (or an extra one) would be caught by this set comparison.
    assert set(agents.AGENT_ICON) == {agents.WATCH, agents.RESPONSE}
    assert all(agents.AGENT_ICON[key] for key in agents.AGENT_ICON)  # every icon non-empty


def test_routine_status_text_from_weather_and_advisory():
    wx = weather.Weather(temperature_c=34.0, relative_humidity=18.0, wind_speed_kmh=25.0,
                         wind_direction_deg=270.0, source="Mock weather", is_live=False)
    adv = weather.assess_risk(wx, [])
    text = agents.routine_status_text(wx, adv)
    assert "risk" in text.lower()
    assert "°C" in text
    assert "fallback" in text.lower()  # is_live=False


def test_routine_status_text_handles_missing_weather():
    # No weather and no advisory → the exact no-data status line.
    assert agents.routine_status_text(None, None) == "No weather available for a risk status right now."


def test_routine_status_text_includes_downwind_line():
    # When the advisory carries a downwind direction, the Watch status surfaces it.
    wx = weather.Weather(temperature_c=34.0, relative_humidity=18.0, wind_speed_kmh=25.0,
                         wind_direction_deg=270.0, source="Mock weather", is_live=False)
    adv = weather.assess_risk(wx, [])
    assert adv.downwind == "E"  # wind from W (270°) blows toward the E
    text = agents.routine_status_text(wx, adv)
    assert "Downwind risk toward E." in text


def test_emergency_open_text_deterministic_fallback(monkeypatch):
    # With Groq unavailable, the opener is the deterministic concise incident line:
    # short, action-oriented, and free of raw telemetry (confidence/temperature).
    monkeypatch.setattr(incident_agent, "_groq_ready", lambda: False)
    text = agents.emergency_open_text(_ctx())
    assert "smoke" in text.lower()
    assert text.strip().endswith("?")          # ends with a single next-action question
    assert "confidence" not in text.lower()    # no raw telemetry in the opener
    assert "87" not in text and "°c" not in text.lower()


def test_emergency_open_text_uses_groq_when_available(monkeypatch):
    # With Groq available, the deterministic line is rephrased by the LLM (no network here).
    monkeypatch.setattr(incident_agent, "_groq_ready", lambda: True)
    monkeypatch.setattr(llm, "ask", lambda prompt, *a, **k: "GROQ-WRITTEN ALERT")
    assert agents.emergency_open_text(_ctx()) == "GROQ-WRITTEN ALERT"


def test_notification_drafts_cover_audiences():
    drafts = agents.notification_drafts(_ctx())
    assert set(drafts) >= {"Property owner", "Fire department summary"}
    # Worker draft never leaks coordinates (incident_agent guardrail).
    assert "lat" not in drafts["Farm worker"].lower()


def test_agent_reply_delegates(monkeypatch):
    captured = {}

    def fake_respond(ctx, message, history=None):
        captured["message"] = message
        captured["history"] = history
        return "canned reply"

    monkeypatch.setattr(incident_agent, "respond_to_operator", fake_respond)
    out = agents.agent_reply(_ctx(), "who should I call?", [{"role": "user", "content": "hi"}])
    assert out == "canned reply"
    assert captured["message"] == "who should I call?"


def test_uses_llm_delegates_to_incident_agent(monkeypatch):
    # uses_llm() must reflect incident_agent.conversation_uses_llm(), not a constant.
    monkeypatch.setattr(incident_agent, "conversation_uses_llm", lambda: True)
    assert agents.uses_llm() is True
    monkeypatch.setattr(incident_agent, "conversation_uses_llm", lambda: False)
    assert agents.uses_llm() is False

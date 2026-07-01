"""Unit tests for src/weather.py — Open-Meteo weather + fire-weather risk logic.

Pure and offline: the one live-path function (fetch_open_meteo_weather) is
monkeypatched, so no network call is ever made. No API key is required anywhere.
"""

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import src.weather as weather
from src.weather import (
    OPEN_METEO_URL,
    OpenMeteoClient,
    Weather,
    assess_risk,
    build_open_meteo_url,
    fetch_weather,
    fire_weather_risk,
    mock_weather,
    parse_open_meteo_current_weather,
)

HOT_DRY_WINDY = Weather(temperature_c=37, relative_humidity=14, wind_speed_kmh=40, wind_direction_deg=0)
MILD = Weather(temperature_c=20, relative_humidity=70, wind_speed_kmh=3, wind_direction_deg=180)

HIGH_BARN = {"zone_name": "Hay Storage", "zone_type": "barn", "priority_label": "high", "enabled": True}
HIGH_GROVE = {"zone_name": "East Grove", "zone_type": "forest_edge", "priority_label": "high", "enabled": True}
LOW_FIELD = {"zone_name": "North Field", "zone_type": "field", "priority_label": "low", "enabled": True}

SAMPLE_PAYLOAD = {
    "current": {
        "temperature_2m": 30.5,
        "relative_humidity_2m": 35,
        "wind_speed_10m": 18.2,
        "wind_direction_10m": 270,
    }
}


# ── build_open_meteo_url ──────────────────────────────────────────────────────


def test_build_url_has_expected_params():
    url = build_open_meteo_url(31.7396, 35.1883)
    assert url.startswith(OPEN_METEO_URL + "?")
    q = parse_qs(urlparse(url).query)
    assert q["latitude"] == ["31.7396"]
    assert q["longitude"] == ["35.1883"]
    assert q["current"] == ["temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"]
    assert q["timezone"] == ["auto"]
    assert "appid" not in q and "apikey" not in q  # no API key in the request


def test_build_url_rejects_out_of_range_coords():
    with pytest.raises(ValueError):
        build_open_meteo_url(91, 0)
    with pytest.raises(ValueError):
        build_open_meteo_url(0, 181)
    with pytest.raises(ValueError):
        build_open_meteo_url(None, None)


# ── parse_open_meteo_current_weather ──────────────────────────────────────────


def test_parse_full_payload():
    wx = parse_open_meteo_current_weather(SAMPLE_PAYLOAD)
    assert wx.is_live is True
    assert wx.source == "Open-Meteo"
    assert wx.temperature_c == 30.5
    assert wx.relative_humidity == 35.0
    assert wx.wind_speed_kmh == 18.2
    assert wx.wind_direction_deg == 270.0
    assert set(wx.as_dict()) == {
        "temperature_c", "relative_humidity", "wind_speed_kmh",
        "wind_direction_deg", "source", "is_live",
    }


def test_parse_missing_current_raises():
    with pytest.raises(ValueError):
        parse_open_meteo_current_weather({})
    with pytest.raises(ValueError):
        parse_open_meteo_current_weather({"current": {}})  # no temperature_2m


def test_parse_partial_fields_are_none():
    wx = parse_open_meteo_current_weather({"current": {"temperature_2m": 22}})
    assert wx.temperature_c == 22.0
    assert wx.relative_humidity is None
    assert wx.wind_speed_kmh is None
    assert wx.is_live is True


# ── fetch_weather: live success / fallbacks ───────────────────────────────────


def test_fetch_weather_live_success(monkeypatch):
    live = Weather(temperature_c=30, relative_humidity=35, wind_speed_kmh=18,
                   wind_direction_deg=270, source="Open-Meteo", is_live=True)
    monkeypatch.setattr(weather, "fetch_open_meteo_weather", lambda lat, lon, timeout=6.0: live)
    wx = fetch_weather(31.7396, 35.1883)
    assert wx.is_live is True
    assert wx.source == "Open-Meteo"


def test_fetch_weather_network_failure_falls_back_to_mock(monkeypatch):
    def boom(lat, lon, timeout=6.0):
        raise RuntimeError("network down")

    monkeypatch.setattr(weather, "fetch_open_meteo_weather", boom)
    wx = fetch_weather(31.7396, 35.1883)
    assert wx.is_live is False
    assert wx.source == "Mock weather"


def test_fetch_weather_missing_fields_falls_back_to_mock(monkeypatch):
    def bad_fields(lat, lon, timeout=6.0):
        # Represents parse_open_meteo_current_weather raising on missing fields.
        raise ValueError("Open-Meteo response is missing 'current.temperature_2m'")

    monkeypatch.setattr(weather, "fetch_open_meteo_weather", bad_fields)
    wx = fetch_weather(31.7396, 35.1883)
    assert wx.is_live is False and wx.source == "Mock weather"


def test_fetch_weather_invalid_coords_uses_mock_without_network(monkeypatch):
    # If the live path were reached it would fail the test; invalid coords must
    # short-circuit to the mock before any network call.
    def must_not_be_called(lat, lon, timeout=6.0):
        raise AssertionError("live fetch should not run for invalid coordinates")

    monkeypatch.setattr(weather, "fetch_open_meteo_weather", must_not_be_called)
    wx = fetch_weather(999, 0)
    assert wx.is_live is False and wx.source == "Mock weather"


def test_openmeteo_client_delegates(monkeypatch):
    live = Weather(temperature_c=25, source="Open-Meteo", is_live=True)
    monkeypatch.setattr(weather, "fetch_open_meteo_weather", lambda lat, lon, timeout=6.0: live)
    assert OpenMeteoClient().fetch(31.7, 35.1).is_live is True


# ── mock_weather ──────────────────────────────────────────────────────────────


def test_mock_is_deterministic_and_flagged():
    a = mock_weather(31.7396, 35.1883)
    b = mock_weather(31.7396, 35.1883)
    assert a == b
    assert a.is_live is False
    assert a.source == "Mock weather"
    assert a.temperature_c is not None and a.wind_speed_kmh is not None and a.wind_direction_deg is not None


def test_mock_handles_invalid_coords():
    wx = mock_weather(None, None)
    assert wx.is_live is False and wx.temperature_c is not None


# ── no OpenWeather / no API key remain ────────────────────────────────────────


def test_no_openweather_symbols_or_key():
    assert not hasattr(weather, "get_openweather_key")
    assert not hasattr(weather, "OpenWeatherProvider")


def test_no_openweather_references_in_source():
    src_text = Path(weather.__file__).read_text(encoding="utf-8").lower()
    assert "openweather" not in src_text
    assert "openweathermap" not in src_text
    assert "open-meteo" in src_text


# ── fire_weather_risk (km/h thresholds) ───────────────────────────────────────


def test_hot_dry_windy_is_extreme():
    score, factors, level = fire_weather_risk(HOT_DRY_WINDY)
    assert level == "extreme"
    assert score >= 6
    assert any("humidity" in f for f in factors)
    assert any("km/h" in f for f in factors)


def test_mild_is_low():
    _, _, level = fire_weather_risk(MILD)
    assert level == "low"


def test_missing_fields_are_skipped():
    score, factors, level = fire_weather_risk(Weather())
    assert score == 0 and level == "low" and factors == []


# ── assess_risk ───────────────────────────────────────────────────────────────


def test_assess_high_risk_barn_tip_and_disclaimer():
    adv = assess_risk(HOT_DRY_WINDY, [HIGH_BARN])
    text = " ".join(adv.advisories)
    assert adv.level == "extreme"
    assert "Avoid smoking or hot work near 'Hay Storage'" in text
    assert "Check the high-priority zones before peak heat hours." in adv.advisories
    assert any("not an alert, an ignition prediction, or an emergency dispatch" in a for a in adv.advisories)
    assert "early warning" not in text.lower()


def test_assess_forest_edge_tip_and_downwind():
    adv = assess_risk(HOT_DRY_WINDY, [HIGH_GROVE])
    text = " ".join(adv.advisories)
    assert "checking water hoses" in text
    assert adv.downwind == "S"  # wind_direction_deg 0 (from N) -> downwind S
    assert any("Downwind risk direction is S" in a for a in adv.advisories)


def test_assess_low_risk_has_no_zone_tips_but_has_disclaimer():
    adv = assess_risk(MILD, [HIGH_BARN])
    assert adv.level == "low"
    assert not any("Hay Storage" in a for a in adv.advisories)
    assert any("preventive risk advisory" in a for a in adv.advisories)


def test_assess_no_zones_prompts_configuration():
    adv = assess_risk(HOT_DRY_WINDY, [])
    assert any("Configure image zones" in a for a in adv.advisories)


def test_assess_prioritises_high_priority_zones():
    adv = assess_risk(HOT_DRY_WINDY, [LOW_FIELD, HIGH_BARN])
    text = " ".join(adv.advisories)
    assert "Hay Storage" in text
    assert "North Field" not in text

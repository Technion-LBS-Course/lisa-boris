"""Weather client and fire-weather risk logic for the Risk Advisory agent.

Provides current-weather context (temperature, relative humidity, wind speed and
direction) and turns it into a *preventive risk advisory* against the operator's
configured image zones.

Weather comes from **Open-Meteo** (https://open-meteo.com), whose free forecast
API needs **no API key, signup, or credit card** for non-commercial use. It is
read over stdlib ``urllib`` (no extra dependency). When the live call fails
(network, HTTP, bad JSON, missing fields) or the location is invalid, the module
falls back to a deterministic offline **mock** so the UI never crashes and always
has something to show; the mock is clearly flagged with ``is_live=False``.

This is advisory guidance only. It never claims certain ignition, physical
fire-spread prediction, or emergency dispatch, and it is not an "early warning"
alert. The risk-scoring and advisory functions are pure and unit-testable; only
:func:`fetch_open_meteo_weather` performs network I/O. The module imports no
Streamlit and no ML, and makes no network call at import time.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field

from src.agent_schemas import downwind_direction, int_to_priority_label

# Best-effort: use the OS certificate store for TLS (mirrors src/llm.py) so the
# HTTPS call works behind networks that intercept traffic with their own root CA.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_CURRENT_FIELDS = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
_DEFAULT_LAT, _DEFAULT_LON = 32.0853, 34.7818  # fallback when camera location unset
_DEFAULT_TIMEOUT = 6.0

SOURCE_LIVE = "Open-Meteo"
SOURCE_MOCK = "Mock weather"


@dataclass
class Weather:
    """Normalized current weather. Any measured field may be ``None``.

    Field names match the project's normalized weather schema; :meth:`as_dict`
    returns exactly those keys. ``source`` is ``"Open-Meteo"`` for a live reading
    or ``"Mock weather"`` for the deterministic fallback, and ``is_live`` says
    which one produced this record.
    """

    temperature_c: float | None = None
    relative_humidity: float | None = None
    wind_speed_kmh: float | None = None
    wind_direction_deg: float | None = None
    source: str = SOURCE_MOCK
    is_live: bool = False

    def as_dict(self) -> dict:
        """Return the normalized weather dict (exactly the schema keys)."""
        return asdict(self)


@dataclass
class RiskAdvisory:
    """A preventive fire-weather advisory tied to configured zones."""

    level: str  # low | moderate | high | extreme
    score: int
    factors: list[str] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    downwind: str | None = None


# ── Fire-weather risk scoring (pure, provider-independent) ────────────────────


def fire_weather_risk(weather: Weather) -> tuple[int, list[str], str]:
    """Return ``(score, factors, level)`` from weather variables.

    A simple, transparent index: heat, dryness, and wind raise the score. Missing
    fields are skipped. Wind uses km/h thresholds (Open-Meteo's default unit).
    Levels: score >= 6 extreme, >= 4 high, >= 2 moderate, else low. A coarse risk
    indicator, not an ignition probability.
    """
    score = 0
    factors: list[str] = []

    t = weather.temperature_c
    if t is not None:
        if t >= 35:
            score += 3; factors.append(f"very high temperature ({t:.0f}°C)")
        elif t >= 30:
            score += 2; factors.append(f"high temperature ({t:.0f}°C)")
        elif t >= 25:
            score += 1; factors.append(f"warm ({t:.0f}°C)")

    h = weather.relative_humidity
    if h is not None:
        if h <= 15:
            score += 3; factors.append(f"very low humidity ({h:.0f}%)")
        elif h <= 25:
            score += 2; factors.append(f"low humidity ({h:.0f}%)")
        elif h <= 40:
            score += 1; factors.append(f"moderately dry ({h:.0f}%)")

    w = weather.wind_speed_kmh
    if w is not None:
        if w >= 35:
            score += 3; factors.append(f"strong wind ({w:.0f} km/h)")
        elif w >= 20:
            score += 2; factors.append(f"fresh wind ({w:.0f} km/h)")
        elif w >= 10:
            score += 1; factors.append(f"light wind ({w:.0f} km/h)")

    score = max(0, score)
    level = "extreme" if score >= 6 else "high" if score >= 4 else "moderate" if score >= 2 else "low"
    return score, factors, level


def _weather_phrase(weather: Weather) -> str:
    """Summarize conditions as 'hot, dry and windy' style adjectives."""
    adjectives: list[str] = []
    if weather.temperature_c is not None and weather.temperature_c >= 30:
        adjectives.append("hot")
    if weather.relative_humidity is not None and weather.relative_humidity <= 30:
        adjectives.append("dry")
    if weather.wind_speed_kmh is not None and weather.wind_speed_kmh >= 20:
        adjectives.append("windy")
    if not adjectives:
        return "mild conditions"
    if len(adjectives) == 1:
        return adjectives[0]
    return ", ".join(adjectives[:-1]) + " and " + adjectives[-1]


def _zone_priority_label(zone: dict) -> str:
    return zone.get("priority_label") or int_to_priority_label(int(zone.get("priority", 5)))


def _zone_tip(zone: dict) -> str:
    """A preventive, zone-type-specific tip (deterministic)."""
    name = zone.get("zone_name") or zone.get("alert_label") or "zone"
    zone_type = zone.get("zone_type", "custom")
    tips = {
        "barn": f"Avoid smoking or hot work near '{name}' (hay/storage).",
        "forest_edge": f"Wind and low humidity increase risk near '{name}'. Consider checking water hoses and access.",
        "field": f"Keep machinery and sparks away from dry vegetation in '{name}'; move tractors away from dry brush.",
        "road": f"Watch for hot exhausts and parked vehicles near '{name}'.",
        "parking": f"Keep vehicles clear of dry vegetation around '{name}'.",
        "fence": f"Clear dry brush along '{name}'.",
    }
    return tips.get(zone_type, f"Keep ignition sources away from '{name}'.")


_DISCLAIMER = (
    "This is a preventive risk advisory based on current weather and configured "
    "zones — not an alert, an ignition prediction, or an emergency dispatch."
)


def assess_risk(weather: Weather, zones: list[dict]) -> RiskAdvisory:
    """Turn weather + configured zones into a preventive :class:`RiskAdvisory`.

    Advisory guidance only — never claims certain ignition or fire spread, and
    never dispatches. Zone-specific tips prioritise high-priority zones.
    """
    score, factors, level = fire_weather_risk(weather)
    downwind = (
        downwind_direction(weather.wind_direction_deg)
        if weather.wind_direction_deg is not None else None
    )
    phrase = _weather_phrase(weather)

    advisories: list[str] = []
    if level in ("high", "extreme"):
        advisories.append(
            f"Fire-weather risk is {level} today ({phrase}). Take extra care around your zones."
        )
    elif level == "moderate":
        advisories.append(f"Fire-weather risk is moderate today ({phrase}).")
    else:
        advisories.append(f"Fire-weather risk is low today ({phrase}).")

    enabled = [z for z in zones if z.get("enabled", True)]
    order = {"high": 0, "medium": 1, "low": 2}
    enabled.sort(key=lambda z: order.get(_zone_priority_label(z), 1))
    high_zones = [z for z in enabled if _zone_priority_label(z) == "high"]

    if level in ("moderate", "high", "extreme"):
        for zone in high_zones[:4]:
            advisories.append(_zone_tip(zone))
        if high_zones:
            advisories.append("Check the high-priority zones before peak heat hours.")
        if downwind and level in ("high", "extreme"):
            advisories.append(
                f"Downwind risk direction is {downwind}; prioritise configured zones in that direction."
            )

    if not enabled:
        advisories.append(
            "Configure image zones (Image Zones tab) to get zone-specific preventive tips."
        )

    advisories.append(_DISCLAIMER)
    return RiskAdvisory(level=level, score=score, factors=factors, advisories=advisories, downwind=downwind)


# ── Coordinate validation ─────────────────────────────────────────────────────


def _valid_lat(lat: object) -> bool:
    try:
        return -90.0 <= float(lat) <= 90.0
    except (TypeError, ValueError):
        return False


def _valid_lon(lon: object) -> bool:
    try:
        return -180.0 <= float(lon) <= 180.0
    except (TypeError, ValueError):
        return False


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── Open-Meteo (live) ─────────────────────────────────────────────────────────


def build_open_meteo_url(lat: float, lon: float) -> str:
    """Build the Open-Meteo current-weather URL. Raises ``ValueError`` on bad coords."""
    if not _valid_lat(lat):
        raise ValueError(f"latitude out of range [-90, 90]: {lat!r}")
    if not _valid_lon(lon):
        raise ValueError(f"longitude out of range [-180, 180]: {lon!r}")
    query = urllib.parse.urlencode({
        "latitude": float(lat),
        "longitude": float(lon),
        "current": _CURRENT_FIELDS,
        "timezone": "auto",
    })
    return f"{OPEN_METEO_URL}?{query}"


def parse_open_meteo_current_weather(payload: dict) -> Weather:
    """Parse an Open-Meteo forecast payload's ``current`` block into a live :class:`Weather`.

    Raises ``ValueError`` if the payload has no ``current`` object or is missing
    ``temperature_2m`` (the primary field); other fields may be absent (``None``).
    Callers turn a raised error into the deterministic fallback.
    """
    current = payload.get("current") if isinstance(payload, dict) else None
    if not isinstance(current, dict):
        raise ValueError("Open-Meteo response has no 'current' object")
    if current.get("temperature_2m") is None:
        raise ValueError("Open-Meteo response is missing 'current.temperature_2m'")
    return Weather(
        temperature_c=_as_float(current.get("temperature_2m")),
        relative_humidity=_as_float(current.get("relative_humidity_2m")),
        wind_speed_kmh=_as_float(current.get("wind_speed_10m")),
        wind_direction_deg=_as_float(current.get("wind_direction_10m")),
        source=SOURCE_LIVE,
        is_live=True,
    )


def fetch_open_meteo_weather(lat: float, lon: float, timeout: float = _DEFAULT_TIMEOUT) -> Weather:
    """Fetch live current weather from Open-Meteo. Raises on network/HTTP/JSON errors.

    No API key is used. Prefer :func:`fetch_weather`, which wraps this with the
    deterministic fallback.
    """
    url = build_open_meteo_url(lat, lon)
    request = urllib.request.Request(url, headers={"User-Agent": "PyroFinder/1.0 (course MVP)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec - fixed Open-Meteo host
        payload = json.loads(response.read().decode("utf-8"))
    return parse_open_meteo_current_weather(payload)


class OpenMeteoClient:
    """Thin Open-Meteo client. Keyless; :meth:`fetch` returns a live :class:`Weather`."""

    source = SOURCE_LIVE

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT):
        self.timeout = timeout

    def fetch(self, lat: float, lon: float) -> Weather:
        return fetch_open_meteo_weather(lat, lon, timeout=self.timeout)


# ── Deterministic offline fallback ────────────────────────────────────────────


def mock_weather(lat: float | None, lon: float | None) -> Weather:
    """Deterministic offline weather (summer, elevated-risk bias) from coordinates.

    Used only when the live Open-Meteo call fails or the location is invalid.
    Flagged with ``is_live=False`` and ``source="Mock weather"``.
    """
    lat = float(lat) if _valid_lat(lat) else _DEFAULT_LAT
    lon = float(lon) if _valid_lon(lon) else _DEFAULT_LON
    seed = int(abs(lat) * 1000 + abs(lon) * 1000)
    temperature = 28 + (seed % 12)              # 28..39 °C
    humidity = 15 + (seed * 7 % 45)             # 15..59 %
    wind_kmh = 10 + (seed * 13 % 30)            # 10..39 km/h
    wind_dir = seed * 3 % 360
    return Weather(
        temperature_c=float(temperature),
        relative_humidity=float(humidity),
        wind_speed_kmh=float(wind_kmh),
        wind_direction_deg=float(wind_dir),
        source=SOURCE_MOCK,
        is_live=False,
    )


def fetch_weather(lat: float | None, lon: float | None, timeout: float = _DEFAULT_TIMEOUT) -> Weather:
    """Return current weather: live Open-Meteo when possible, else the deterministic mock.

    Requires no API key. Invalid coordinates, or any live-call failure (network,
    HTTP, JSON, missing fields), fall back to the mock so the advisory panel always
    has data to show.
    """
    if not (_valid_lat(lat) and _valid_lon(lon)):
        return mock_weather(lat, lon)
    try:
        return fetch_open_meteo_weather(lat, lon, timeout=timeout)
    except Exception:
        return mock_weather(lat, lon)

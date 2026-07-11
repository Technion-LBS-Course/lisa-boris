"""Incident Assistant — operational response for a confirmed fire/smoke detection.

This agent runs *after* the detection pipeline (YOLO11s detection in
``src/inference.py`` + N-frame confirmation in ``src/tracking.py``) confirms a
hazard. It receives the detection facts plus the operator's configured camera,
image zones, reference points, and optional Open-Meteo weather context, and
produces:

* an :class:`IncidentContext` (event location, matched image zone, estimated map
  point, apparent in-frame movement, wind/downwind context),
* operational recommendations,
* an operational-conversation narrative + a responder for operator chat, and
* draft messages (property owner / neighbor / farm worker / fire-department
  summary).

The assistant only *drafts and recommends*. It never contacts anyone, never
dispatches emergency services, and never predicts physical fire spread. Worker
messages use the zone name and task — not coordinates.

Pure module: no Streamlit and no ML imports. It reuses ``src/mapping.py``,
``src/tracking.py``, and ``src/alerts.py`` (all pure). An optional AI wording
polish imports ``src/llm.py`` lazily and degrades to the original text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# compass_label / downwind_direction live in agent_schemas (shared with src/weather.py)
# and are re-exported here for callers and tests that import them from this module.
from src.agent_schemas import compass_label, downwind_direction, int_to_priority_label
from src.alerts import create_alert_record
from src.mapping import (
    estimate_map_position,
    image_quadrant,
    point_in_polygon,
    polygon_centroid_norm,
    zone_reference_point_norm,
)
from src.tracking import estimate_apparent_direction

__all__ = [
    "compass_label",
    "downwind_direction",
    "IncidentContext",
    "build_incident_context",
    "find_nearest_zones",
    "summarize_operational_context",
    "contact_guidance",
    "recommend_actions",
    "incident_narrative",
    "format_initial_incident_message",
    "initial_incident_message",
    "incident_reasoning",
    "respond_to_operator",
    "build_incident_system_prompt",
    "conversation_uses_llm",
    "build_drafts",
    "create_incident_alert",
    "polish_message",
]


@dataclass
class IncidentContext:
    """Assembled facts for one confirmed incident (all locations approximate)."""

    camera_id: str
    detected_class: str
    confidence: float
    centroid_norm: tuple[float, float]
    location_text: str
    matched_zone: str | None = None
    zone_type: str | None = None
    zone_priority_label: str | None = None
    approximate_lat: float | None = None
    approximate_lon: float | None = None
    # Where the approximate map point came from: "zone_reference_point" when the
    # matched zone's operator-set reference point was projected, "detection_anchor"
    # when the detection's own image point was projected, or None when no map
    # point exists.
    map_point_source: str | None = None
    image_plane_direction: str | None = None
    downwind_risk_direction: str | None = None
    # Weather context (Open-Meteo live reading or offline mock; may be absent).
    temperature_c: float | None = None
    relative_humidity: float | None = None
    wind_speed_kmh: float | None = None
    wind_direction_deg: float | None = None
    wind_compass: str | None = None
    weather_source: str | None = None
    weather_is_live: bool = False
    camera_name: str = ""
    site_id: str | None = None
    customer_id: str | None = None
    timestamp: str | None = None
    notes: str = ""
    # Nearest defined zones to the detection (approximate image-space distance),
    # used to reason about incidents near — but outside — a marked zone. This is
    # derived incident context, not zone metadata.
    nearest_zones: list = field(default_factory=list)
    # Optional operational context (landmarks, sensitive receptors, contact policy)
    # loaded from an external file. Used only for reasoning + first-message wording;
    # it is NOT zone metadata and never changes the zone schema.
    operational_context: dict | None = None
    operational_context_md: str | None = None

    def display_rows(self) -> list[tuple[str, str]]:
        """Return (label, value) rows for a compact incident summary table."""
        rows: list[tuple[str, str]] = [
            ("Camera", f"{self.camera_id}" + (f" ({self.camera_name})" if self.camera_name else "")),
            ("Detected", f"{self.detected_class} · confidence {self.confidence:.0%}"),
            ("Event location", self.location_text),
        ]
        if self.matched_zone:
            zone = self.matched_zone
            if self.zone_priority_label:
                zone += f" ({self.zone_priority_label} priority)"
            rows.append(("Mapped zone", zone))
        if self.approximate_lat is not None and self.approximate_lon is not None:
            source = ""
            if self.map_point_source == "zone_reference_point":
                source = " (from zone reference point)"
            elif self.map_point_source == "detection_anchor":
                source = " (from detection point)"
            rows.append((
                "Estimated map point",
                f"~{self.approximate_lat:.5f}, {self.approximate_lon:.5f}{source}",
            ))
        elif self.matched_zone and "reference point not set" in self.location_text:
            rows.append((
                "Estimated map point",
                "unavailable — set a zone reference point for this zone",
            ))
        if self.image_plane_direction:
            rows.append(("Apparent movement (in frame)", self.image_plane_direction))
        if self.temperature_c is not None or self.relative_humidity is not None:
            parts = []
            if self.temperature_c is not None:
                parts.append(f"{self.temperature_c:.0f}°C")
            if self.relative_humidity is not None:
                parts.append(f"RH {self.relative_humidity:.0f}%")
            label = "Weather" + (" (live)" if self.weather_is_live else " (fallback)")
            rows.append((label, ", ".join(parts) + (f" · {self.weather_source}" if self.weather_source else "")))
        if self.wind_speed_kmh is not None and self.wind_compass:
            wind = f"{self.wind_speed_kmh:.0f} km/h from {self.wind_compass}"
            if self.downwind_risk_direction:
                wind += f" · risk toward {self.downwind_risk_direction}"
            rows.append(("Wind", wind))
        return rows


def _wget(weather, name):
    """Read a field from a weather object (dataclass attr) or dict, else None."""
    if weather is None:
        return None
    if isinstance(weather, dict):
        return weather.get(name)
    return getattr(weather, name, None)


def _match_zone(centroid_norm: tuple[float, float], image_zones: list[dict]) -> dict | None:
    """Return the first enabled, drawn image zone containing the centroid, else None."""
    px, py = centroid_norm
    for zone in image_zones:
        if not zone.get("enabled", True):
            continue
        vertices_norm = zone.get("vertices_norm", [])
        if len(vertices_norm) >= 3 and point_in_polygon(px, py, vertices_norm):
            return zone
    return None


def find_nearest_zones(
    centroid_norm: tuple[float, float],
    image_zones: list[dict],
    limit: int = 3,
) -> list[dict]:
    """Return the nearest enabled, drawn zones to a normalized image point.

    Distance is the approximate image-space distance from the point to each zone
    polygon's centroid, in normalized [0, 1] units — approximate, never a
    geographic distance. A zone that contains the point sorts first with distance
    0. Each result is ``{"zone_name", "priority_label", "distance_norm",
    "contains"}``. Pure and testable — used to reason about an incident that is
    near, but outside, a marked zone.
    """
    import math

    px, py = centroid_norm
    scored: list[dict] = []
    for zone in image_zones:
        if not zone.get("enabled", True):
            continue
        vertices_norm = [tuple(v) for v in zone.get("vertices_norm", [])]
        if len(vertices_norm) < 3:
            continue
        center = polygon_centroid_norm(vertices_norm)
        if center is None:
            continue
        contains = point_in_polygon(px, py, vertices_norm)
        distance = 0.0 if contains else math.hypot(px - center[0], py - center[1])
        scored.append({
            "zone_name": zone.get("zone_name") or zone.get("alert_label"),
            "priority_label": zone.get("priority_label")
            or int_to_priority_label(int(zone.get("priority", 5))),
            "distance_norm": round(distance, 4),
            "contains": contains,
        })
    scored.sort(key=lambda z: (not z["contains"], z["distance_norm"]))
    return scored[:limit]


# ── Optional operational context (landmarks / receptors / contact policy) ─────
#
# An external file (loaded by src/live_ops_config.py) supplies operational meaning
# around the scene. It is consumed ONLY for incident reasoning and first-message
# wording — never for detection, and never merged into zone records. Missing
# context degrades gracefully (helpers return "" / generic wording).


def _verified_contacts(operational_context: dict | None) -> list[dict]:
    """Contacts from the operational context that carry a real value (not null)."""
    if not isinstance(operational_context, dict):
        return []
    return [
        c for c in (operational_context.get("authorities_and_contacts") or [])
        if isinstance(c, dict) and c.get("contact")
    ]


def summarize_operational_context(
    operational_context: dict | None,
    operational_context_md: str | None = None,
    *,
    include_contacts: bool = True,
    max_landmarks: int = 6,
) -> str:
    """Return a compact operational-context brief for grounding, or ``""``.

    Prefers the structured JSON (deterministic, bounded); falls back to a truncated
    Markdown excerpt when only the Markdown is available. Used to ground the LLM
    (concise first message, operator chat) and to enrich the detailed reasoning.
    When ``include_contacts`` is False the verified contact list is omitted (so raw
    phone numbers never leak into the short first message) while the generic contact
    policy line is kept. Pure — no file/network access.
    """
    oc = operational_context if isinstance(operational_context, dict) else None
    if oc:
        lines: list[str] = []
        site = oc.get("primary_site_context") or {}
        if isinstance(site, dict) and site.get("name"):
            rel = site.get("operational_relevance", "")
            lines.append(f"Camera site: {site['name']}." + (f" {rel}" if rel else ""))
        landmarks = oc.get("nearby_operational_landmarks") or []
        named = [lm for lm in landmarks if isinstance(lm, dict) and lm.get("name")]
        if named:
            lines.append("Nearby operational places (approximate, context-based):")
            for lm in named[:max_landmarks]:
                sens = lm.get("sensitivity", "")
                hint = lm.get("recommended_action_hint", "")
                bits = "; ".join(b for b in (f"sensitivity {sens}" if sens else "", hint) if b)
                lines.append(f"- {lm['name']}" + (f" ({bits})" if bits else ""))
        if include_contacts:
            verified = _verified_contacts(oc)
            if verified:
                lines.append("Verified contacts (operator places the call — never automatic):")
                for c in verified:
                    usage = f" — {c['usage_rule']}" if c.get("usage_rule") else ""
                    lines.append(f"- {c.get('name')}: {c.get('contact')}{usage}")
        lines.append(
            "Contact policy: never invent a contact. Use a verified contact when relevant, "
            "else generic wording (relevant local authority / local emergency contact / "
            "site operator / property owner contact). If a needed contact is missing, you may "
            "offer to search but must never search automatically. All locations are approximate."
        )
        return "\n".join(lines)
    if operational_context_md:
        text = operational_context_md.strip()
        return text if len(text) <= 2400 else text[:2400].rstrip() + "\n… (context truncated)"
    return ""


def contact_guidance(context: IncidentContext) -> str:
    """Deterministic answer to a 'who do I contact?' question.

    Lists verified contacts from the operational context when present; otherwise
    uses generic authority wording and offers (without performing) a contact search.
    Never invents a contact. PyroFinder never contacts anyone automatically.
    """
    verified = _verified_contacts(context.operational_context)
    if verified:
        lines = [
            "Verified contacts from the operational context (you place the call — PyroFinder "
            "never contacts anyone automatically):"
        ]
        for c in verified:
            usage = f" — {c['usage_rule']}" if c.get("usage_rule") else ""
            lines.append(f"- {c.get('name')}: {c.get('contact')}{usage}")
        lines.append(
            "For any recipient without a verified contact, I use generic wording and never "
            "invent a name or number."
        )
        return "\n".join(lines)
    return (
        "I don't have a verified contact on file, so I can only refer to the relevant local "
        "authority / local emergency contact / site operator / property owner contact. Do you "
        "want me to search for the relevant contact? I won't search the web unless you say yes."
    )


def build_incident_context(
    *,
    camera: dict,
    image_zones: list[dict],
    reference_points: list[dict],
    detected_class: str,
    confidence: float,
    centroid_norm: tuple[float, float],
    prev_centroid_norm: tuple[float, float] | None = None,
    weather=None,
    timestamp: str | None = None,
    operational_context: dict | None = None,
    operational_context_md: str | None = None,
) -> IncidentContext:
    """Assemble an :class:`IncidentContext` from a confirmed detection and config.

    ``weather`` is optional and may be a ``src.weather.Weather`` (or any object /
    dict exposing ``temperature_c``, ``relative_humidity``, ``wind_speed_kmh``,
    ``wind_direction_deg``, ``source``, ``is_live``). When wind direction is known
    the downwind risk direction and compass label are derived. Missing weather is
    fine — the incident still assembles. Locations are approximate.
    """
    cx, cy = centroid_norm

    nearest_zones = find_nearest_zones(centroid_norm, image_zones)
    zone = _match_zone(centroid_norm, image_zones)
    matched_zone = zone_type = zone_priority_label = None
    if zone is not None:
        matched_zone = zone.get("zone_name") or zone.get("alert_label")
        zone_type = zone.get("zone_type")
        zone_priority_label = zone.get("priority_label") or int_to_priority_label(
            int(zone.get("priority", 5))
        )

    # Map-point priority: the matched zone's operator-set reference point comes
    # first; the detection-anchor projection is used only when no zone matched;
    # otherwise the image quadrant / camera-frame text is the fallback. A matched
    # zone WITHOUT a reference point never gets an invented map point — the
    # missing point is reported instead. All map outputs are approximate.
    approx_lat = approx_lon = None
    map_point_source = None
    zone_ref_missing = False
    if zone is not None:
        zone_ref = zone_reference_point_norm(zone)
        if zone_ref is not None:
            projection = estimate_map_position(reference_points, zone_ref)
            if projection is not None:
                approx_lat, approx_lon = projection
                map_point_source = "zone_reference_point"
        else:
            zone_ref_missing = True
    else:
        projection = estimate_map_position(reference_points, centroid_norm)
        if projection is not None:
            approx_lat, approx_lon = projection
            map_point_source = "detection_anchor"

    if matched_zone:
        base = f"mapped zone '{matched_zone}'"
        if map_point_source == "zone_reference_point":
            location_text = f"{base} — approximate map point from zone reference point"
        elif zone_ref_missing:
            location_text = (
                f"{base} — zone reference point not set, no approximate map point"
            )
        else:
            location_text = f"{base} — camera-frame location"
    else:
        base = f"the {image_quadrant(cx, cy)} area of the camera frame"
        if approx_lat is not None and approx_lon is not None:
            location_text = f"{base} — estimated location ~{approx_lat:.4f}, {approx_lon:.4f}"
        else:
            location_text = f"{base} — camera-frame location"

    image_plane_direction = None
    if prev_centroid_norm is not None:
        image_plane_direction = estimate_apparent_direction(prev_centroid_norm, centroid_norm)

    wind_direction_deg = _wget(weather, "wind_direction_deg")
    wind_compass = compass_label(wind_direction_deg) if wind_direction_deg is not None else None
    downwind = downwind_direction(wind_direction_deg) if wind_direction_deg is not None else None

    return IncidentContext(
        camera_id=camera.get("camera_id", ""),
        camera_name=camera.get("camera_name", ""),
        site_id=camera.get("site_id") or None,
        customer_id=camera.get("customer_id") or None,
        detected_class=detected_class,
        confidence=confidence,
        centroid_norm=centroid_norm,
        matched_zone=matched_zone,
        zone_type=zone_type,
        zone_priority_label=zone_priority_label,
        location_text=location_text,
        approximate_lat=approx_lat,
        approximate_lon=approx_lon,
        map_point_source=map_point_source,
        image_plane_direction=image_plane_direction,
        downwind_risk_direction=downwind,
        temperature_c=_wget(weather, "temperature_c"),
        relative_humidity=_wget(weather, "relative_humidity"),
        wind_speed_kmh=_wget(weather, "wind_speed_kmh"),
        wind_direction_deg=wind_direction_deg,
        wind_compass=wind_compass,
        weather_source=_wget(weather, "source"),
        weather_is_live=bool(_wget(weather, "is_live")),
        timestamp=timestamp,
        nearest_zones=nearest_zones,
        operational_context=operational_context if isinstance(operational_context, dict) else None,
        operational_context_md=operational_context_md,
    )


def recommend_actions(context: IncidentContext) -> list[str]:
    """Return operational recommendations. Advisory only — never auto-actions."""
    recommendations: list[str] = []
    is_high = context.zone_priority_label == "high"

    if context.detected_class == "fire":
        recommendations.append(
            "Confirmed fire — recommend immediate on-site verification of the event area."
        )
        if context.matched_zone and is_high:
            recommendations.append(
                f"'{context.matched_zone}' is a high-priority zone — prepare to contact the "
                "fire department if the fire is verified."
            )
    else:  # smoke
        recommendations.append(
            "Smoke detected — verify whether it indicates an early fire or a non-fire source "
            "(dust, haze, exhaust, cloud) before escalating."
        )
        if context.matched_zone and is_high:
            recommendations.append(
                f"Smoke is in high-priority zone '{context.matched_zone}' — verify promptly."
            )

    if context.wind_compass and context.downwind_risk_direction:
        recommendations.append(
            f"Wind is from the {context.wind_compass}, so smoke/risk may move toward the "
            f"{context.downwind_risk_direction} — check zones and access routes in that direction."
        )
    if (context.temperature_c is not None and context.temperature_c >= 30) or (
        context.relative_humidity is not None and context.relative_humidity <= 30
    ):
        recommendations.append(
            "Hot / dry conditions raise risk — keep water and suppression tools ready."
        )
    if context.approximate_lat is not None and context.approximate_lon is not None:
        recommendations.append("An estimated location is available to share with responders.")

    recommendations.append(
        "Confirm the alert before contacting anyone — PyroFinder never contacts emergency "
        "services or dispatches automatically."
    )
    return recommendations


# ── Operational conversation ──────────────────────────────────────────────────


def incident_narrative(context: IncidentContext) -> str:
    """Return the opening operational-conversation line for the incident."""
    where = context.matched_zone or "the monitored area"
    priority = ""
    if context.matched_zone and context.zone_priority_label:
        priority = f" ({context.zone_priority_label} priority)"
    opener = f"{context.detected_class.capitalize()} event detected near {where}{priority}."

    wind_sentence = ""
    if context.wind_compass and context.downwind_risk_direction:
        wind_sentence = (
            f" Wind is from the {context.wind_compass}, so the risk may move toward the "
            f"{context.downwind_risk_direction}."
        )
    next_step = "notify people assigned to nearby zones and check access routes"
    return (
        f"{opener}{wind_sentence} Recommended next step: {next_step}. "
        "What would you like to do?"
    )


# ── Concise initial incident message (driven by the structured context) ───────
#
# The opening operator-facing line must be short and action-oriented: what was
# detected, where in operational terms, likely drift, and one next-step question.
# Raw telemetry (temperature, humidity, confidence, frame coordinates) is kept out
# of the opener and surfaced only via incident_reasoning() when the operator asks.

_COMPASS_WORDS = {
    "N": "north", "NE": "northeast", "E": "east", "SE": "southeast",
    "S": "south", "SW": "southwest", "W": "west", "NW": "northwest",
}

_INITIAL_MESSAGE_POLISH = (
    "You are PyroFinder's incident assistant briefing a site operator. Rewrite the "
    "message below as ONE short, calm, operational line — at most two sentences — that "
    "ends with a single yes/no question. Do NOT add any number, temperature, humidity, "
    "confidence value, coordinate, camera ID, or contact name that is not already "
    "present, and do not invent facts. Return only the rewritten message.\n\n"
)


def _nearest_outside_zone(context: IncidentContext) -> dict | None:
    """First nearest zone that does NOT contain the detection, else None."""
    return next(
        (z for z in (context.nearest_zones or []) if not z.get("contains")), None
    )


def _initial_where(context: IncidentContext) -> str:
    """Operational 'where' phrase for the opener — no coordinates or telemetry."""
    if context.matched_zone:
        return f"near {context.matched_zone}"
    nearest = _nearest_outside_zone(context)
    if nearest and nearest.get("zone_name"):
        return f"outside the marked zones, near {nearest['zone_name']}"
    cx, cy = context.centroid_norm
    return f"in the {image_quadrant(cx, cy)} area of the frame"


def _initial_priority_label(context: IncidentContext) -> str | None:
    """Priority driving urgency: the matched zone's, else the nearest zone's."""
    if context.matched_zone:
        return context.zone_priority_label
    nearest = _nearest_outside_zone(context)
    return nearest.get("priority_label") if nearest else None


def format_initial_incident_message(context: IncidentContext) -> str:
    """Build the concise, action-oriented opening incident line (deterministic).

    Reasons from the structured incident context — what was detected, where in
    operational terms (matched zone, else nearest zone, else image quadrant), and
    the likely drift direction — then ends with one recommended next-action
    question. Deliberately omits raw telemetry (temperature, humidity, confidence,
    frame coordinates) and never invents a contact: an unknown recipient is
    referred to generically ('the relevant local authority').
    """
    camera = context.camera_name or context.camera_id or "The camera"
    subject = "a possible fire" if context.detected_class == "fire" else "smoke"
    where = _initial_where(context)

    drift = ""
    if context.downwind_risk_direction:
        word = _COMPASS_WORDS.get(
            context.downwind_risk_direction, context.downwind_risk_direction
        )
        drift = f", drifting {word}"

    high = _initial_priority_label(context) == "high"
    known_area = bool(context.matched_zone or _nearest_outside_zone(context))
    if context.detected_class == "fire":
        question = (
            "Do you want me to prepare a dispatch to the relevant local authority?"
            if high or context.matched_zone
            else "It looks low priority for now — should I keep monitoring?"
        )
    else:  # smoke
        question = (
            "Should I prepare a response update?"
            if high or known_area
            else "Should I keep monitoring, or prepare a notification?"
        )

    return f"{camera} detected {subject} {where}{drift}. {question}"


def initial_incident_message(context: IncidentContext) -> str:
    """Return the first operator-facing incident line.

    The deterministic :func:`format_initial_incident_message` is the grounded
    source of truth; when Groq is configured it only *rephrases* that line under a
    strict no-new-facts instruction. Any failure returns the deterministic text so
    the opener is always concise and never invents telemetry or contacts.
    """
    base = format_initial_incident_message(context)
    if _groq_ready():
        try:
            from src import llm

            # Contacts are excluded from the opener brief so phone numbers / addresses
            # never leak into the short first message; place names + policy still guide it.
            brief = summarize_operational_context(
                context.operational_context, context.operational_context_md,
                include_contacts=False,
            )
            prompt = _INITIAL_MESSAGE_POLISH
            if brief:
                prompt += (
                    "Operational context you MAY use for place names and sensitivity (do not "
                    "add phone numbers, addresses, coordinates, or invented contacts):\n"
                    f"{brief}\n\n"
                )
            prompt += "Message to rewrite:\n" + base
            refined = llm.ask(prompt)
            if refined and refined.strip():
                return refined.strip()
        except Exception:
            pass
    return base


def incident_reasoning(context: IncidentContext) -> str:
    """Return the detailed reasoning + supporting context for the incident.

    Surfaced only when the operator explicitly asks ('why', 'how did you decide',
    'explain') — the opening message stays concise. Built from the same structured
    context: the incident summary rows, the nearest zone, and recommendations.
    """
    lines = ["Here's how I read this incident:"]
    for label, value in context.display_rows():
        lines.append(f"- {label}: {value}")
    if not context.matched_zone:
        nearest = _nearest_outside_zone(context)
        if nearest:
            lines.append(
                f"- Nearest zone: {nearest['zone_name']} "
                f"({nearest['priority_label']} priority), ~{nearest['distance_norm']:.2f} "
                "away in the frame (approximate image-space distance)."
            )
    lines.append("")
    lines.append("Recommended actions:")
    for rec in recommend_actions(context):
        lines.append(f"- {rec}")
    brief = summarize_operational_context(
        context.operational_context, context.operational_context_md
    )
    if brief:
        lines.append("")
        lines.append("Operational context:")
        lines.append(brief)
    return "\n".join(lines)


def build_incident_system_prompt(context: IncidentContext) -> str:
    """Build the LLM system prompt: the incident facts + operational guardrails."""
    facts = [
        f"Detected: {context.detected_class} (confidence {context.confidence:.0%}).",
        f"Camera: {context.camera_id}"
        + (f" ({context.camera_name})" if context.camera_name else "") + ".",
        f"Event location: {context.location_text}.",
    ]
    if context.matched_zone:
        prio = f" ({context.zone_priority_label} priority)" if context.zone_priority_label else ""
        facts.append(f"Mapped zone: {context.matched_zone}{prio}.")
    if context.wind_compass and context.wind_speed_kmh is not None:
        wind = f"Wind: from the {context.wind_compass} at {context.wind_speed_kmh:.0f} km/h"
        if context.downwind_risk_direction:
            wind += f"; risk may move toward the {context.downwind_risk_direction}"
        facts.append(wind + ".")
    if context.temperature_c is not None or context.relative_humidity is not None:
        cond = []
        if context.temperature_c is not None:
            cond.append(f"{context.temperature_c:.0f}°C")
        if context.relative_humidity is not None:
            cond.append(f"RH {context.relative_humidity:.0f}%")
        src = f" (source {context.weather_source})" if context.weather_source else ""
        facts.append("Conditions: " + ", ".join(cond) + src + ".")
    if context.image_plane_direction:
        facts.append(f"Apparent movement in the camera frame: {context.image_plane_direction}.")
    facts_block = "\n".join(f"- {f}" for f in facts)
    prompt = (
        "You are PyroFinder's Incident Assistant, helping a site operator respond to a "
        "CONFIRMED fire/smoke detection. Be concise, calm, and operational.\n\n"
        "Incident facts (rely on these; do not invent camera IDs, zones, or coordinates):\n"
        f"{facts_block}\n\n"
        "Rules:\n"
        "- You only draft messages and give operational recommendations. You cannot send "
        "anything, contact anyone, or dispatch emergency services — produce drafts for the "
        "operator to send.\n"
        "- Never claim automatic emergency dispatch, and never predict physical fire spread. "
        "PyroFinder detects only fire and smoke.\n"
        "- For a WORKER / farm hand / field team: give the zone name and a concrete task, the "
        "urgency, what to check, what to avoid, and how to report back. Do NOT include GPS "
        "coordinates or lat/lon in worker messages.\n"
        "- For the OWNER / operator: you may include the camera, zone, class, confidence, "
        "weather/wind, and a recommended next step.\n"
        "- For a NEIGHBOUR: keep it short and plain, no coordinates.\n"
        "- For a FIRE-DEPARTMENT summary: you may include the estimated coordinates if "
        "available, clearly marked approximate.\n"
        "- Never invent a contact. Use a verified contact from the operational context when "
        "relevant, otherwise generic wording (relevant local authority / local emergency "
        "contact / site operator). Do not perform any web/API contact lookup.\n"
        "- If the operator asks to confirm or dismiss, tell them to use the 'Confirm alert' or "
        "'Mark as false alarm' button.\n"
        "- Keep replies to a few sentences unless asked for a full draft."
    )
    brief = summarize_operational_context(
        context.operational_context, context.operational_context_md
    )
    if brief:
        prompt += (
            "\n\nOperational context (use for place names, sensitivity, and contact policy; "
            "do not invent facts or contacts):\n" + brief
        )
    return prompt


def _groq_ready() -> bool:
    """True when Groq is importable and a key is configured (for the LLM chat)."""
    try:
        from src import llm

        return llm.groq_available() and llm.api_key_present()
    except Exception:
        return False


def conversation_uses_llm() -> bool:
    """Whether operator replies will be generated by Groq (vs the deterministic responder)."""
    return _groq_ready()


def respond_to_operator(context: IncidentContext, message: str, history=None) -> str:
    """Answer an operator chat message.

    Uses Groq as a free-form operational assistant when a key is configured
    (passing the incident facts as a system prompt plus the conversation history);
    otherwise falls back to the deterministic keyword responder. Never raises.
    """
    if _groq_ready():
        try:
            from src import llm

            messages = [{"role": "system", "content": build_incident_system_prompt(context)}]
            for turn in history or []:
                role, content = turn.get("role"), turn.get("content")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": message})
            reply = llm.chat(messages)
            if reply and reply.strip():
                return reply.strip()
        except Exception:
            pass
    return _deterministic_reply(context, message)


def _deterministic_reply(context: IncidentContext, message: str) -> str:
    """Deterministic keyword-routed reply — the offline fallback when Groq is absent.

    Returns a worker/owner/neighbor/fire-department draft, or a short menu.
    """
    text = (message or "").lower()
    if any(k in text for k in (
        "why", "explain", "reason", "how did you", "how do you", "justif", "detail"
    )):
        return incident_reasoning(context)
    if any(k in text for k in (
        "who do i call", "who should i call", "who to call", "contact", "phone number",
        "search for the contact", "search for a contact",
    )):
        return contact_guidance(context)
    if any(k in text for k in ("worker", "farm", "staff", "crew", "field team")):
        return draft_farm_worker_message(context)
    if any(k in text for k in ("neighbor", "neighbour")):
        return draft_neighbor_message(context)
    if "fire" in text and any(k in text for k in ("depart", "brigade", "service", "102", "911", "summary")):
        return prepare_fire_department_summary(context)
    if any(k in text for k in ("owner", "manager", "boss", "operator")):
        return draft_owner_message(context)
    if any(k in text for k in ("false alarm", "dismiss", "not real")):
        return "To dismiss this, use the 'Mark as false alarm' button below the conversation."
    if "confirm" in text:
        return "To record this as a confirmed alert, use the 'Confirm alert' button below."
    recs = recommend_actions(context)
    return (
        "I can draft a message for the farm workers, the owner, a neighbour, or a "
        f"fire-department summary — tell me who. Current recommendation: {recs[0]}"
    )


# ── Draft messages (operator reviews and sends; nothing is sent automatically) ─


def _time_str(context: IncidentContext) -> str:
    return context.timestamp or "the reported time"


def draft_owner_message(context: IncidentContext) -> str:
    """Draft an alert message to the property owner (may include weather + estimated location)."""
    lines = [
        f"PyroFinder alert: {context.detected_class} detected on camera "
        f"{context.camera_id}"
        + (f" ({context.camera_name})" if context.camera_name else "")
        + f" at {_time_str(context)}.",
        f"Confidence: {context.confidence:.0%}.",
        f"Location: {context.location_text}.",
    ]
    if context.image_plane_direction:
        lines.append(f"Apparent movement in the frame: {context.image_plane_direction}.")
    if context.wind_compass and context.wind_speed_kmh is not None:
        wind = f"Wind from the {context.wind_compass} ({context.wind_speed_kmh:.0f} km/h)"
        if context.downwind_risk_direction:
            wind += f"; risk may move toward the {context.downwind_risk_direction}"
        lines.append(wind + ".")
    if context.temperature_c is not None or context.relative_humidity is not None:
        cond = []
        if context.temperature_c is not None:
            cond.append(f"{context.temperature_c:.0f}°C")
        if context.relative_humidity is not None:
            cond.append(f"RH {context.relative_humidity:.0f}%")
        lines.append("Conditions: " + ", ".join(cond) + ".")
    lines.append(
        "Recommended next step: verify on-site and decide whether to escalate. This is an "
        "operational alert, not an automatic emergency call."
    )
    return "\n".join(lines)


def _worker_task(context: IncidentContext) -> str:
    """Zone-type-specific field task for a worker (no coordinates)."""
    tasks = {
        "forest_edge": "check the tree line and move equipment away from the smoke path.",
        "field": "check the field edge and move machinery/tractors away from the smoke path.",
        "barn": "check around the hay/storage area and keep ignition sources away.",
        "road": "check the access route and keep vehicles clear.",
        "parking": "check the parking area and move vehicles clear of the smoke path.",
        "fence": "check the fence line and clear dry brush nearby.",
    }
    return tasks.get(context.zone_type or "", "check the area and move equipment away from the smoke path.")


def draft_farm_worker_message(context: IncidentContext) -> str:
    """Draft a worker task: zone + task + urgency + what to check/avoid + report. No coordinates."""
    zone = context.matched_zone or "the monitored area"
    priority = context.zone_priority_label or "standard"
    return (
        f"{zone} — {priority} priority: {_worker_task(context)} "
        "Avoid entering the smoky area. Report whether flames are visible."
    )


def draft_neighbor_message(context: IncidentContext) -> str:
    """Draft a short courtesy heads-up to a neighbor. No coordinates."""
    where = context.matched_zone or "the boundary"
    return (
        f"Heads-up: {context.detected_class} was detected near {where} at {_time_str(context)}. "
        "You may want to check your side. Courtesy notice, not an emergency dispatch."
    )


def prepare_fire_department_summary(context: IncidentContext) -> str:
    """Prepare a fire-department call summary for the operator to relay (not auto-sent)."""
    lines = [
        "PyroFinder incident summary (for the operator to relay — PyroFinder does not "
        "contact emergency services automatically):",
        f"- Time: {_time_str(context)}",
        f"- Site / Customer: {context.site_id or 'n/a'} / {context.customer_id or 'n/a'}",
        f"- Camera: {context.camera_id}"
        + (f" ({context.camera_name})" if context.camera_name else ""),
        f"- Detected: {context.detected_class}, confidence {context.confidence:.0%}",
        f"- Location: {context.location_text}",
    ]
    if context.matched_zone:
        priority = f" ({context.zone_priority_label} priority)" if context.zone_priority_label else ""
        lines.append(f"- Mapped zone: {context.matched_zone}{priority}")
    if context.approximate_lat is not None and context.approximate_lon is not None:
        lines.append(
            f"- Estimated coordinates: ~{context.approximate_lat:.5f}, "
            f"{context.approximate_lon:.5f} (approximate — confirm on arrival)"
        )
    if context.wind_compass and context.wind_speed_kmh is not None:
        lines.append(f"- Wind: from the {context.wind_compass}, {context.wind_speed_kmh:.0f} km/h")
    return "\n".join(lines)


def build_drafts(context: IncidentContext) -> dict[str, str]:
    """Return all draft messages keyed by audience."""
    return {
        "Property owner": draft_owner_message(context),
        "Neighbor": draft_neighbor_message(context),
        "Farm worker": draft_farm_worker_message(context),
        "Fire department summary": prepare_fire_department_summary(context),
    }


def create_incident_alert(context: IncidentContext, status: str = "active") -> dict:
    """Build an alert record for this incident via ``src/alerts.create_alert_record``.

    ``geographic_bearing`` is intentionally omitted (None) — the app's camera
    metadata carries no registered compass bearing, so a true bearing is not claimed.
    """
    return create_alert_record(
        camera_id=context.camera_id,
        detected_class=context.detected_class,
        confidence=context.confidence,
        approximate_location=context.location_text,
        apparent_direction=context.image_plane_direction or "n/a",
        status=status,
        timestamp=context.timestamp,
        site_id=context.site_id,
        customer_id=context.customer_id,
        image_polygon_name=context.matched_zone,
        approximate_lat=context.approximate_lat,
        approximate_lon=context.approximate_lon,
        geographic_bearing=None,
    )


def polish_message(text: str, audience: str = "recipient") -> str:
    """Optionally refine a draft's wording via Groq; return the original on any failure.

    Best-effort and fully optional — importing Groq lazily keeps this safe when the
    package or key is absent. The instruction forbids adding emergency-dispatch or
    fire-spread claims, so the refined text stays within scope.
    """
    if not text or not text.strip():
        return text
    try:
        from src import llm

        prompt = (
            "Rewrite the following operational alert message to be clear and calm for a "
            f"{audience}. Keep every fact unchanged. Do NOT add any claim of automatic "
            "emergency dispatch or fire-spread prediction. Return only the rewritten "
            "message.\n\n" + text
        )
        refined = llm.ask(prompt)
    except Exception:
        return text
    return refined.strip() or text

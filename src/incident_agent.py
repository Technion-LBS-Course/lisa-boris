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
    zone_reference_point_norm,
)
from src.tracking import estimate_apparent_direction

__all__ = [
    "compass_label",
    "downwind_direction",
    "IncidentContext",
    "build_incident_context",
    "recommend_actions",
    "incident_narrative",
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
) -> IncidentContext:
    """Assemble an :class:`IncidentContext` from a confirmed detection and config.

    ``weather`` is optional and may be a ``src.weather.Weather`` (or any object /
    dict exposing ``temperature_c``, ``relative_humidity``, ``wind_speed_kmh``,
    ``wind_direction_deg``, ``source``, ``is_live``). When wind direction is known
    the downwind risk direction and compass label are derived. Missing weather is
    fine — the incident still assembles. Locations are approximate.
    """
    cx, cy = centroid_norm

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
    return (
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
        "- If the operator asks to confirm or dismiss, tell them to use the 'Confirm alert' or "
        "'Mark as false alarm' button.\n"
        "- Keep replies to a few sentences unless asked for a full draft."
    )


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

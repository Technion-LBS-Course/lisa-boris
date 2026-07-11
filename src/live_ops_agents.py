"""Dual-agent ops chat for the Live Ops dashboard.

Two agents share one chat transcript on the Live tab:

* **Watch** (routine) — posts periodic fire-risk status from weather + configured
  zones (``src/weather.py``).
* **Response** (emergency) — activated on a confirmed detection / DISPATCH: reports
  the incident, recommends actions, answers operator questions, and drafts
  notification messages (``src/incident_agent.py``).

This module is a thin, pure composition over the existing agents — no Streamlit,
no ML. It never sends anything: the Response agent only drafts and recommends
(PyroFinder never contacts anyone or dispatches automatically).
"""

from __future__ import annotations

from src import incident_agent

# Agent identities (shown as a prefix/icon on each chat bubble).
WATCH = "Watch"
RESPONSE = "Response"
AGENT_ICON = {WATCH: "🌤️", RESPONSE: "🚨"}


def _wget(weather, name):
    if weather is None:
        return None
    if isinstance(weather, dict):
        return weather.get(name)
    return getattr(weather, name, None)


def routine_status_text(weather, advisory) -> str:
    """Watch agent's periodic fire-risk status from a Weather + RiskAdvisory."""
    parts: list[str] = []
    if weather is not None:
        bits = []
        t = _wget(weather, "temperature_c")
        h = _wget(weather, "relative_humidity")
        w = _wget(weather, "wind_speed_kmh")
        if t is not None:
            bits.append(f"{t:.0f}°C")
        if h is not None:
            bits.append(f"RH {h:.0f}%")
        if w is not None:
            bits.append(f"wind {w:.0f} km/h")
        source = " (live)" if _wget(weather, "is_live") else " (fallback)"
        if bits:
            parts.append("Conditions: " + ", ".join(bits) + source + ".")
    if advisory is not None:
        parts.append(f"Fire-weather risk: {advisory.level.upper()} (score {advisory.score}).")
        if getattr(advisory, "downwind", None):
            parts.append(f"Downwind risk toward {advisory.downwind}.")
    if not parts:
        return "No weather available for a risk status right now."
    return " ".join(parts)


_ANNOUNCE_PROMPT = (
    "A fire/smoke detection has just been CONFIRMED. Announce it to the site operator in "
    "2-3 short sentences: what was detected, where (zone or area), the confidence, and any "
    "wind/weather risk. End with the single most important next step. Do not invent facts."
)


def emergency_open_text(context) -> str:
    """Announce a confirmed fire/smoke detection in the chat.

    Written by Groq (grounded in the incident facts via the Incident Assistant system
    prompt) when a key is configured; otherwise falls back to the deterministic
    narrative + top recommendation so the announcement always appears.
    """
    if uses_llm():
        try:
            reply = incident_agent.respond_to_operator(context, _ANNOUNCE_PROMPT, history=[])
            if reply and reply.strip():
                return reply.strip()
        except Exception:
            pass
    narrative = incident_agent.incident_narrative(context)
    recommendations = incident_agent.recommend_actions(context)
    top = recommendations[0] if recommendations else ""
    return narrative + (f"\n\nTop recommendation: {top}" if top else "")


def agent_reply(context, message: str, history=None) -> str:
    """Free-form Response reply (Groq when configured, else deterministic)."""
    return incident_agent.respond_to_operator(context, message, history=history or [])


def notification_drafts(context) -> dict[str, str]:
    """Return the audience-keyed notification drafts (nothing is sent)."""
    return incident_agent.build_drafts(context)


def uses_llm() -> bool:
    """Whether Response replies come from Groq (vs the deterministic responder)."""
    return incident_agent.conversation_uses_llm()

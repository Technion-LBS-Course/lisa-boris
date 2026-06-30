"""Groq LLM client helper for operational text (alert summaries, EDA explanations).

Not used for detection — YOLO11s/YOLO11n remain the detectors. The Groq key is
read from st.secrets (Streamlit Cloud / local .streamlit/secrets.toml) with an
environment-variable fallback. The key is never hard-coded.
"""

from __future__ import annotations

import json
import os

# Use the OS certificate store (Windows / Streamlit Cloud) so TLS verification
# works behind networks that intercept HTTPS with their own root CA. Best-effort:
# if truststore is unavailable, fall back to the default certifi bundle.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

import streamlit as st
from groq import Groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _get_api_key() -> str:
    """Return the Groq API key from st.secrets, falling back to the environment."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    key = key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Add it to .streamlit/secrets.toml locally "
            "or to Settings -> Secrets on Streamlit Cloud."
        )
    return key


@st.cache_resource
def get_client() -> Groq:
    """One cached Groq client reused across Streamlit reruns."""
    return Groq(api_key=_get_api_key())


def ask(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a single prompt to Groq and return the reply text."""
    resp = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ── Image-zone structuring (text only — never produces coordinates) ───────────

ZONE_SYSTEM_PROMPT = (
    "You convert a user's free-text description of areas visible in a fixed "
    "camera frame into structured monitoring zones for a fire-detection setup "
    "tool. Return ONLY a JSON object of the form "
    '{"zones": [{"zone_name": str, "zone_type": str, "alert_label": str, '
    '"priority": int, "notes": str}]}.\n'
    "Rules:\n"
    "- One entry per distinct area the user mentions. Do NOT invent areas.\n"
    "- zone_type MUST be exactly one of: {allowed}. Pick the closest; use "
    '"custom" if none fit.\n'
    "- priority is an integer 1-10 (higher = more important). Map words "
    "low->2, medium->5, high->9, critical->10; if the user gives a number use "
    "it; default 5 if unstated.\n"
    "- zone_name: a short human label (e.g. 'White Building (left)').\n"
    "- alert_label: short text for an alert, default same as zone_name.\n"
    "- notes: any extra detail the user gave, else an empty string.\n"
    "- Never output pixel coordinates or polygons; the user draws those."
)


def sanitize_zone_records(raw_zones, allowed_types: list[str]) -> list[dict]:
    """Coerce raw LLM zone dicts into validated zone-detail records.

    Pure function (no network / no Streamlit) so it can be unit-tested. Drops
    entries without a name, maps unknown types to 'custom', and clamps priority
    to 1-10. Never returns geometry.
    """
    cleaned: list[dict] = []
    if not isinstance(raw_zones, list):
        return cleaned
    for z in raw_zones:
        if not isinstance(z, dict):
            continue
        name = str(z.get("zone_name", "")).strip()
        if not name:
            continue
        ztype = str(z.get("zone_type", "custom")).strip()
        if ztype not in allowed_types:
            ztype = "custom"
        try:
            priority = int(z.get("priority", 5))
        except (TypeError, ValueError):
            priority = 5
        priority = max(1, min(10, priority))
        alert_label = str(z.get("alert_label", "")).strip() or name
        notes = str(z.get("notes", "")).strip()
        cleaned.append({
            "zone_name": name,
            "zone_type": ztype,
            "alert_label": alert_label,
            "priority": priority,
            "notes": notes,
        })
    return cleaned


def extract_zones(
    description: str, allowed_types: list[str], model: str = DEFAULT_MODEL
) -> list[dict]:
    """Turn a free-text area description into structured zone records via Groq.

    Returns a list of {zone_name, zone_type, alert_label, priority, notes}.
    Never returns geometry — the user draws each polygon themselves.
    """
    if not description or not description.strip():
        return []
    system = ZONE_SYSTEM_PROMPT.replace("{allowed}", ", ".join(allowed_types))
    resp = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": description.strip()},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    raw_zones = data.get("zones") if isinstance(data, dict) else None
    return sanitize_zone_records(raw_zones, allowed_types)

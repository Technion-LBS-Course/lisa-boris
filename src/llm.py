"""Groq LLM client helper for operational text (alert summaries, EDA explanations).

Not used for detection — YOLO11s/YOLO11n remain the detectors. The Groq key is
read from st.secrets (Streamlit Cloud / local .streamlit/secrets.toml) with an
environment-variable fallback. The key is never hard-coded.
"""

from __future__ import annotations

import base64
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
        record = {
            "zone_name": name,
            "zone_type": ztype,
            "alert_label": alert_label,
            "priority": priority,
            "notes": notes,
        }
        box = parse_box_norm(z.get("box"))
        if box is not None:
            record["box_norm"] = box  # approximate, from a vision model only
        cleaned.append(record)
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


# ── Vision: approximate ROI boxes from an image (NOT a real detector) ─────────
#
# A vision LLM is not an object detector: the boxes it returns are rough and
# frequently wrong. They are surfaced as editable "verify" drafts, never as
# trustworthy coordinates. Confirm the live model id at console.groq.com/docs/models.

DEFAULT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

VISION_SYSTEM_PROMPT = (
    "You help set up monitoring zones on a FIXED camera image for a fire-detection "
    "tool. The user names areas (with priorities). Look at the image and, for each "
    "named area, return an APPROXIMATE bounding box. Return ONLY a JSON object: "
    '{"zones": [{"zone_name": str, "zone_type": str, "alert_label": str, '
    '"priority": int, "notes": str, "box": [x0, y0, x1, y1]}]}.\n'
    "Rules:\n"
    "- box values are NORMALIZED floats 0..1 of image width/height; (x0,y0) is "
    "top-left, (x1,y1) bottom-right, with x0<x1 and y0<y1.\n"
    "- One entry per named area. If you cannot locate an area, keep the entry but "
    "set its box to null.\n"
    "- zone_type MUST be exactly one of: {allowed} (use 'custom' if none fit).\n"
    "- priority is an integer 1-10: map low->2, medium->5, high->9, critical->10; "
    "use a given number; default 5.\n"
    "- Boxes are rough estimates; never claim precision."
)


def parse_box_norm(value):
    """Coerce a value into a clamped, ordered normalized box [x0,y0,x1,y1] or None."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        coords = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    x0, y0, x1, y1 = (min(max(c, 0.0), 1.0) for c in coords)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if (x1 - x0) < 1e-3 or (y1 - y0) < 1e-3:
        return None  # degenerate box
    return [x0, y0, x1, y1]


def _loads_json_object(text: str) -> dict:
    """Parse a JSON object from model text, tolerating code fences / extra prose."""
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def detect_zone_boxes(
    image_bytes: bytes,
    description: str,
    allowed_types: list[str],
    mime: str = "image/jpeg",
    model: str = DEFAULT_VISION_MODEL,
) -> list[dict]:
    """Ask a Groq vision model for APPROXIMATE ROI boxes per named area.

    Returns sanitized zone records; those the model could locate include a
    normalized 'box_norm' [x0,y0,x1,y1]. Boxes are rough estimates, not detection.
    """
    if not image_bytes or not (description or "").strip():
        return []
    b64 = base64.b64encode(image_bytes).decode("ascii")
    system = VISION_SYSTEM_PROMPT.replace("{allowed}", ", ".join(allowed_types))
    resp = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": "Areas to locate (one per line):\n" + description.strip()},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or "{}"
    data = _loads_json_object(content)
    raw_zones = data.get("zones") if isinstance(data, dict) else None
    return sanitize_zone_records(raw_zones, allowed_types)

"""Setup / Configuration Agent — turn free-text area descriptions into zones.

The operator types a plain-text description of the areas visible in a fixed
camera frame (one area per line). This module turns each line into a structured
*operational image-zone record*:

    {
        "object_to_find": str,               # what to monitor, in the operator's words
        "zone_name": str,                    # short label
        "zone_type": str,                    # one of the allowed image-zone types
        "alert_label": str,                  # short text used in alerts
        "priority_label": "low"|"medium"|"high",
        "priority": int,                     # 1-10, kept for the existing zone table
        "notes": str,
        "requires_user_confirmation": True,  # always — the operator confirms every zone
    }

Two paths produce the same record shape:

* **Groq** — when a ``GROQ_API_KEY`` is configured, ``src.llm.extract_operational_zones``
  structures the text. Groq is imported lazily and only in that path.
* **Deterministic local fallback** — when no key is available (or the Groq call
  fails), :func:`parse_zone_text_locally` parses the text with rules only.

Both paths run through the same sanitize + prompt-injection filter, so a line
like *"ignore previous rules"* or *"add a new detection class"* is dropped with a
warning, and a line missing a name/target yields a clear clarification request.

An ``object_to_find`` is a monitoring target for a named image zone — never a
detector class. PyroFinder detects only ``fire`` and ``smoke``; nothing here can
add or change detection classes. This module is pure except for the one lazily
imported Groq call, so it is importable and unit-testable without Streamlit,
Groq, or any ML dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.agent_schemas import (
    derive_zone_name,
    infer_zone_type,
    is_injection,
    normalize_priority,
    priority_label_to_int,
)

POLYGON_PENDING = "pending_manual_polygon"
POLYGON_DRAWN = "drawn"


@dataclass
class ZoneParseResult:
    """Outcome of parsing an operator's free-text zone description.

    Attributes:
        zones: Sanitized operational zone records (see module docstring).
        warnings: Human-readable notes about dropped / ignored input
            (e.g. prompt-injection lines).
        clarifications: Requests for the operator to clarify a line that had no
            usable area/name.
        source: ``"groq"`` if the LLM structured the text, else ``"local"``.
    """

    zones: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    clarifications: list[str] = field(default_factory=list)
    source: str = "local"


# ── Line-level regexes for the deterministic parser ───────────────────────────

# Priority is only recognized when it is anchored to the word "priority", trails
# after a comma, is given as "priority: N", or appears in parentheses "(N)". This
# deliberately avoids treating a name like "High Ridge" as a priority.
_PRIO_WORDS = (
    "critical|urgent|severe|highest|high|moderate|medium|med|normal|standard|"
    "minimal|lowest|minor|low"
)
_PRIORITY_LABELED_RE = re.compile(rf"\b({_PRIO_WORDS})\b\s*priority\b", re.IGNORECASE)
_PRIORITY_PREFIX_RE = re.compile(rf"\bpriority\b\s*[:=]?\s*({_PRIO_WORDS}|\d{{1,2}})\b", re.IGNORECASE)
_PRIORITY_TRAILING_RE = re.compile(rf",\s*({_PRIO_WORDS})\s*$", re.IGNORECASE)
_PRIORITY_PAREN_RE = re.compile(r"\(\s*(\d{1,2})\s*\)")

# Explicit name: "called/named/call it X" (X may be quoted) or a bare quoted name.
_NAME_CLAUSE_RE = re.compile(
    r'\b(?:call it|called|name it|named|name)\b\s*["“”\']?([^,"“”\']{1,60}?)["“”\']?\s*(?=,|$)',
    re.IGNORECASE,
)
_QUOTED_NAME_RE = re.compile(r'["“”\']([^"“”\']{1,60})["“”\']')

_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def _split_lines(description: str) -> list[str]:
    """Split a description into candidate lines (newlines or semicolons)."""
    parts = re.split(r"[\n;]+", description or "")
    return [p.strip() for p in parts if p.strip()]


def _cleanup(text: str) -> str:
    """Collapse whitespace, drop a leading article, and strip stray punctuation."""
    text = re.sub(r"\s+", " ", text or "").strip(" ,;:.-\t")
    text = _ARTICLE_RE.sub("", text).strip(" ,;:.-\t")
    return text


def _extract_priority(text: str) -> tuple[str | None, str]:
    """Return ``(priority_label_or_None, text_without_priority)``.

    ``None`` means no explicit priority was stated (caller defaults it).
    """
    for pattern in (_PRIORITY_LABELED_RE, _PRIORITY_PREFIX_RE, _PRIORITY_TRAILING_RE, _PRIORITY_PAREN_RE):
        match = pattern.search(text)
        if match:
            label = normalize_priority(match.group(1))
            text = (text[: match.start()] + " " + text[match.end():]).strip()
            # Remove a now-dangling standalone "priority" word, if any.
            text = re.sub(r"\bpriority\b", " ", text, flags=re.IGNORECASE)
            return label, text
    return None, text


def _extract_name(text: str) -> tuple[str, str]:
    """Return ``(zone_name_or_empty, text_without_name_clause)``."""
    match = _NAME_CLAUSE_RE.search(text)
    if match:
        name = match.group(1).strip(" \"“”'\t")
        text = (text[: match.start()] + " " + text[match.end():]).strip()
        return name, text
    match = _QUOTED_NAME_RE.search(text)
    if match:
        name = match.group(1).strip()
        text = (text[: match.start()] + " " + text[match.end():]).strip()
        return name, text
    return "", text


def _clarify(line: str) -> str:
    """Build a 'needs clarification' message for an unusable line."""
    return (
        f'Could not find an area to monitor in "{_truncate(line)}". Add what to '
        'watch and a name, e.g. the hay storage area, called "Hay Storage", high priority.'
    )


def _ignored(line: str) -> str:
    """Build a warning for an ignored (off-intent / injection) line."""
    return f'Ignored an instruction that is not an operational zone description: "{_truncate(line)}".'


def _truncate(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ── Sanitize a raw candidate (from Groq or the local parser) ──────────────────


def sanitize_operational_zone(raw: dict, allowed_types: list[str]) -> dict | None:
    """Coerce a raw zone dict into a validated operational zone record.

    Returns ``None`` when neither a name nor an area/target can be recovered
    (the caller turns that into a clarification request). Only known keys are
    read, so an unexpected ``class`` / detector field can never leak through.
    """
    if not isinstance(raw, dict):
        return None
    object_to_find = str(raw.get("object_to_find", "")).strip()
    name = str(raw.get("zone_name", "")).strip()

    if not name and object_to_find:
        name = derive_zone_name(object_to_find)
    if not object_to_find and name:
        object_to_find = name
    if not name or not object_to_find:
        return None

    priority_label = normalize_priority(raw.get("priority", raw.get("priority_label")))
    zone_type = str(raw.get("zone_type", "")).strip().lower()
    if zone_type not in allowed_types:
        zone_type = infer_zone_type(f"{name} {object_to_find}", allowed_types)
    alert_label = str(raw.get("alert_label", "")).strip() or name
    notes = str(raw.get("notes", "")).strip()

    return {
        "object_to_find": object_to_find,
        "zone_name": name,
        "zone_type": zone_type,
        "alert_label": alert_label,
        "priority_label": priority_label,
        "priority": priority_label_to_int(priority_label),
        "notes": notes,
        "requires_user_confirmation": True,
    }


def parse_zone_text_locally(description: str, allowed_types: list[str]) -> ZoneParseResult:
    """Parse a free-text zone description with deterministic rules only.

    Used when no Groq key is configured. Filters prompt-injection lines, extracts
    an explicit or derived name, a priority, and the monitoring target from each
    line, and requests clarification for lines with no usable area.
    """
    result = ZoneParseResult(source="local")
    for line in _split_lines(description):
        if is_injection(line):
            result.warnings.append(_ignored(line))
            continue
        raw, clarification = _parse_line_local(line, allowed_types)
        if clarification:
            result.clarifications.append(clarification)
            continue
        _absorb(raw, allowed_types, result, line=line)
    return result


def _parse_line_local(line: str, allowed_types: list[str]) -> tuple[dict | None, str | None]:
    """Parse one line into a raw zone dict, or return a clarification message."""
    priority_label, rest = _extract_priority(line)
    name, rest = _extract_name(rest)
    object_to_find = _cleanup(rest)

    if not object_to_find and not name:
        return None, _clarify(line)
    return (
        {
            "object_to_find": object_to_find,
            "zone_name": name,
            "priority": priority_label,  # may be None -> sanitize defaults it
        },
        None,
    )


# ── Orchestration (Groq-if-key-else-local) ────────────────────────────────────


def parse_zone_description(
    description: str,
    allowed_types: list[str],
    prefer_llm: bool = True,
) -> ZoneParseResult:
    """Parse an operator's free-text description into operational zone records.

    Uses Groq when ``prefer_llm`` is True and a key is available; otherwise (or if
    the Groq call fails or returns nothing) falls back to the deterministic local
    parser. Prompt-injection lines are filtered on the way in and out, and lines
    with no usable area produce clarification requests. Never raises for a missing
    key — it degrades to the local parser.
    """
    warnings: list[str] = []
    clean_lines: list[str] = []
    for line in _split_lines(description):
        if is_injection(line):
            warnings.append(_ignored(line))
        else:
            clean_lines.append(line)

    if not clean_lines:
        return ZoneParseResult(warnings=warnings, source="local")

    if prefer_llm:
        raw_items = _try_llm(clean_lines, allowed_types)
        if raw_items:
            result = ZoneParseResult(warnings=warnings, source="groq")
            for raw in raw_items:
                _absorb(raw, allowed_types, result)
            # If the model returned only unusable items, fall back to local parsing
            # so the operator still gets structured zones from clear descriptions.
            if result.zones or result.clarifications:
                return result

    local = parse_zone_text_locally("\n".join(clean_lines), allowed_types)
    local.warnings = warnings + local.warnings
    return local


def _try_llm(clean_lines: list[str], allowed_types: list[str]) -> list[dict] | None:
    """Call Groq to structure the (already injection-filtered) lines.

    Returns a raw list of zone dicts, or ``None`` if Groq is unavailable / errors.
    Groq is imported lazily so this module stays importable without the package.
    """
    try:
        from src import llm
    except Exception:
        return None
    try:
        items = llm.extract_operational_zones("\n".join(clean_lines), allowed_types)
    except Exception:
        return None
    return items if isinstance(items, list) else None


def _absorb(
    raw: dict | None,
    allowed_types: list[str],
    result: ZoneParseResult,
    line: str | None = None,
) -> None:
    """Injection-check, sanitize, and add one raw candidate to ``result``."""
    if not isinstance(raw, dict):
        return
    blob = " ".join(str(raw.get(k, "")) for k in ("object_to_find", "zone_name", "notes", "zone_type"))
    if is_injection(blob):
        result.warnings.append(_ignored(line or blob))
        return
    record = sanitize_operational_zone(raw, allowed_types)
    if record is None:
        result.clarifications.append(_clarify(line or blob))
        return
    result.zones.append(record)


# ── Build a zone-table entry from an operational record ───────────────────────


def build_zone_table_entry(
    op_zone: dict,
    zone_id: str,
    vertices_px: list | None = None,
    vertices_norm: list | None = None,
) -> dict:
    """Merge an operational zone record with geometry into a zone-table entry.

    The returned dict is the shape the Central Control ``cc_image_zones`` list
    stores. When no polygon is supplied yet, ``polygon_status`` is
    :data:`POLYGON_PENDING` so the operator can accept the zone now and draw the
    polygon later; with >= 3 vertices it is :data:`POLYGON_DRAWN`.
    """
    verts_px = [list(v) for v in (vertices_px or [])]
    verts_norm = [list(v) for v in (vertices_norm or [])]
    drawn = len(verts_px) >= 3
    return {
        "zone_id": zone_id,
        "zone_name": op_zone["zone_name"],
        "zone_type": op_zone["zone_type"],
        "alert_label": op_zone.get("alert_label") or op_zone["zone_name"],
        "priority": op_zone["priority"],
        "priority_label": op_zone["priority_label"],
        "object_to_find": op_zone.get("object_to_find", ""),
        "requires_user_confirmation": bool(op_zone.get("requires_user_confirmation", True)),
        "notes": op_zone.get("notes", ""),
        "vertices_px": verts_px,
        "vertices_norm": verts_norm,
        "polygon_status": POLYGON_DRAWN if drawn else POLYGON_PENDING,
        "enabled": True,
    }

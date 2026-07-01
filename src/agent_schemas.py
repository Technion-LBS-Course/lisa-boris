"""Shared vocabulary and pure helpers for PyroFinder's operational agents.

This module holds the small, dependency-free building blocks used by the
Setup / Configuration Agent (``src/zone_agent.py``) and, later, the Incident
Assistant and Risk Advisory agents:

* priority normalization between low/medium/high labels and the integer 1-10
  priority the existing Central Control zone table already uses,
* conservative image-zone ``zone_type`` inference from free text,
* a deterministic zone-name derivation, and
* a prompt-injection / off-intent detector.

Everything here is pure (no Streamlit, no network, no ML imports) so it can be
unit-tested directly. An ``object_to_find`` is a *monitoring target for a named
image zone* — it is never a detector class. PyroFinder detects only ``fire`` and
``smoke``; nothing in this module can add, rename, or change detection classes.
"""

from __future__ import annotations

import re

# ── Priority ──────────────────────────────────────────────────────────────────

PRIORITY_LABELS: tuple[str, str, str] = ("low", "medium", "high")
DEFAULT_PRIORITY_LABEL = "medium"

# Label -> integer priority used by the existing image-zone table (1-10). The
# mapping matches the words already documented in the Groq zone system prompt
# (low->2, medium->5, high->9), so AI-assisted and manual zones stay comparable.
PRIORITY_LABEL_TO_INT: dict[str, int] = {"low": 2, "medium": 5, "high": 9}

# Common priority synonyms an operator might type, mapped to the canonical label.
_PRIORITY_WORD_ALIASES: dict[str, str] = {
    "critical": "high", "urgent": "high", "severe": "high", "highest": "high",
    "top": "high", "max": "high", "maximum": "high",
    "moderate": "medium", "med": "medium", "normal": "medium", "standard": "medium",
    "default": "medium", "mid": "medium",
    "minimal": "low", "lowest": "low", "minor": "low", "min": "low",
}


def normalize_priority(value: object) -> str:
    """Return a canonical ``low`` / ``medium`` / ``high`` label for any input.

    Accepts a label, a synonym (``critical`` -> ``high``), or a number (1-3 low,
    4-7 medium, 8-10 high; out-of-range values are clamped). Anything unparseable
    falls back to :data:`DEFAULT_PRIORITY_LABEL`.
    """
    if isinstance(value, bool):  # bool is an int subclass; treat as unset
        return DEFAULT_PRIORITY_LABEL
    if isinstance(value, (int, float)):
        return _int_to_priority_label(value)
    if value is None:
        return DEFAULT_PRIORITY_LABEL
    text = str(value).strip().lower()
    if not text:
        return DEFAULT_PRIORITY_LABEL
    if text in PRIORITY_LABELS:
        return text
    if text in _PRIORITY_WORD_ALIASES:
        return _PRIORITY_WORD_ALIASES[text]
    if text.isdigit():
        return _int_to_priority_label(int(text))
    return DEFAULT_PRIORITY_LABEL


def _int_to_priority_label(number: float) -> str:
    """Map a numeric priority (1-10 scale) to a low/medium/high label."""
    if number <= 3:
        return "low"
    if number >= 8:
        return "high"
    return "medium"


# Public alias — the UI derives a label from a legacy integer priority for display.
int_to_priority_label = _int_to_priority_label


def priority_label_to_int(label: str) -> int:
    """Return the integer (1-10) priority for a low/medium/high label."""
    return PRIORITY_LABEL_TO_INT.get(normalize_priority(label), PRIORITY_LABEL_TO_INT[DEFAULT_PRIORITY_LABEL])


# ── Zone types ──────────────────────────────────────────────────────────────

# The default image-zone types (mirrors ``_ZONE_TYPES`` in the Central Control
# dashboard). Callers pass their own ``allowed_types`` list, so this is only a
# convenience default; inference never returns a type outside ``allowed_types``.
DEFAULT_ZONE_TYPES: list[str] = [
    "barn", "field", "road", "fence", "parking", "forest_edge", "custom",
]

# Ordered keyword map — the first matching, allowed type wins. "grove" appears
# under forest_edge before field on purpose, so "East Grove" reads as a forest
# edge rather than a field.
_ZONE_TYPE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("forest_edge", ("forest", "grove", "wood", "woods", "tree", "trees",
                     "brush", "thicket", "scrub", "bush", "shrub")),
    ("barn", ("barn", "shed", "stable", "coop", "silo", "warehouse", "storehouse",
              "depot", "storage", "store", "hay", "granary", "hangar")),
    ("field", ("field", "pasture", "meadow", "crop", "orchard", "grass", "lawn",
               "paddock", "vineyard", "plot", "acre", "acreage")),
    ("road", ("road", "path", "track", "driveway", "access", "lane", "route",
              "trail", "alley")),
    ("fence", ("fence", "gate", "perimeter", "wall", "boundary", "hedge")),
    ("parking", ("parking", "car park", "carpark", "lot", "carport", "garage")),
]


def infer_zone_type(text: str, allowed_types: list[str]) -> str:
    """Infer an image-zone type from free text, restricted to ``allowed_types``.

    Returns the first keyword-matched type that is allowed. Falls back to
    ``custom`` when present, otherwise the last allowed type.
    """
    blob = f" {(text or '').lower()} "
    for zone_type, keywords in _ZONE_TYPE_KEYWORDS:
        if zone_type in allowed_types and any(kw in blob for kw in keywords):
            return zone_type
    if "custom" in allowed_types:
        return "custom"
    return allowed_types[-1] if allowed_types else "custom"


def derive_zone_name(object_to_find: str) -> str:
    """Derive a short, human zone name from an object description.

    Used only when the operator gives no explicit name. Title-cases the first few
    words and caps the length; the operator confirms it before use anyway.
    """
    words = [w for w in re.sub(r"\s+", " ", object_to_find or "").strip().split(" ") if w]
    name = " ".join(words[:5]).title().strip()
    return name[:40].strip() or "Zone"


# ── Prompt-injection / off-intent detection ─────────────────────────────────
#
# The LLM system prompt is hardened too, but a natural-language prompt is never a
# security boundary. These deterministic patterns filter operator input BEFORE it
# reaches the model, and filter model output AFTER, so an instruction like
# "ignore previous rules" or "add a new detection class" can never take effect.
# Patterns intentionally target meta-instructions (change the rules / classes /
# reveal secrets), NOT ordinary nouns — a zone target that mentions "vehicles" or
# "people" is a legitimate area description, not an attempt to add a class.

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore\s+(all\s+|the\s+|any\s+)?(previous|above|prior|earlier|these|preceding)\b",
        r"\bdisregard\b",
        r"\bforget\s+(everything|all|the\s+above|previous|prior|your)\b",
        r"\bsystem\s+prompt\b",
        r"\boverride\s+(the\s+)?(rules?|instructions?|prompt|system|guardrails?)\b",
        r"(reveal|show|print|expose|leak|dump|give\s+me|tell\s+me)\b.{0,40}\b(secret|secrets|api[\s_-]?key|password|token|credential)",
        r"\bapi[\s_-]?key\b",
        r"\b(developer|admin|god)\s+mode\b",
        r"\bjailbreak\b",
        # Adding / changing detection classes (the one product-scope guardrail).
        r"\b(add|create|introduce|register|train|include)\b.{0,30}\b(new\s+)?(detection\s+)?class(es)?\b",
        r"\b(new|extra|additional|another)\s+(detection\s+)?class(es)?\b",
        r"\b(change|modify|replace|update|redefine|set)\b.{0,30}\b(class(es)?|model|detector|rules?|prompt)\b",
        r"\bdetect\b.{0,20}\binstead of\b.{0,20}\b(fire|smoke)\b",
    )
)


def is_injection(text: str) -> bool:
    """Return True if ``text`` contains a prompt-injection / off-intent instruction."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


# ── Compass / wind direction ─────────────────────────────────────────────────

# 8-point compass, clockwise from North.
_COMPASS_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def compass_label(bearing_deg: float) -> str:
    """Return an 8-point compass label (N, NE, …) for a bearing in degrees."""
    index = int((bearing_deg % 360) / 45.0 + 0.5) % 8
    return _COMPASS_8[index]


def downwind_direction(wind_from_deg: float) -> str:
    """Return the downwind compass label from the meteorological wind-from bearing.

    Wind *from* the north (0°) blows *toward* the south, so downwind = from + 180.
    """
    return compass_label(wind_from_deg + 180.0)

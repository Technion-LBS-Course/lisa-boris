"""Unit tests for src/llm.py zone-record sanitization + Groq availability helpers.

Pure and offline: no network, no real Groq call. `src.llm` must import even when
the optional `groq` package is not installed (it is imported lazily).
"""
import importlib.util

import src.llm as llm
from src.llm import parse_box_norm, sanitize_zone_records

ALLOWED = ["barn", "field", "road", "fence", "parking", "forest_edge", "custom"]


def test_llm_imports_without_groq_and_exposes_vision_model():
    # Importing the module must not require the groq package (lazy import in get_client).
    assert hasattr(llm, "extract_zones")
    assert hasattr(llm, "extract_operational_zones")
    assert hasattr(llm, "detect_zone_boxes")
    assert llm.GROQ_VISION_MODEL == "meta-llama/llama-4-scout-17b-16e-instruct"


def test_groq_available_reflects_package(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert llm.groq_available() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert llm.groq_available() is True


def test_api_key_present_is_controlled(monkeypatch):
    def _no_key():
        raise RuntimeError("GROQ_API_KEY not found")

    monkeypatch.setattr(llm, "_get_api_key", _no_key)
    assert llm.api_key_present() is False
    monkeypatch.setattr(llm, "_get_api_key", lambda: "test-key")
    assert llm.api_key_present() is True


def test_chat_returns_reply_from_client(monkeypatch):
    # Fake the cached Groq client so no package/key/network is needed.
    class _Msg:
        content = "operational reply"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        @staticmethod
        def create(model, messages, temperature=0.3):
            assert messages[0]["role"] == "system"
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(llm, "get_client", lambda: _Client())
    out = llm.chat([{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}])
    assert out == "operational reply"


def test_unknown_type_maps_to_custom():
    out = sanitize_zone_records(
        [{"zone_name": "White Building", "zone_type": "building", "priority": 9}], ALLOWED
    )
    assert len(out) == 1
    assert out[0]["zone_type"] == "custom"
    assert out[0]["zone_name"] == "White Building"
    assert out[0]["priority"] == 9
    assert out[0]["alert_label"] == "White Building"  # defaults to the name


def test_priority_is_clamped_and_defaulted():
    out = sanitize_zone_records(
        [
            {"zone_name": "A", "priority": 99},
            {"zone_name": "B", "priority": 0},
            {"zone_name": "C"},
            {"zone_name": "D", "priority": "high"},  # unparseable -> default
        ],
        ALLOWED,
    )
    assert [z["priority"] for z in out] == [10, 1, 5, 5]


def test_known_type_is_preserved():
    out = sanitize_zone_records(
        [{"zone_name": "East Barn", "zone_type": "barn", "priority": 7}], ALLOWED
    )
    assert out[0]["zone_type"] == "barn"


def test_unnamed_and_non_dict_entries_dropped():
    out = sanitize_zone_records(
        [{"zone_type": "barn"}, "nope", {"zone_name": "   "}, 42], ALLOWED
    )
    assert out == []


def test_non_list_input_returns_empty():
    assert sanitize_zone_records(None, ALLOWED) == []
    assert sanitize_zone_records({"zones": []}, ALLOWED) == []
    assert sanitize_zone_records("text", ALLOWED) == []


def test_text_records_have_no_box():
    out = sanitize_zone_records([{"zone_name": "A", "priority": 5}], ALLOWED)
    assert "box_norm" not in out[0]


def test_valid_box_is_carried_and_clamped():
    out = sanitize_zone_records(
        [{"zone_name": "A", "box": [-0.2, 0.1, 1.5, 0.9]}], ALLOWED
    )
    assert out[0]["box_norm"] == [0.0, 0.1, 1.0, 0.9]


def test_box_corners_are_reordered():
    box = parse_box_norm([0.8, 0.9, 0.2, 0.3])  # bottom-right given first
    assert box == [0.2, 0.3, 0.8, 0.9]


def test_degenerate_or_malformed_box_is_dropped():
    assert parse_box_norm([0.5, 0.5, 0.5, 0.5]) is None  # zero area
    assert parse_box_norm([0.1, 0.2, 0.3]) is None        # wrong length
    assert parse_box_norm(["a", "b", "c", "d"]) is None    # non-numeric
    assert parse_box_norm(None) is None
    out = sanitize_zone_records([{"zone_name": "A", "box": [0.5, 0.5, 0.5, 0.5]}], ALLOWED)
    assert "box_norm" not in out[0]

"""Unit tests for src/llm.py zone-record sanitization + Groq availability helpers.

Pure and offline: no network, no real Groq call. `src.llm` must import even when
the optional `groq` package is not installed (it is imported lazily).
"""
import importlib.util
import os
import subprocess
import sys

import src.llm as llm
from src.llm import parse_box_norm, sanitize_zone_records

ALLOWED = ["barn", "field", "road", "fence", "parking", "forest_edge", "custom"]


def _fake_client_returning(content):
    """A minimal stand-in for the cached Groq client whose one completion returns
    ``content`` as the assistant message — no package, key, or network required."""

    class _Msg:
        def __init__(self):
            self.content = content

    class _Choice:
        def __init__(self):
            self.message = _Msg()

    class _Resp:
        def __init__(self):
            self.choices = [_Choice()]

    class _Completions:
        @staticmethod
        def create(*args, **kwargs):
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


def test_importing_llm_does_not_import_groq():
    # Lazy-import contract: importing src.llm must NOT import the optional `groq`
    # package (it is imported only inside get_client()). Verified in a CLEAN child
    # interpreter so no in-process import state or monkeypatching can mask a
    # regression — if someone moved `from groq import Groq` to module top this fails.
    # groq_available() (find_spec based) must also stay import-free and return a bool.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import sys, importlib; "
        "m = importlib.import_module('src.llm'); "
        "assert isinstance(m.groq_available(), bool); "
        "sys.exit(1 if 'groq' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=repo_root
    )
    assert result.returncode == 0, (
        "importing src.llm must not import the groq package (lazy import expected):\n"
        + result.stderr
    )


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


def test_ask_returns_reply_from_client(monkeypatch):
    # Mirror the chat() fake-client test for the single-prompt ask() wrapper.
    class _Msg:
        content = "single-shot reply"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        @staticmethod
        def create(model, messages):
            # ask() wraps the prompt as one user message; no temperature/response_format.
            assert messages == [{"role": "user", "content": "hello there"}]
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(llm, "get_client", lambda: _Client())
    assert llm.ask("hello there") == "single-shot reply"


# ── extract_zones / extract_operational_zones (parse + sanitize the model JSON) ─


def test_extract_zones_parses_and_sanitizes(monkeypatch):
    content = (
        '{"zones": [{"zone_name": "White Building", "zone_type": "building", "priority": 9},'
        ' {"zone_name": "East Barn", "zone_type": "barn", "priority": 3}]}'
    )
    monkeypatch.setattr(llm, "get_client", lambda: _fake_client_returning(content))
    out = llm.extract_zones("two areas", ALLOWED)
    assert [z["zone_name"] for z in out] == ["White Building", "East Barn"]
    assert out[0]["zone_type"] == "custom"   # unknown 'building' -> custom
    assert out[1]["zone_type"] == "barn"     # known type preserved
    assert out[0]["priority"] == 9
    assert out[0]["alert_label"] == "White Building"  # defaults to the name


def test_extract_zones_malformed_json_returns_empty(monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: _fake_client_returning("this is not json"))
    assert llm.extract_zones("something", ALLOWED) == []


def test_extract_operational_zones_returns_raw_list(monkeypatch):
    content = '{"zones": [{"object_to_find": "hay storage", "zone_name": "Hay Storage", "priority": "high"}]}'
    monkeypatch.setattr(llm, "get_client", lambda: _fake_client_returning(content))
    out = llm.extract_operational_zones("hay storage area", ALLOWED)
    # Unlike extract_zones, this returns the raw list from the JSON (the caller sanitizes).
    assert out == [{"object_to_find": "hay storage", "zone_name": "Hay Storage", "priority": "high"}]


def test_extract_operational_zones_malformed_json_returns_empty(monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: _fake_client_returning("<<not json>>"))
    assert llm.extract_operational_zones("hay storage", ALLOWED) == []


# ── _loads_json_object (tolerant parser: strips code fences / surrounding prose) ─


def test_loads_json_object_strips_code_fence_and_prose():
    fenced = '```json\n{"zones": [{"zone_name": "A"}]}\n```'
    assert llm._loads_json_object(fenced) == {"zones": [{"zone_name": "A"}]}
    prose = 'Sure! Here it is: {"zone_name": "East Grove", "priority": 9} — hope that helps.'
    assert llm._loads_json_object(prose) == {"zone_name": "East Grove", "priority": 9}


def test_loads_json_object_plain_object_and_failures():
    assert llm._loads_json_object('{"a": 1}') == {"a": 1}
    assert llm._loads_json_object("no json here") == {}      # no braces at all
    assert llm._loads_json_object("[1, 2, 3]") == {}          # valid JSON but not an object
    assert llm._loads_json_object("{ not: valid json }") == {}  # braces present but unparseable


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

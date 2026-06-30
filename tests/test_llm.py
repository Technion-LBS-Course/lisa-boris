"""Unit tests for src/llm.py zone-record sanitization (pure, no network)."""
from src.llm import sanitize_zone_records

ALLOWED = ["barn", "field", "road", "fence", "parking", "forest_edge", "custom"]


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

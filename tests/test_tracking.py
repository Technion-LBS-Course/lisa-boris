"""Unit tests for src/tracking.py (pure — no Streamlit, no ML, no network)."""

import pytest

from src.tracking import (
    estimate_apparent_direction,
    is_confirmed_detection,
    is_confirmed_with_tolerance,
)


# ── is_confirmed_with_tolerance ────────────────────────────────────────────────


def test_confirmed_when_exactly_n_frames_all_positive():
    assert is_confirmed_with_tolerance([True, True, True], required_frames=3) is True


def test_confirmed_with_one_missed_frame_in_window():
    assert is_confirmed_with_tolerance([True, False, True, True], required_frames=3) is True


def test_not_confirmed_with_two_missed_frames():
    assert is_confirmed_with_tolerance([True, False, False, True], required_frames=3) is False


def test_not_confirmed_when_current_frame_negative():
    # Even though 3 of the prior frames were positive, the current (last) frame
    # must itself be positive.
    assert is_confirmed_with_tolerance([True, True, True, False], required_frames=3) is False


def test_not_confirmed_with_fewer_than_required_frames():
    assert is_confirmed_with_tolerance([True, True], required_frames=3) is False


def test_not_confirmed_empty_list():
    assert is_confirmed_with_tolerance([], required_frames=3) is False


def test_only_trailing_window_considered_not_full_history():
    # 8 frames total; only the last 4 (required_frames + 1) matter. The window
    # [False, True, True, True] has 3 positives -> confirmed, regardless of the
    # earlier positives outside the window.
    history = [True, True, True, True, False, True, True, True]
    assert is_confirmed_with_tolerance(history, required_frames=3) is True


def test_required_frames_must_be_positive():
    with pytest.raises(ValueError):
        is_confirmed_with_tolerance([True, True, True], required_frames=0)
    with pytest.raises(ValueError):
        is_confirmed_with_tolerance([True, True, True], required_frames=-1)


def test_required_frames_one_needs_only_current_frame():
    assert is_confirmed_with_tolerance([True], required_frames=1) is True
    assert is_confirmed_with_tolerance([False], required_frames=1) is False


# ── is_confirmed_detection (strict N-consecutive confirmation) ─────────────────


def test_strict_confirms_on_unbroken_run():
    assert is_confirmed_detection([True, True, True], required_frames=3) is True
    # Only the trailing window matters, so leading misses are irrelevant.
    assert is_confirmed_detection([False, True, True], required_frames=2) is True


def test_strict_rejects_when_fewer_than_required_frames():
    assert is_confirmed_detection([True, True], required_frames=3) is False


def test_strict_rejects_any_gap_in_the_trailing_window():
    # A miss inside the last N frames breaks strict confirmation.
    assert is_confirmed_detection([True, False, True], required_frames=3) is False
    # Current frame negative also fails.
    assert is_confirmed_detection([True, True, False], required_frames=2) is False


def test_strict_is_less_forgiving_than_tolerant_on_the_same_input():
    # One missed frame inside the window: tolerant confirms, strict does not.
    window = [True, False, True, True]
    assert is_confirmed_with_tolerance(window, required_frames=3) is True
    assert is_confirmed_detection(window, required_frames=3) is False


def test_strict_required_frames_must_be_positive():
    with pytest.raises(ValueError):
        is_confirmed_detection([True, True, True], required_frames=0)
    with pytest.raises(ValueError):
        is_confirmed_detection([True, True, True], required_frames=-2)


# ── estimate_apparent_direction (image-plane movement, y increases downward) ───


def test_direction_stationary_below_threshold():
    # Sub-threshold shift on both axes reads as no movement (noise gating).
    assert estimate_apparent_direction((0.5, 0.5), (0.505, 0.503)) == "stationary"


def test_direction_pure_axis_moves():
    assert estimate_apparent_direction((0.5, 0.5), (0.7, 0.5)) == "right"
    assert estimate_apparent_direction((0.5, 0.5), (0.3, 0.5)) == "left"
    # Image y grows downward: decreasing y is "up", increasing y is "down".
    assert estimate_apparent_direction((0.5, 0.5), (0.5, 0.3)) == "up"
    assert estimate_apparent_direction((0.5, 0.5), (0.5, 0.7)) == "down"


def test_direction_all_four_diagonals():
    assert estimate_apparent_direction((0.5, 0.5), (0.7, 0.3)) == "upper-right"
    assert estimate_apparent_direction((0.5, 0.5), (0.3, 0.3)) == "upper-left"
    assert estimate_apparent_direction((0.5, 0.5), (0.7, 0.7)) == "lower-right"
    assert estimate_apparent_direction((0.5, 0.5), (0.3, 0.7)) == "lower-left"


def test_direction_threshold_is_a_strict_boundary():
    # A shift exactly equal to the threshold does NOT count as movement (uses >).
    # Base from 0.0 so dx is exactly the float literal 0.01 (no rounding drift).
    assert estimate_apparent_direction((0.0, 0.5), (0.01, 0.5), movement_threshold=0.01) == "stationary"
    # Clearly past the threshold does count as movement.
    assert estimate_apparent_direction((0.0, 0.5), (0.02, 0.5), movement_threshold=0.01) == "right"


def test_direction_custom_threshold_gates_larger_shifts():
    # A 0.03 shift is movement under the default threshold but noise under 0.05.
    assert estimate_apparent_direction((0.5, 0.5), (0.53, 0.5)) == "right"
    assert estimate_apparent_direction((0.5, 0.5), (0.53, 0.5), movement_threshold=0.05) == "stationary"

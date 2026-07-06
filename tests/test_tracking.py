"""Unit tests for src/tracking.py (pure — no Streamlit, no ML, no network)."""

import pytest

from src.tracking import is_confirmed_with_tolerance


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

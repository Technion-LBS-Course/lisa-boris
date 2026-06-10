"""Tests for src.evaluation — cost-sensitive operational alert metrics.

These tests are pure-Python and require no ML models, datasets, or image files.
They lock in the operational logic PyroFinder uses to rank models:
    * a missed hazard (false negative) is 10x more costly than a false alert,
    * `fire` and `smoke` both count as "hazard" at the alert level,
    * location metrics use the bbox bottom-center anchor in normalized image space.
"""

import math

import pytest

from src.evaluation import (
    to_hazard_label,
    alert_outcome,
    alert_outcome_from_flags,
    compute_alert_confusion,
    alert_confusion_from_confusion_matrix,
    compute_operational_alert_metrics,
    operational_alert_metrics_from_confusion_matrix,
    bbox_bottom_center,
    grid_cell,
    same_grid_cell,
    normalized_euclidean_distance,
    bbox_iou,
    best_iou_fire_match,
    fire_location_error,
    DEFAULT_FN_WEIGHT,
    DEFAULT_FP_WEIGHT,
)


# ── to_hazard_label ────────────────────────────────────────────────────────────

def test_to_hazard_label_fire_and_smoke_are_hazards():
    assert to_hazard_label("fire") is True
    assert to_hazard_label("smoke") is True


def test_to_hazard_label_background_is_not_hazard():
    assert to_hazard_label("background") is False
    assert to_hazard_label("") is False
    assert to_hazard_label(None) is False


def test_to_hazard_label_is_case_insensitive():
    assert to_hazard_label("Fire") is True
    assert to_hazard_label(" SMOKE ") is True


# ── alert_outcome ───────────────────────────────────────────────────────────────

def test_alert_outcome_from_labels():
    assert alert_outcome("fire", "fire") == "TP"
    assert alert_outcome("smoke", "background") == "FN"      # missed hazard
    assert alert_outcome("background", "fire") == "FP"       # false alert
    assert alert_outcome("background", "background") == "TN"


def test_alert_outcome_from_flags():
    assert alert_outcome_from_flags(True, True) == "TP"
    assert alert_outcome_from_flags(True, False) == "FN"
    assert alert_outcome_from_flags(False, True) == "FP"
    assert alert_outcome_from_flags(False, False) == "TN"


# ── compute_alert_confusion ────────────────────────────────────────────────────

def test_alert_confusion_with_background_fire_smoke_labels():
    # true vs pred at the image/alert level
    y_true = ["background", "fire", "smoke", "background", "fire", "smoke"]
    y_pred = ["background", "fire", "background", "fire", "smoke", "smoke"]
    conf = compute_alert_confusion(y_true, y_pred)
    # hazard present: indices 1,2,4,5 ; background: 0,3
    #   idx1 fire->fire      = TP
    #   idx2 smoke->background = FN (missed hazard)
    #   idx4 fire->smoke     = TP (smoke still counts as hazard detected)
    #   idx5 smoke->smoke    = TP
    #   idx0 background->background = TN
    #   idx3 background->fire = FP (false alert)
    assert conf["tp_alert"] == 3
    assert conf["fn_alert"] == 1
    assert conf["fp_alert"] == 1
    assert conf["tn_alert"] == 1
    assert conf["total_hazard_cases"] == 4
    assert conf["total_background_cases"] == 2


def test_alert_confusion_smoke_counts_as_hazard():
    # smoke-only prediction on smoke-only ground truth is a detected hazard (TP)
    conf = compute_alert_confusion(["smoke"], ["smoke"])
    assert conf["tp_alert"] == 1 and conf["fn_alert"] == 0


def test_alert_confusion_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_alert_confusion(["fire"], ["fire", "smoke"])


def test_alert_confusion_from_confusion_matrix_matches_pairwise():
    # sklearn-style matrix, rows=true, cols=pred, labels=[background, fire, smoke]
    labels = ["background", "fire", "smoke"]
    matrix = [
        [50, 3, 2],   # background truth: 50 TN, 5 FP
        [4, 30, 6],   # fire truth: 4 FN, 36 TP
        [5, 1, 40],   # smoke truth: 5 FN, 41 TP
    ]
    conf = alert_confusion_from_confusion_matrix(matrix, labels)
    assert conf["tn_alert"] == 50
    assert conf["fp_alert"] == 5
    assert conf["fn_alert"] == 9          # 4 + 5
    assert conf["tp_alert"] == 36 + 41    # 77
    assert conf["total_hazard_cases"] == 86
    assert conf["total_background_cases"] == 55


# ── cost weighting: FN penalty is 10x FP penalty ───────────────────────────────

def test_false_negative_penalty_is_ten_times_false_positive():
    assert DEFAULT_FN_WEIGHT == 10
    assert DEFAULT_FP_WEIGHT == 1

    # One missed hazard (FN) vs one false alert (FP), same dataset size.
    one_missed = compute_operational_alert_metrics(
        ["fire", "background"], ["background", "background"]
    )
    one_false_alert = compute_operational_alert_metrics(
        ["fire", "background"], ["fire", "fire"]
    )
    assert one_missed["weighted_error_cost"] == 10   # 10 * 1 FN
    assert one_false_alert["weighted_error_cost"] == 1  # 1 * 1 FP
    assert one_missed["weighted_error_cost"] == 10 * one_false_alert["weighted_error_cost"]


def test_weighted_error_cost_formula():
    # 2 FN and 3 FP -> 10*2 + 1*3 = 23
    y_true = ["fire", "smoke", "background", "background", "background", "smoke"]
    y_pred = ["background", "background", "fire", "fire", "fire", "smoke"]
    m = compute_operational_alert_metrics(y_true, y_pred)
    assert m["fn_alert"] == 2
    assert m["fp_alert"] == 3
    assert m["weighted_error_cost"] == 10 * 2 + 1 * 3


# ── operational_alert_score: missed hazard hurts more than a false alert ────────

def test_operational_score_drops_more_for_missed_hazard_than_false_alert():
    base_true = ["fire", "fire", "background", "background"]

    perfect = compute_operational_alert_metrics(base_true, base_true)
    missed = compute_operational_alert_metrics(
        base_true, ["background", "fire", "background", "background"]
    )
    false_alert = compute_operational_alert_metrics(
        base_true, ["fire", "fire", "fire", "background"]
    )

    assert perfect["operational_alert_score"] == 1.0
    # A single missed hazard pulls the score down further than a single false alert.
    assert missed["operational_alert_score"] < false_alert["operational_alert_score"]
    # The underlying cost penalty is exactly 10x (FN weight 10 vs FP weight 1).
    assert missed["weighted_error_cost"] == 10 * false_alert["weighted_error_cost"]
    # Same ground truth -> same denominator, so the score drop is ~10x as large
    # (abs tolerance absorbs 4-decimal rounding of the score).
    drop_missed = perfect["operational_alert_score"] - missed["operational_alert_score"]
    drop_false = perfect["operational_alert_score"] - false_alert["operational_alert_score"]
    assert drop_missed == pytest.approx(10 * drop_false, abs=2e-3)


def test_operational_score_bounds_and_perfect_and_worst():
    # Perfect predictions -> score 1.0
    perfect = compute_operational_alert_metrics(["fire", "background"], ["fire", "background"])
    assert perfect["operational_alert_score"] == 1.0
    assert perfect["hazard_recall"] == 1.0
    assert perfect["false_alert_rate"] == 0.0

    # Worst case: every hazard missed AND every background false-alarmed -> score 0.0
    worst = compute_operational_alert_metrics(
        ["fire", "smoke", "background"], ["background", "background", "fire"]
    )
    assert worst["operational_alert_score"] == 0.0
    assert worst["hazard_recall"] == 0.0
    assert worst["false_alert_rate"] == 1.0


def test_hazard_recall_and_precision_values():
    # 3 hazards, 2 detected -> recall 2/3 ; 2 TP + 1 FP -> precision 2/3
    y_true = ["fire", "smoke", "fire", "background"]
    y_pred = ["fire", "smoke", "background", "fire"]
    m = compute_operational_alert_metrics(y_true, y_pred)
    assert m["tp_alert"] == 2 and m["fn_alert"] == 1 and m["fp_alert"] == 1
    assert m["hazard_recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["alert_precision"] == pytest.approx(2 / 3, abs=1e-4)
    # the single background case was false-alarmed -> false_alert_rate = 1/1
    assert m["false_alert_rate"] == 1.0


def test_no_cases_gives_vacuous_perfect_score():
    m = compute_operational_alert_metrics([], [])
    assert m["operational_alert_score"] == 1.0
    assert m["weighted_error_cost"] == 0
    assert m["max_possible_cost"] == 0


def test_custom_weights_change_cost():
    y_true = ["fire", "background"]
    y_pred = ["background", "fire"]  # 1 FN, 1 FP
    m = compute_operational_alert_metrics(y_true, y_pred, fn_weight=5, fp_weight=2)
    assert m["weighted_error_cost"] == 5 * 1 + 2 * 1
    assert m["fn_weight"] == 5 and m["fp_weight"] == 2


def test_operational_metrics_from_confusion_matrix_roundtrip():
    labels = ["background", "fire", "smoke"]
    matrix = [
        [10, 2, 0],
        [1, 8, 1],
        [0, 0, 9],
    ]
    from_matrix = operational_alert_metrics_from_confusion_matrix(matrix, labels)
    # Equivalent flat label sequences for the same matrix:
    y_true, y_pred = [], []
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            y_true += [t] * matrix[i][j]
            y_pred += [p] * matrix[i][j]
    from_labels = compute_operational_alert_metrics(y_true, y_pred)
    assert from_matrix == from_labels


# ── bbox_bottom_center ──────────────────────────────────────────────────────────

def test_bbox_bottom_center_anchor():
    # box centered at (0.5, 0.5), height 0.4 -> bottom edge at y = 0.7
    anchor = bbox_bottom_center(0.5, 0.5, 0.2, 0.4)
    assert anchor == (0.5, 0.7)


def test_bbox_bottom_center_keeps_x():
    anchor = bbox_bottom_center(0.25, 0.3, 0.1, 0.2)
    assert anchor[0] == 0.25
    assert anchor[1] == pytest.approx(0.4)


# ── grid_cell / same_grid_cell ──────────────────────────────────────────────────

def test_grid_cell_default_three_by_three():
    assert grid_cell(0.1, 0.1) == (0, 0)   # top-left
    assert grid_cell(0.5, 0.5) == (1, 1)   # center
    assert grid_cell(0.9, 0.9) == (2, 2)   # bottom-right


def test_grid_cell_clamps_border():
    # exactly 1.0 should map to the last cell, not overflow to index 3
    assert grid_cell(1.0, 1.0) == (2, 2)
    assert grid_cell(-0.2, 0.5) == (0, 1)


def test_same_grid_cell_true_and_false():
    assert same_grid_cell((0.1, 0.1), (0.2, 0.2)) is True     # both top-left
    assert same_grid_cell((0.1, 0.1), (0.9, 0.9)) is False    # opposite corners


def test_grid_cell_custom_size():
    assert grid_cell(0.5, 0.5, grid_size=2) == (1, 1)
    assert grid_cell(0.4, 0.4, grid_size=2) == (0, 0)


# ── normalized_euclidean_distance ────────────────────────────────────────────────

def test_normalized_distance_zero_for_identical_points():
    assert normalized_euclidean_distance((0.5, 0.5), (0.5, 0.5)) == 0.0


def test_normalized_distance_345_triangle():
    # (0,0) -> (0.3, 0.4) is a classic 3-4-5 triangle scaled -> 0.5
    assert normalized_euclidean_distance((0.0, 0.0), (0.3, 0.4)) == pytest.approx(0.5)


def test_normalized_distance_opposite_corners_is_sqrt2():
    assert normalized_euclidean_distance((0.0, 0.0), (1.0, 1.0)) == pytest.approx(math.sqrt(2))


# ── bbox_iou / best_iou_fire_match / fire_location_error ─────────────────────────

def test_bbox_iou_identical_boxes_is_one():
    box = (0.5, 0.5, 0.2, 0.2)
    assert bbox_iou(box, box) == pytest.approx(1.0)


def test_bbox_iou_disjoint_boxes_is_zero():
    assert bbox_iou((0.1, 0.1, 0.1, 0.1), (0.9, 0.9, 0.1, 0.1)) == 0.0


def test_best_iou_fire_match_picks_highest_overlap():
    gt = [(0.5, 0.5, 0.2, 0.2)]
    preds = [(0.9, 0.9, 0.2, 0.2), (0.51, 0.51, 0.2, 0.2)]
    match = best_iou_fire_match(gt, preds)
    assert match is not None
    _, pred_box, iou = match
    assert pred_box == (0.51, 0.51, 0.2, 0.2)
    assert iou > 0.5


def test_best_iou_fire_match_none_when_empty():
    assert best_iou_fire_match([], [(0.5, 0.5, 0.2, 0.2)]) is None
    assert best_iou_fire_match([(0.5, 0.5, 0.2, 0.2)], []) is None


def test_fire_location_error_same_box_zero_error_same_cell():
    box = (0.5, 0.5, 0.2, 0.2)
    result = fire_location_error([box], [box])
    assert result is not None
    assert result["error"] == 0.0
    assert result["grid_hit"] is True
    assert result["iou"] == pytest.approx(1.0)


def test_fire_location_error_none_when_no_predicted_fire():
    # GT has fire but the detector predicted no fire box -> no numeric location error
    assert fire_location_error([(0.5, 0.5, 0.2, 0.2)], []) is None

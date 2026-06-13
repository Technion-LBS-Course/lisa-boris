# M3 Results Summary — YOLO11s vs YOLO11n

## 1. Purpose

This document summarizes the **measured M3 detector comparison** for PyroFinder: the fine-tuned **YOLO11s** primary detector versus the fine-tuned **YOLO11n** lightweight baseline/fallback, evaluated on the D-Fire held-out test split. It records the object-detection metrics, the cost-sensitive operational alert metrics, the failure breakdown by hazard subtype, the approximate fire-location comparison, and the justification for the selected detector.

All numbers come from the committed result files (`results/baseline_yolo11s.json`, `results/baseline_yolo11n.json`, `results/yolo11s_operational_metrics.json`, `results/yolo11n_operational_metrics.json`, `results/yolo11s_test_predictions.csv`, `results/yolo11n_test_predictions.csv`) and the completed YOLO11s-vs-YOLO11n failure analysis. No metric is recomputed or invented here.

## 2. Evaluation setup

- **Dataset:** D-Fire held-out **test split — 4,306 images** (train 17,221 / test 4,306; class mapping 0 = smoke, 1 = fire).
- **Detectors:** YOLO11s and YOLO11n evaluated on the **same** test split under equivalent conditions.
- **Confidence threshold:** 0.25.
- **Image size:** 640.
- **Operational evaluation is evaluation-only — no training or retraining occurred.** The fine-tuned checkpoints were run on the test split and reduced to image-level alert outcomes.
- **Classes:** strictly `fire` and `smoke`.
- **Location metrics are approximate image-space metrics only — never precise geolocation.**
- D-Fire uses a two-way train/test split, with the `test` folder mapped to Ultralytics' `val` role for evaluation; there is no separate development split.

## 3. Metric families

Three metric families are reported and kept **separate**. They answer different questions and must not be merged into a single ranking.

- **Object-detection metrics:** mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1 — measure bounding-box and class-detection quality.
- **Operational alert metrics:** Hazard Recall, False Alert Rate, Operational Alert Score (plus alert-level TP/FN/FP/TN and weighted error cost) — measure whether the system raises or misses a hazard-level alert, with a missed hazard weighted 10× a false alert.
- **Approximate fire-location metrics:** coverage count/rate, mean/median location error, P90/P95, max error, and 3×3 grid-hit rate — measure approximate image-space fire location for object detectors only.

Explicitly:

- **Do not compare sklearn Macro F1 directly with YOLO mAP.** They measure different tasks at different granularities (image-level classification vs object detection).
- **Detection metrics and operational alert metrics answer different questions** and are never treated as interchangeable.

## 4. Object-detection comparison

Object-detection quality on the D-Fire test split (boxes and classes; never compared against sklearn accuracy or Macro F1):

| Metric | YOLO11s | YOLO11n | Δ (11s − 11n) |
|---|--:|--:|--:|
| mAP@0.5 | 0.7668 | 0.7470 | +0.0198 |
| mAP@0.5:0.95 | 0.4414 | 0.4249 | +0.0165 |
| Precision | 0.7573 | 0.7397 | +0.0176 |
| Recall | 0.6967 | 0.6825 | +0.0142 |
| F1 | 0.7257 | 0.7099 | +0.0158 |

YOLO11s shows a **clearer improvement on detection quality** than on the operational layer, most visibly a **+1.98 pp gain in mAP@0.5** (0.7470 → 0.7668), with consistent gains across all five detection metrics. Per-class detection metrics for YOLO11s are: smoke — mAP@0.5 0.8222, mAP@0.5:0.95 0.5054, Precision 0.8028, Recall 0.7563, F1 0.7789; fire — mAP@0.5 0.7115, mAP@0.5:0.95 0.3774, Precision 0.7119, Recall 0.6370, F1 0.6724.

## 5. Operational alert comparison

Cost-sensitive alert-level evaluation (`fire` or `smoke` = hazard; FN weight 10, FP weight 1; confidence 0.25; 4,306 images):

| Metric | YOLO11s | YOLO11n | Δ (11s − 11n) |
|---|--:|--:|--:|
| TP (hazard caught) | 2,156 | 2,147 | +9 |
| FN (hazard missed) | 145 | 154 | −9 |
| FP (false alert) | 37 | 42 | −5 |
| TN | 1,968 | 1,963 | +5 |
| Hazard Recall | 0.9370 | 0.9331 | +0.0039 |
| False Alert Rate | 0.0185 | 0.0209 | −0.0024 |
| Operational Alert Score | 0.9406 | 0.9368 | +0.0038 |
| Weighted error cost (10×FN + FP) | 1,487 | 1,582 | −95 |

YOLO11s improves the operational metrics, but the **margin is modest/incremental**:

- **9 fewer missed hazards** (145 vs 154).
- **5 fewer false alerts** (37 vs 42).
- **+0.0038 Operational Alert Score** (0.9406 vs 0.9368), and a 95-point (≈6%) reduction in weighted error cost.

## 6. Failure breakdown by subtype

Each test image is one of four ground-truth subtypes. Subtype recall = TP / (TP + FN); background uses TN / FP:

| Subtype | GT images | YOLO11s TP / FN | YOLO11s recall | YOLO11n TP / FN | YOLO11n recall |
|---|--:|--:|--:|--:|--:|
| smoke-only | 1,186 | 1,057 / 129 | 0.8912 | 1,053 / 133 | 0.8879 |
| fire-only | 220 | 211 / 9 | 0.9591 | 205 / 15 | 0.9318 |
| fire+smoke | 895 | 888 / 7 | 0.9922 | 889 / 6 | 0.9933 |
| background | 2,005 | 1,968 TN / 37 FP | FAR 0.0185 | 1,963 TN / 42 FP | FAR 0.0209 |

Observations:

- **smoke-only is the dominant failure mode for both models.** It carries the overwhelming majority of all missed hazards (YOLO11s 129/145 ≈ 89%; YOLO11n 133/154 ≈ 86%) and the lowest subtype recall (~0.89, versus ~0.96–0.99 for fire-bearing images). This is consistent with the known D-Fire bias that smoke can resemble cloud, fog, haze, dust, or bright sky.
- **YOLO11s improves fire-only recall meaningfully** (0.9318 → 0.9591; 15 → 9 misses). Fire-only misses are operationally important because visible fire is a direct hazard signal.
- **fire+smoke is almost saturated for both models** (recall ≈ 0.99; the two models differ by a single image and are effectively tied).
- **YOLO11s slightly reduces background false alerts** (37 vs 42 false positives on 2,005 background images).

## 7. Paired disagreement analysis

Comparing the two detectors image-by-image on the same 4,306 test images shows that **YOLO11s is not simply a superset of YOLO11n**:

- The models share only **90 common false negatives**.
- YOLO11s **catches 64 hazards that YOLO11n misses**, but also **misses 55 hazards that YOLO11n catches** — a 64/55 trade, not a clean improvement on every image.
- The same pattern holds for false alerts: YOLO11s fixes 23 of YOLO11n's false positives but introduces 18 new ones.
- **The net result still favors YOLO11s** (net −9 false negatives, net −5 false positives).

Because the errors are partly complementary rather than nested, this **suggests future threshold-tuning and complementary-model (e.g. ensemble) analysis**. That work is **not part of M3** — it is noted here only as a direction for later milestones.

## 8. Approximate fire-location comparison

Computed only on images that have both ground-truth fire and predicted fire boxes (class-1 fire, bottom-center anchor `anchor_x = x_center`, `anchor_y = y_center + height/2`, matched by best IoU per image). These are **approximate image-space estimates, never precise geolocation**:

| Location metric | YOLO11s | YOLO11n |
|---|--:|--:|
| GT-fire images | 1,115 | 1,115 |
| Coverage count / rate | 1,040 / 0.9327 | 1,020 / 0.9148 |
| Mean error (normalized) | 0.01350 | 0.01343 |
| Median error (normalized) | 0.00548 | 0.00570 |
| P90 / P95 error | 0.0263 / 0.0524 | 0.0289 / 0.0530 |
| Max error (outlier) | 0.9119 | 0.5206 |
| 3×3 grid-hit rate | 0.9644 | 0.9559 |

Careful wording:

- **Location central-tendency metrics are practically tied** (mean and median differ by ~0.0001–0.0002 in normalized units).
- **YOLO11s has slightly better coverage and grid-hit rate** (1,040 vs 1,020 images localized; 0.9644 vs 0.9559 grid-hit), and a marginally tighter P90/P95 tail.
- **The large YOLO11s outlier (max error ≈ 0.91) likely indicates a wrong fire-box match or a difficult image and should be reviewed visually if time allows.** It does not move the aggregate statistics.
- No location output is a precise geographic coordinate.

## 9. Selected detector justification

**YOLO11s remains the selected detector.** It wins by the documented operational selection hierarchy:

1. higher **Hazard Recall** (0.9370 vs 0.9331),
2. lower **False Alert Rate** (0.0185 vs 0.0209),
3. higher **Operational Alert Score** (0.9406 vs 0.9368),
4. stronger supporting **detection metrics** (mAP@0.5 0.7668 vs 0.7470; Recall 0.6967 vs 0.6825).

The **operational lead is small, but consistent** across every step of the hierarchy — YOLO11s is never behind on any metric or tiebreak. The **detection-quality lead is clearer** (≈ +2 pp mAP@0.5). **YOLO11n remains the lightweight baseline/fallback**, not an equal parallel model.

> On the D-Fire held-out test set, YOLO11s is selected over YOLO11n: it improves Hazard Recall to 0.9370 versus 0.9331, lowers False Alert Rate to 0.0185 versus 0.0209, and raises Operational Alert Score to 0.9406 versus 0.9368, while also improving mAP@0.5 from 0.7470 to 0.7668. The operational improvement is modest, but it is consistent across the selection hierarchy; the clearest remaining weakness for both detectors is smoke-only imagery.

## 10. M4 / future work notes

- Improve **smoke-only recall**, the dominant failure mode for both detectors.
- Revisit **confidence-threshold tuning** (the smoke-recall vs false-alert-rate trade-off; all numbers here are at conf 0.25).
- **Visual review of the worst approximate fire-location outliers** (e.g. the YOLO11s max-error case).
- Possible **complementary-model analysis** given the partly non-overlapping errors between detectors.
- **Validation beyond D-Fire** on supplementary/robustness datasets and private-property camera angles.

---

*Guardrails: PyroFinder is not an early warning system. All location outputs are approximate image-space estimates, never precise geolocation. No emergency dispatch integration and no physical fire-spread prediction are claimed. Model-specific statements use `YOLO11s` and `YOLO11n` explicitly.*

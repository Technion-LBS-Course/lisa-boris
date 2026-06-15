# Hazard-Level Confusion Matrices — All Models

## Definition

**Hazard** = any fire or/and smoke is present in the image (regardless of count or combination).  
Binary reduction: `fire` or `smoke` → **Hazard** (Positive); empty/background → **No Hazard** (Negative).

**Dataset:** D-Fire held-out test split — **4,306 images total**  
- Hazard images: **2,301** (fire-only 220 · smoke-only 1,186 · fire+smoke 895)  
- Background images: **2,005**

---

## Evaluation Objective — Asymmetric Cost Matrix

The core goal is to **minimize two error types simultaneously, with unequal importance**:

| Error | Meaning | Cost |
|---|---|--:|
| **False Negative (FN)** | Predicted *no hazard* — actual *hazard* (missed fire/smoke) | **10** |
| **False Positive (FP)** | Predicted *hazard* — actual *no hazard* (false alarm) | **1** |

A missed hazard is **10× more costly** than a false alarm — the asymmetric cost reflects the real-world consequence of failing to detect a fire outbreak.

This is formally called **cost-sensitive (asymmetric-cost) evaluation**. The cost matrix is:

|  | Predicted: Hazard | Predicted: No Hazard |
|---|:---:|:---:|
| **Actual: Hazard** | 0 (TP — correct alert) | **10** (FN — missed hazard) |
| **Actual: No Hazard** | **1** (FP — false alert) | 0 (TN — correct silence) |

**Objective:** minimize the **expected cost**:

```
Loss = 10 · FN + 1 · FP
```

**Operational Alert Score (OAS)** is the normalized form of this objective — maps Loss to [0, 1] where 1 = perfect and 0 = worst possible:

```
OAS = 1 − Loss / Loss_max
    = 1 − (10·FN + FP) / (10·N_pos + N_neg)
    = 1 − (10·FN + FP) / (10·2301 + 2005)
```

where `Loss_max = 10·N_pos + N_neg` is the cost if every hazard is missed **and** every background image raises a false alert.

Equivalently, OAS is a **cost-weighted balanced accuracy** — a weighted average of Hazard Recall (True Positive Rate) and Specificity (True Negative Rate), with the cost ratio as the weight:

```
OAS = (10 · HazardRecall · N_pos + Specificity · N_neg) / (10 · N_pos + N_neg)
```

where `HazardRecall = TP / N_pos` and `Specificity = TN / N_neg = 1 − False Alert Rate`.

> Note: sklearn classifiers (DummyClassifier, Logistic Regression, Random Forest) are image-level classifiers — they produce no bounding boxes. Their hazard confusion matrix is derived from the 3-class prediction (fire/smoke = hazard, background = no hazard). YOLO11n and YOLO11s are object detectors — any fire or smoke box at confidence ≥ 0.25 triggers a hazard alert.

---

## Summary Comparison

| Model | TP | FN | FP | TN | Hazard Recall | False Alert Rate | Op. Alert Score |
|---|--:|--:|--:|--:|--:|--:|--:|
| DummyClassifier | 0 | 2,301 | 0 | 2,005 | 0.0000 | 0.0000 | 0.0802 |
| Logistic Regression | 1,854 | 447 | 1,010 | 995 | 0.8057 | 0.5037 | 0.7809 |
| Random Forest | 2,020 | 281 | 166 | 1,839 | 0.8779 | 0.0828 | 0.8810 |
| YOLO11n *(baseline)* | 2,147 | 154 | 42 | 1,963 | 0.9331 | 0.0209 | 0.9368 |
| **YOLO11s *(primary)*** | **2,156** | **145** | **37** | **1,968** | **0.9370** | **0.0185** | **0.9406** |

---

## Per-Model Confusion Matrices

### DummyClassifier (most_frequent)

Always predicts the majority class (background) — never raises a hazard alert.

|  | **Predicted: Hazard** | **Predicted: No Hazard** |
|---|--:|--:|
| **Actual: Hazard** | TP = 0 | FN = 2,301 |
| **Actual: No Hazard** | FP = 0 | TN = 2,005 |

- Hazard Recall: 0.0000 — misses every hazard image
- False Alert Rate: 0.0000 — never raises a false alert
- Operational Alert Score: **0.0802** — effectively the minimum bar; all 2,301 hazards missed at 10× cost

---

### Logistic Regression (color features)

Image-level classifier; 60-dim color feature vector; balanced class weights.

|  | **Predicted: Hazard** | **Predicted: No Hazard** |
|---|--:|--:|
| **Actual: Hazard** | TP = 1,854 | FN = 447 |
| **Actual: No Hazard** | FP = 1,010 | TN = 995 |

- Hazard Recall: 0.8057 — catches most hazards
- False Alert Rate: **0.5037** — fires a false alert on more than half of all background images
- Operational Alert Score: **0.7809** — strong recall but very poor precision; excessive false alerts dominate the cost

---

### Random Forest (color features)

Image-level classifier; 200 trees; 60-dim color feature vector; balanced class weights.

|  | **Predicted: Hazard** | **Predicted: No Hazard** |
|---|--:|--:|
| **Actual: Hazard** | TP = 2,020 | FN = 281 |
| **Actual: No Hazard** | FP = 166 | TN = 1,839 |

- Hazard Recall: 0.8779 — best recall among classical classifiers
- False Alert Rate: 0.0828 — roughly 4× higher than YOLO detectors
- Operational Alert Score: **0.8810** — large step up from Logistic Regression; best classical baseline

---

### YOLO11n — fine-tuned D-Fire (baseline detector)

Object detector; confidence ≥ 0.25; any fire or smoke box = hazard alert.

|  | **Predicted: Hazard** | **Predicted: No Hazard** |
|---|--:|--:|
| **Actual: Hazard** | TP = 2,147 | FN = 154 |
| **Actual: No Hazard** | FP = 42 | TN = 1,963 |

- Hazard Recall: 0.9331
- False Alert Rate: 0.0209
- Operational Alert Score: **0.9368**

---

### YOLO11s — fine-tuned D-Fire (primary detector, selected)

Object detector; confidence ≥ 0.25; any fire or smoke box = hazard alert.

|  | **Predicted: Hazard** | **Predicted: No Hazard** |
|---|--:|--:|
| **Actual: Hazard** | TP = 2,156 | FN = 145 |
| **Actual: No Hazard** | FP = 37 | TN = 1,968 |

- Hazard Recall: **0.9370** — 9 fewer missed hazards than YOLO11n
- False Alert Rate: **0.0185** — 5 fewer false alerts than YOLO11n
- Operational Alert Score: **0.9406** — highest across all five models

---

## Key Observations

1. **YOLO detectors vs classical classifiers:** both YOLO models far outperform all sklearn classifiers on the operational metrics. The gap is largest in False Alert Rate — Logistic Regression raises false alerts on 50% of background images versus 2% for YOLO11n/11s.

2. **YOLO11s vs YOLO11n:** YOLO11s wins on every metric — 9 fewer missed hazards, 5 fewer false alerts, +0.0038 Operational Alert Score. The margin is small but consistent across the entire selection hierarchy.

3. **DummyClassifier floor:** an Operational Alert Score of 0.0802 reflects the baseline of always predicting "no hazard." Any real model must substantially exceed this.

4. **Dominant failure mode for YOLO models:** smoke-only images drive ~89% of all false negatives (YOLO11s 129/145 ≈ 89%; YOLO11n 133/154 ≈ 86%). This is the primary target for future improvement.

---

*All values are from committed result files: `results/baseline_*.json`, `results/yolo11n_operational_metrics.json`, `results/yolo11s_operational_metrics.json`. No metric is recomputed or invented here. Location outputs are approximate image-space estimates, never precise geolocation.*

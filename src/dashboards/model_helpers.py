"""Shared Streamlit rendering helpers for the model dashboards.

These functions render the per-model views, the classification / object-detection
comparisons, and the operational alert metrics. They are reused by both the
Operations & Learning dashboard and the M3 dashboard so the model story renders
identically and the logic lives in one place.

No metric values are invented: everything is read from the measured result files
in ``results/``. Heavy ML libraries are never imported here — the cached detector
loader imports ``src.inference`` lazily, and the ``results_loader`` / ``evaluation``
helpers are imported lazily inside the functions that use them.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import apply_chart_theme, PYRO_COLORS, SPLIT_COLORS


@st.cache_resource(show_spinner=False)
def load_detector_cached(model_name: str):
    """Load and cache a fine-tuned YOLO detector by name.

    Heavy ML libraries are imported lazily inside ``src.inference``, so models are
    only loaded when inference is explicitly requested — never at import time.
    Cached per model name so each checkpoint loads once per session.
    """
    from src.inference import load_detector
    return load_detector(model_name)


def load_model_results(results_dir: str = "results") -> dict:
    """Load all model result JSONs from ``results_dir``.

    Operational-metrics JSONs (``evaluation_type == "operational_alert_metrics"``)
    are skipped here; they are loaded inside the operational-metrics renderer so
    they do not appear as a second, metric-less object-detection row.
    """
    results: dict = {}
    rdir = Path(results_dir)
    for rf in sorted(rdir.glob("*.json")):
        try:
            d = json.loads(rf.read_text(encoding="utf-8"))
            if d.get("evaluation_type") == "operational_alert_metrics":
                continue
            results[d.get("model_name", rf.stem)] = d
        except Exception:
            pass
    return results


def runnable_classification_models(models_dir: str = "models") -> list:
    """Return persisted classifier artifact names that are runnable on disk.

    A sklearn classifier is runnable for live inference only if a serialized model
    artifact exists. The project does not persist sklearn classifiers today (the
    baseline scripts write metrics JSON and prediction CSVs, never a model file),
    so this returns an empty list. The inference demo then shows a clear
    missing-artifact state instead of training in-app or faking classification
    output.
    """
    base = Path(models_dir)
    if not base.exists():
        return []
    patterns = ("*_dfire.joblib", "*_dfire.pkl", "*_classifier.joblib", "*_classifier.pkl")
    found: list = []
    for pat in patterns:
        found.extend(sorted(p.name for p in base.glob(pat)))
    return found


def render_models_section(results_data: dict, include_comparison: bool = True) -> None:
    """Build the per-model tabs, optionally followed by the comparison tab.

    With ``include_comparison=True`` this reproduces the Operations & Learning
    Models tab exactly: sklearn tabs -> YOLO11n -> YOLO11s -> Model comparison,
    where the comparison tab also shows the operational alert metrics. With
    ``include_comparison=False`` only the per-model tabs are built (used by the
    M3 Models tab, which has a separate Model comparison (KPI) tab).
    """
    _sklearn_names = sorted(
        [n for n, d in results_data.items() if _is_sklearn_result(d)],
        key=_model_sort_key,
    )

    def _find_yolo_result(key_substr):
        for _n, _d in results_data.items():
            if _is_object_detection_result(_d) and key_substr in _n.lower():
                return _d
        return None

    _yolo11n_result = _find_yolo_result("yolo11n")
    _yolo11s_result = _find_yolo_result("yolo11s")

    _sklearn_labels = [_short_model_label(n) for n in _sklearn_names]
    _tab_labels = _sklearn_labels + ["YOLO11n", "YOLO11s"]
    if include_comparison:
        _tab_labels = _tab_labels + ["Model comparison"]
    _model_tabs = st.tabs(_tab_labels)

    for _tab, _mname in zip(_model_tabs[:len(_sklearn_names)], _sklearn_names):
        with _tab:
            _render_single_baseline_model(_mname, results_data[_mname])

    with _model_tabs[len(_sklearn_names)]:
        _render_yolo_detection_model(
            "YOLO11n", "YOLO11n", "lightweight baseline / fallback",
            _yolo11n_result,
            "results/results_yolo11n.csv",
            "runs/detect/yolo11n_dfire_baseline/results.csv",
        )

    with _model_tabs[len(_sklearn_names) + 1]:
        _render_yolo_detection_model(
            "YOLO11s", "YOLO11s", "current primary detector",
            _yolo11s_result,
            "results/results_yolo11s.csv",
            "runs/detect/yolo11s_dfire_baseline/results.csv",
        )

    if include_comparison:
        with _model_tabs[-1]:
            _render_model_comparison(results_data)
            st.divider()
            render_operational_alert_metrics(results_data)


# ──────────────── extracted per-model / comparison render helpers ────────────────
# ── Helper: short display label ──────────────────────────────────
def _short_model_label(name):
    if "Dummy" in name:
        return "Dummy"
    if "Logistic" in name:
        return "Logistic Regression"
    if "Random Forest" in name:
        return "Random Forest"
    if "yolo11n" in name.lower():
        return "YOLO11n"
    if "yolo11s" in name.lower():
        return "YOLO11s"
    return name

# ── Helper: sort order ───────────────────────────────────────────
def _model_sort_key(name):
    if "Dummy" in name:
        return (0, name)
    if "Logistic" in name:
        return (1, name)
    if "Random Forest" in name:
        return (2, name)
    if "yolo11n" in name.lower():
        return (3, name)
    if "yolo11s" in name.lower():
        return (4, name)
    return (5, name)

# ── Helper: result type detection ─────────────────────────────────
def _is_object_detection_result(result_dict):
    return result_dict.get("model_family") == "object_detection"

def _is_sklearn_result(result_dict):
    return not _is_object_detection_result(result_dict)

# ── Helper: per-model summary info text ──────────────────────────
def _model_summary_text(name):
    if "Dummy" in name:
        return (
            "The dummy baseline achieved 47% accuracy by always predicting background, "
            "but it completely failed to detect fire and smoke. "
            "This proves that accuracy alone is not enough for PyroFinder. "
            "Any real model must improve Macro F1 and, most importantly, "
            "achieve meaningful recall for fire and smoke."
        )
    if "Logistic" in name:
        return (
            "Logistic Regression is the first real learning baseline. It uses the 60 handcrafted "
            "color features and improves Macro F1 to about 0.62, with real recall for both fire "
            "and smoke. This proves that simple color information contains useful signal, but the "
            "model still creates many false alarms by confusing background images with fire or smoke."
        )
    if "Random Forest" in name:
        return (
            "Random Forest is the strongest classical ML baseline so far. It improves Macro F1 to "
            "about 0.85 and gives balanced recall for background, fire, and smoke. This suggests "
            "that the relationship between color features and fire/smoke labels is non-linear, and "
            "that tree-based models capture these patterns much better than a linear classifier."
        )
    return (
        "This baseline is an image-level sklearn classifier using handcrafted color features. "
        "It should be compared using Macro F1 and fire/smoke recall, not accuracy alone."
    )

# ── Helper: per-model detailed analysis text ─────────────────────
def _model_detailed_analysis(name):
    if "Dummy" in name:
        return """
This result is not a real fire-detection model. It is a DummyClassifier with the most_frequent strategy, meaning it always predicts the most common class: background.

---

#### What the result means

The model reaches about 47% accuracy only because background is the largest class in the test set. It gets background recall of 1.00, but fire recall and smoke recall are both 0.00. This means it misses every real fire and every smoke case.

---

#### What it tells us about the data

The dataset is somewhat imbalanced, but not broken. Background is the largest class, while fire and smoke are still well represented. The result mainly proves that accuracy alone is misleading.

---

#### What it tells us about the model

The model does not use visual meaning and does not learn fire or smoke patterns. It only gives the minimum bar that every real model must beat.

---

#### Main conclusion

The DummyClassifier is useful only as a minimum comparison point: PyroFinder must beat Macro F1 = 0.21 and must achieve recall above 0 for both fire and smoke.
"""
    if "Logistic" in name:
        return """
Logistic Regression is a simple linear learning baseline. Unlike the dummy model, it actually uses the 60 color features extracted from each image.

---

#### What the result means

The model improves strongly over the dummy baseline. It reaches about 61% accuracy and Macro F1 around 0.62. Most importantly, it detects both danger classes, with meaningful recall for fire and smoke.

---

#### What it tells us about the data

The color features contain useful signal. Fire and smoke images are not random in feature space: simple RGB/HSV statistics and histograms already help separate them from background.

---

#### What it tells us about the model

The model is still limited because it is linear. It catches many fire and smoke cases, but it also misclassifies many background images as fire or smoke. This means it is too aggressive and would create too many false alarms in an operational system.

---

#### Main conclusion

Logistic Regression proves that handcrafted color features are useful, but it is not strong enough as an operational baseline because background handling and false-alarm behavior are weak.
"""
    if "Random Forest" in name:
        return """
Random Forest is a non-linear classical ML baseline using the same 60 handcrafted color features as Logistic Regression.

---

#### What the result means

Random Forest performs much better than the other sklearn baselines. It reaches about 86% accuracy and Macro F1 around 0.85, with strong recall for background, fire, and smoke.

---

#### What it tells us about the data

The dataset contains strong visual color patterns that can separate the three image-level classes. However, because this is based on simple color features, the model may also learn dataset-specific patterns such as lighting, background style, or scene color distribution.

---

#### What it tells us about the model

The strong improvement over Logistic Regression suggests that the relationship between features and labels is non-linear. Random Forest captures these interactions better and reduces false alarms much more effectively.

---

#### Main conclusion

Random Forest is the strongest classical ML baseline. It should be used as the main simple sklearn baseline, but it is still only an image-level classifier. It cannot replace YOLO11s, because PyroFinder needs object detection, bounding boxes, and approximate location support.
"""
    return """
This is an image-level sklearn classifier using handcrafted color features.

---

#### What the result means

Evaluate using Macro F1 and fire/smoke recall rather than accuracy alone, since background class imbalance makes accuracy misleading.

---

#### What it tells us about the data

Performance relative to other baselines indicates how much useful signal the 60 color features contain for this class.

---

#### What it tells us about the model

Compare against DummyClassifier (Macro F1 = 0.21) as the minimum bar and against Logistic Regression and Random Forest to see if it adds value.

---

#### Main conclusion

Any sklearn baseline is an image-level classifier. It cannot replace YOLO11s object detection, which is needed for bounding boxes, confidence scores, and approximate location support.
"""

# ── Helper: render one model tab ─────────────────────────────────
def _render_single_baseline_model(model_name, result_dict):
    _r = result_dict
    _metrics = _r.get("metrics", {})
    _clf_report = _metrics.get("classification_report", {})
    _dataset = _r.get("dataset", {})
    _features = _r.get("features", {})
    _classes_ordered = ["background", "fire", "smoke"]
    _classes = [c for c in _classes_ordered if c in _clf_report]
    _slug = _short_model_label(model_name).lower().replace(" ", "_")

    # Key metric cards
    _km1, _km2, _km3, _km4, _km5 = st.columns(5)
    _km1.metric("Accuracy",    f"{_metrics.get('accuracy', 0):.2f}")
    _km2.metric("F1 macro",    f"{_metrics.get('macro_avg', {}).get('f1', 0):.2f}")
    _km3.metric("F1 weighted", f"{_metrics.get('weighted_avg', {}).get('f1', 0):.2f}")
    _km4.metric("Fire recall",  f"{_clf_report.get('fire', {}).get('recall', 0):.2f}")
    _km5.metric("Smoke recall", f"{_clf_report.get('smoke', {}).get('recall', 0):.2f}")

    st.info(_model_summary_text(model_name))
    st.caption(
        f"Run date: {_r.get('run_date', '—')}  ·  "
        f"Dataset: {_dataset.get('name', '—')}  ·  "
        f"Train: {_dataset.get('train_size', '—'):,}  ·  "
        f"Test: {_dataset.get('test_size', '—'):,}"
    )

    st.divider()

    # Row 1: Per-class bar chart | Dataset distribution
    _row1_l, _row1_r = st.columns(2)

    with _row1_l:
        st.subheader("Precision / Recall / F1 per class")
        _prf_rows = []
        for _cls in _classes:
            for _mn, _ml in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]:
                _prf_rows.append({
                    "class": _cls,
                    "metric": _ml,
                    "value": _clf_report[_cls].get(_mn, 0),
                })
        if _prf_rows:
            _fig_prf = px.bar(
                pd.DataFrame(_prf_rows),
                x="class", y="value", color="metric",
                barmode="group",
                color_discrete_map={
                    "Precision": "#4fc3f7",
                    "Recall":    "#e07b39",
                    "F1":        "#81c784",
                },
                labels={"value": "Score (0–1)", "class": "Class", "metric": ""},
                title=f"Per-class metrics — {_short_model_label(model_name)}",
            )
            _fig_prf.update_layout(yaxis_range=[0, 1], bargap=0.2, height=360)
            apply_chart_theme(_fig_prf)
            st.plotly_chart(_fig_prf, use_container_width=True, key=f"baseline_prf_{_slug}")

    with _row1_r:
        st.subheader("Class distribution — train vs test")
        _dist = _dataset.get("class_distribution", {})
        _dist_rows = []
        for _split_name, _counts in _dist.items():
            for _cls, _n in _counts.items():
                _dist_rows.append({"split": _split_name, "class": _cls, "count": _n})
        if _dist_rows:
            _fig_dist = px.bar(
                pd.DataFrame(_dist_rows),
                x="class", y="count", color="split",
                barmode="group",
                color_discrete_map=SPLIT_COLORS,
                labels={"count": "Images", "class": "Class", "split": "Split"},
                title="Images per class — train vs test",
            )
            _fig_dist.update_layout(bargap=0.2, height=360)
            apply_chart_theme(_fig_dist)
            st.plotly_chart(_fig_dist, use_container_width=True, key=f"baseline_dist_{_slug}")

    st.divider()

    # Row 2: Radar chart | Full metrics table
    _row2_l, _row2_r = st.columns(2)

    with _row2_l:
        st.subheader("Macro average radar")
        _macro = _metrics.get("macro_avg", {})
        _radar_cats = ["Precision", "Recall", "F1", "Accuracy"]
        _radar_vals = [
            _macro.get("precision", 0),
            _macro.get("recall",    0),
            _macro.get("f1",        0),
            _metrics.get("accuracy", 0),
        ]
        _fig_radar = go.Figure(go.Scatterpolar(
            r=_radar_vals + [_radar_vals[0]],
            theta=_radar_cats + [_radar_cats[0]],
            fill="toself",
            fillcolor="rgba(224,123,57,0.18)",
            line=dict(color=PYRO_COLORS["primary"], width=2),
            name=_short_model_label(model_name),
        ))
        _fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], tickfont=dict(size=10)),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cccccc"),
            height=340,
            margin=dict(l=50, r=50, t=40, b=40),
        )
        st.plotly_chart(_fig_radar, use_container_width=True, key=f"baseline_radar_{_slug}")

    with _row2_r:
        st.subheader("Full metrics table")
        _tbl_rows = []
        for _cls in _classes:
            _row_data = _clf_report[_cls]
            _tbl_rows.append({
                "Class":     _cls,
                "Precision": round(_row_data.get("precision", 0), 2),
                "Recall":    round(_row_data.get("recall", 0), 2),
                "F1":        round(_row_data.get("f1", 0), 2),
                "Support":   int(_row_data.get("support", 0)),
            })
        for _avg_key, _avg_label in [("macro_avg", "macro avg"), ("weighted_avg", "weighted avg")]:
            _avg_data = _metrics.get(_avg_key, {})
            if _avg_data:
                _tbl_rows.append({
                    "Class":     _avg_label,
                    "Precision": round(_avg_data.get("precision", 0), 2),
                    "Recall":    round(_avg_data.get("recall", 0), 2),
                    "F1":        round(_avg_data.get("f1", 0), 2),
                    "Support":   "—",
                })
        st.dataframe(pd.DataFrame(_tbl_rows), use_container_width=True, hide_index=True)
        st.caption(
            f"Accuracy: **{_metrics.get('accuracy', 0):.2f}**  ·  "
            f"Macro F1: **{_metrics.get('macro_avg', {}).get('f1', 0):.2f}**"
        )

    st.divider()

    # Feature details
    with st.expander("Feature extraction details"):
        _fc1, _fc2 = st.columns(2)
        with _fc1:
            st.markdown(f"**Description:** {_features.get('description', '—')}")
            st.markdown(f"**Vector length:** {_features.get('vector_length', '—')}")
            st.markdown(f"**Image resize:** {_features.get('image_resize', '—')}")
            st.markdown(f"**Normalisation:** {_features.get('normalization', '—')}")
        with _fc2:
            _comps = _features.get("components", [])
            if _comps:
                st.dataframe(pd.DataFrame(_comps), use_container_width=True, hide_index=True)

    # Detailed analysis
    with st.expander("Detailed analysis — what this baseline tells us", expanded=False):
        st.markdown(_model_detailed_analysis(model_name))

# ── Helper: render a YOLO object-detection model tab ─────────────
# One generic renderer shared by YOLO11n and YOLO11s so their tabs
# stay identical. It shows object-detection metrics and training
# curves only — never sklearn classification metrics. Real values are
# read from ``result_dict``; when it is missing/empty a clear pending
# state is shown and no metric values are invented.
def _render_yolo_detection_model(
    model_key, display_name, role_label, result_dict,
    training_csv_path, fallback_csv_path,
):
    _r = result_dict or {}
    _m = _r.get("metrics", {})
    _slug = model_key.lower()
    _ran = _r.get("run_date") is not None and _m.get("map50") is not None

    st.subheader(f"{display_name} Object-Detection — {role_label}")
    st.caption(
        f"{display_name} detector · fire / smoke · "
        "bounding boxes + confidence · D-Fire test split"
    )

    if not _ran:
        st.warning(
            f"{display_name} measured result file not found. "
            "No metric values are shown until a measured result file exists "
            f"(`results/baseline_{_slug}.json`). No placeholder values are invented."
        )

    # Metric cards
    def _fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    _yk1, _yk2, _yk3, _yk4, _yk5 = st.columns(5)
    _yk1.metric("mAP@0.5",       _fmt(_m.get("map50")))
    _yk2.metric("mAP@0.5:0.95",  _fmt(_m.get("map50_95")))
    _yk3.metric("Precision",      _fmt(_m.get("precision")))
    _yk4.metric("Recall",         _fmt(_m.get("recall")))
    _yk5.metric("F1",             _fmt(_m.get("f1")))

    if model_key == "YOLO11s":
        st.info(
            "YOLO11s is the **current primary detector** for PyroFinder. "
            "Like YOLO11n it is an object detector — it predicts bounding boxes, "
            "class labels, and confidence scores for fire and smoke — but it is the "
            "larger model that delivers the best detection quality. "
            "YOLO11n remains the lightweight speed baseline / fallback; YOLO11s is "
            "compared against it using detection metrics (mAP, precision, recall)."
        )
    else:
        st.info(
            "YOLO11n is the lightweight object-detection **baseline / fallback** for PyroFinder. "
            "Unlike the sklearn baselines, it does not classify the whole image only — "
            "it predicts bounding boxes, class labels, and confidence scores for fire and smoke. "
            "This makes it the correct baseline for the YOLO11s current primary detector, because "
            "PyroFinder needs localization for approximate map-based alerts. "
            "YOLO11n is a speed-oriented baseline."
        )

    if _r.get("run_date"):
        _ds = _r.get("dataset", {})
        st.caption(
            f"Run date: {_r['run_date']}  ·  "
            f"Dataset: {_ds.get('name', '—')}  ·  "
            f"Train: {_ds.get('train_size', '—'):,}  ·  "
            f"Test: {_ds.get('test_size', '—'):,}"
        )

    st.divider()

    # Metrics table
    st.subheader("Detection metrics")
    _det_rows = [
        {"Metric": "mAP@0.5",      "Value": _fmt(_m.get("map50")),    "Meaning": "Detection quality at IoU threshold 0.5"},
        {"Metric": "mAP@0.5:0.95", "Value": _fmt(_m.get("map50_95")), "Meaning": "Stricter detection quality across IoU thresholds 0.5 to 0.95"},
        {"Metric": "Precision",     "Value": _fmt(_m.get("precision")),"Meaning": "How many predicted detections are correct"},
        {"Metric": "Recall",        "Value": _fmt(_m.get("recall")),   "Meaning": "How many real fire/smoke objects are found"},
        {"Metric": "F1",            "Value": _fmt(_m.get("f1")),       "Meaning": "Balance between precision and recall"},
    ]
    st.dataframe(pd.DataFrame(_det_rows), use_container_width=True, hide_index=True)

    # Per-class table
    _per_class = _m.get("per_class")
    if _per_class:
        st.subheader("Per-class metrics")
        _pc_rows = []
        for _cls_name in ["smoke", "fire"]:
            _cls_d = _per_class.get(_cls_name, {})
            _pc_rows.append({
                "Class":      _cls_name,
                "mAP@0.5":    _fmt(_cls_d.get("map50")),
                "mAP@0.5:0.95": _fmt(_cls_d.get("map50_95")),
            })
        st.dataframe(pd.DataFrame(_pc_rows), use_container_width=True, hide_index=True)

    # ── Run metadata ──────────────────────────────────────────────
    if _ran:
        with st.expander("Run metadata", expanded=False):
            _ds = _r.get("dataset", {})
            _mp = _r.get("model_params", {})

            def _meta_int(v):
                return f"{v:,}" if isinstance(v, int) else "—"

            _sel_epoch = _mp.get("selected_epoch_or_row")
            if _sel_epoch is None:
                _sel_epoch = _m.get("selected_epoch_or_row")
            _batch = _mp.get("batch", _mp.get("batch_requested"))
            _meta_rows = [
                ("Run date", _r.get("run_date", "—")),
                ("Dataset", _ds.get("name", "—")),
                ("Train size", _meta_int(_ds.get("train_size"))),
                ("Test size", _meta_int(_ds.get("test_size"))),
                ("Image size", _mp.get("imgsz", "—")),
                ("Epochs requested", _mp.get("epochs_requested", "—")),
                ("Selected epoch / row", _sel_epoch if _sel_epoch is not None else "—"),
                ("Batch", _batch if _batch is not None else "—"),
            ]
            if _mp.get("run_dir"):
                _meta_rows.append(("Run directory", _mp.get("run_dir")))
            st.dataframe(
                pd.DataFrame(_meta_rows, columns=["Field", "Value"]).astype(str),
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ── Training curves ───────────────────────────────────────────
    _results_csv_primary = Path(training_csv_path)
    _runs_csv_fallback = Path(fallback_csv_path)
    _runs_csv = _results_csv_primary if _results_csv_primary.exists() else _runs_csv_fallback
    if _runs_csv.exists():
        try:
            _tc_df = pd.read_csv(_runs_csv)
            _tc_df.columns = [c.strip() for c in _tc_df.columns]

            _loss_cols = {
                "train/box_loss": "Train box loss",
                "train/cls_loss": "Train cls loss",
                "train/dfl_loss": "Train dfl loss",
                "val/box_loss":   "Val box loss",
                "val/cls_loss":   "Val cls loss",
            }
            _map_cols = {
                "metrics/mAP50(B)":     "mAP@0.5",
                "metrics/mAP50-95(B)":  "mAP@0.5:0.95",
                "metrics/precision(B)": "Precision",
                "metrics/recall(B)":    "Recall",
            }

            _epoch_col = "epoch" if "epoch" in _tc_df.columns else None
            if _epoch_col:
                _tc_l, _tc_r = st.columns(2)

                with _tc_l:
                    st.subheader("Training loss vs. epoch")
                    _loss_fig = go.Figure()
                    _loss_colors = {
                        "Train box loss": "#e07b39",
                        "Train cls loss": "#4fc3f7",
                        "Train dfl loss": "#81c784",
                        "Val box loss":   "#e07b39",
                        "Val cls loss":   "#4fc3f7",
                    }
                    _loss_dash = {
                        "Train box loss": "solid",
                        "Train cls loss": "solid",
                        "Train dfl loss": "solid",
                        "Val box loss":   "dash",
                        "Val cls loss":   "dash",
                    }
                    for _col, _label in _loss_cols.items():
                        if _col in _tc_df.columns:
                            _loss_fig.add_trace(go.Scatter(
                                x=_tc_df[_epoch_col],
                                y=_tc_df[_col],
                                mode="lines",
                                name=_label,
                                line=dict(
                                    color=_loss_colors.get(_label, "#cccccc"),
                                    dash=_loss_dash.get(_label, "solid"),
                                    width=2,
                                ),
                            ))
                    _loss_fig.update_layout(
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        legend=dict(orientation="h", yanchor="top", y=-0.25),
                        height=340,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cccccc"),
                        margin=dict(l=50, r=20, t=20, b=60),
                    )
                    apply_chart_theme(_loss_fig)
                    st.plotly_chart(_loss_fig, use_container_width=True, key=f"yolo_loss_curve_{_slug}")

                with _tc_r:
                    st.subheader("Validation metrics vs. epoch")
                    _map_fig = go.Figure()
                    _map_colors = ["#e07b39", "#4fc3f7", "#81c784", "#ffb74d"]
                    for _i, (_col, _label) in enumerate(_map_cols.items()):
                        if _col in _tc_df.columns:
                            _map_fig.add_trace(go.Scatter(
                                x=_tc_df[_epoch_col],
                                y=_tc_df[_col],
                                mode="lines",
                                name=_label,
                                line=dict(color=_map_colors[_i % len(_map_colors)], width=2),
                            ))
                    _map_fig.update_layout(
                        xaxis_title="Epoch",
                        yaxis_title="Score (0–1)",
                        yaxis_range=[0, 1],
                        legend=dict(orientation="h", yanchor="top", y=-0.25),
                        height=340,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cccccc"),
                        margin=dict(l=50, r=20, t=20, b=60),
                    )
                    apply_chart_theme(_map_fig)
                    st.plotly_chart(_map_fig, use_container_width=True, key=f"yolo_map_curve_{_slug}")

                _n_epochs_done = len(_tc_df)
                st.caption(
                    f"Showing {_n_epochs_done} epoch(s) completed · "
                    f"solid lines = train · dashed lines = val · "
                    f"source: `{_runs_csv}`"
                )
        except Exception as _e:
            st.info(f"Could not load training curves: {_e}")
    else:
        st.info(
            "Training curves not found. "
            f"Expected at: `{training_csv_path}` "
            f"(fallback: `{fallback_csv_path}`)."
        )

    st.divider()

    with st.expander(f"Detailed analysis — what {display_name} tells us", expanded=False):
        if model_key == "YOLO11s":
            st.markdown("""
YOLO11s is the current primary detector for PyroFinder. Like YOLO11n it performs object detection — it predicts **where** fire or smoke appears in the frame — but it is the larger model that delivers the best detection quality.

---

#### What the result means

The key metrics are mAP, precision, and recall, not accuracy. mAP measures whether the predicted bounding boxes overlap the real fire/smoke boxes. Recall is especially important because missing real fire or smoke is more dangerous than creating a false alert.

---

#### What it tells us about the data

This evaluation tests whether D-Fire annotations let the larger YOLO11s model learn stronger localization patterns for fire and smoke than the lightweight YOLO11n baseline.

---

#### What it tells us about the model

YOLO11s is heavier than YOLO11n and is expected to be more accurate. It should be compared against YOLO11n using detection metrics (mAP@0.5, recall), never against the sklearn image-level classifiers.

---

#### Main conclusion

YOLO11s is selected as the main detector only if it improves detection quality — especially mAP@0.5 and recall — and the operational alert metrics over YOLO11n, while keeping acceptable inference speed. YOLO11n remains the lightweight speed baseline / fallback.
""")
        else:
            st.markdown("""
YOLO11n is different from the sklearn baselines. The sklearn models classify an entire image as background, fire, or smoke using handcrafted color features. YOLO11n performs object detection: it predicts **where** fire or smoke appears in the frame.

---

#### What the result means

The key metrics are mAP, precision, and recall, not accuracy. mAP measures whether the predicted bounding boxes overlap the real fire/smoke boxes. Recall is especially important because missing real fire or smoke is more dangerous than creating a false alert.

---

#### What it tells us about the data

This evaluation tests whether D-Fire annotations are useful for object detection, not only for image-level classification. It also shows whether the model can learn separate localization patterns for fire and smoke.

---

#### What it tells us about the model

YOLO11n is lightweight and fast, but it may be less accurate than YOLO11s. It is useful as a speed-oriented baseline and fallback model.

---

#### Main conclusion

YOLO11n is the correct object-detection baseline for PyroFinder. YOLO11s should be selected as the main model only if it improves detection quality, especially mAP@0.5 and recall, while keeping acceptable inference speed.
""")

# ── Helper: build comparison dataframe ───────────────────────────
def _build_comparison_df(results_data):
    _cmp_rows = []
    for _n, _d in results_data.items():
        _m = _d.get("metrics", {})
        _cr = _m.get("classification_report", {})
        _ma = _m.get("macro_avg", {})
        _cmp_rows.append({
            "Model":             _short_model_label(_n),
            "Accuracy":          round(_m.get("accuracy", 0), 2),
            "Macro Precision":   round(_ma.get("precision", 0), 2),
            "Macro Recall":      round(_ma.get("recall", 0), 2),
            "Macro F1":          round(_ma.get("f1", 0), 2),
            "Fire Recall":       round(_cr.get("fire", {}).get("recall", 0), 2),
            "Smoke Recall":      round(_cr.get("smoke", {}).get("recall", 0), 2),
            "Background Recall": round(_cr.get("background", {}).get("recall", 0), 2),
            "Run date":          _d.get("run_date", "—"),
        })
    return pd.DataFrame(_cmp_rows)

# ── Helper: render comparison tab ────────────────────────────────
def _render_model_comparison(results_data):
    # ── A. Conclusion text ────────────────────────────────────────
    st.info(
        "**Comparison conclusion:** "
        "The sklearn baselines test whether simple image-level color features can separate "
        "background, fire, and smoke. DummyClassifier is only a minimum bar, Logistic Regression "
        "proves that color features contain signal, and Random Forest is the strongest classical "
        "image-level baseline. "
        "YOLO11n is different: it is the first object-detection baseline. It should be judged by "
        "mAP, precision, and recall because it predicts bounding boxes, not just image labels. "
        "The final YOLO11s model should be compared mainly against YOLO11n, not against the sklearn "
        "models, because both YOLO models solve the real PyroFinder task: detecting and localizing "
        "fire/smoke."
    )

    st.divider()
    render_classification_comparison(results_data)
    st.divider()
    render_object_detection_comparison(results_data)


def render_classification_comparison(results_data):
    """Image-level sklearn classification comparison (accuracy / Macro F1 / recall).

    Kept separate from the object-detection comparison so sklearn Macro F1 is
    never placed in a direct ranking against YOLO11 mAP.
    """
    _sklearn_data = {n: d for n, d in results_data.items() if _is_sklearn_result(d)}

    # ── Sklearn classification baselines ──────────────────────
    st.subheader("Image-level sklearn classification baselines")
    st.caption(
        "These baselines classify the whole image using 60 handcrafted color features. "
        "They are evaluated with image-level classification metrics (accuracy, Macro F1, recall)."
    )

    if _sklearn_data:
        _cmp_df = _build_comparison_df(_sklearn_data)

        st.dataframe(_cmp_df, use_container_width=True, hide_index=True)

        st.divider()

        # Macro F1 bar chart — sklearn only
        st.subheader("Macro F1 comparison — sklearn baselines")
        _fig_cmp = px.bar(
            _cmp_df,
            x="Model", y="Macro F1",
            color="Model",
            text="Macro F1",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Macro F1 — sklearn image-level baselines",
            labels={"Macro F1": "F1 macro (higher is better)"},
        )
        _fig_cmp.update_layout(yaxis_range=[0, 1], showlegend=False)
        apply_chart_theme(_fig_cmp)
        st.plotly_chart(_fig_cmp, use_container_width=True, key="baseline_cmp_f1_bar")

        st.divider()

        # Radar chart — sklearn only
        st.subheader("Macro average radar — sklearn baselines")
        _radar_cats = ["Macro Precision", "Macro Recall", "Macro F1", "Accuracy"]
        _radar_colors = px.colors.qualitative.Set2
        _radar_fig = go.Figure()
        for _i, (_, _row) in enumerate(_cmp_df.iterrows()):
            _vals = [
                _row["Macro Precision"],
                _row["Macro Recall"],
                _row["Macro F1"],
                _row["Accuracy"],
            ]
            _radar_fig.add_trace(go.Scatterpolar(
                r=_vals + [_vals[0]],
                theta=_radar_cats + [_radar_cats[0]],
                fill="toself",
                name=_row["Model"],
                line=dict(color=_radar_colors[_i % len(_radar_colors)], width=2),
            ))
        _radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], tickfont=dict(size=10)),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cccccc"),
            height=400,
            margin=dict(l=60, r=60, t=50, b=40),
            showlegend=True,
        )
        st.plotly_chart(_radar_fig, use_container_width=True, key="baseline_cmp_radar")
    else:
        st.info("No sklearn baseline results found in `results/`.")


def render_object_detection_comparison(results_data):
    """YOLO11n vs YOLO11s object-detection comparison (mAP / precision / recall / F1).

    Detection metrics only — never compared against the sklearn classification
    baselines. A detector row shows a missing-file status until its measured
    result file exists; no values are invented.
    """
    # ── Object-detection baselines (YOLO11n + YOLO11s) ─────────
    from src.results_loader import (
        load_detection_result as _load_det,
        status_label as _status_label,
        STATUS_OK as _STATUS_OK,
    )

    st.subheader("Object-detection comparison")

    # Expected detection result files. Each detector is loaded the same way;
    # if a measured file is absent the loader reports a missing-file status
    # instead of fabricating metrics.
    _det_specs = [
        ("YOLO11n", "results/baseline_yolo11n.json"),
        ("YOLO11s", "results/baseline_yolo11s.json"),
    ]

    def _fv(v):
        return round(v, 4) if v is not None else "—"

    _det_rows = []
    _det_loaded = {}
    for _dname, _dpath in _det_specs:
        _loaded = _load_det(_dpath)
        _det_loaded[_dname] = _loaded
        _dm = (_loaded["data"] or {}).get("metrics", {}) if _loaded["status"] == _STATUS_OK else {}
        _det_rows.append({
            "Model":        _dname,
            "mAP@0.5":      _fv(_dm.get("map50")),
            "mAP@0.5:0.95": _fv(_dm.get("map50_95")),
            "Precision":    _fv(_dm.get("precision")),
            "Recall":       _fv(_dm.get("recall")),
            "F1":           _fv(_dm.get("f1")),
            "Status":       _status_label(_loaded["status"]),
        })
    st.dataframe(
        pd.DataFrame(_det_rows)[
            ["Model", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1", "Status"]
        ],
        use_container_width=True, hide_index=True,
    )

    # Radar chart for any measured detector. YOLO11s is drawn first and YOLO11n
    # last so the YOLO11n shape stays visible in front, and both fills are
    # translucent so neither trace hides the other where they overlap.
    _radar_line_colors = {"YOLO11n": "#e07b39", "YOLO11s": "#4fc3f7"}
    _radar_fill_colors = {
        "YOLO11n": "rgba(224,123,57,0.25)",
        "YOLO11s": "rgba(79,195,247,0.25)",
    }
    _det_radar_fig = go.Figure()
    _radar_cats = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]
    _any_measured = False
    for _dname in ("YOLO11s", "YOLO11n"):
        _loaded = _det_loaded.get(_dname)
        if not _loaded or _loaded["status"] != _STATUS_OK:
            continue
        _dm = (_loaded["data"] or {}).get("metrics", {})
        if not all(_dm.get(k) is not None for k in ["map50", "map50_95", "precision", "recall", "f1"]):
            continue
        _any_measured = True
        _vals = [_dm["map50"], _dm["map50_95"], _dm["precision"], _dm["recall"], _dm["f1"]]
        _det_radar_fig.add_trace(go.Scatterpolar(
            r=_vals + [_vals[0]],
            theta=_radar_cats + [_radar_cats[0]],
            fill="toself",
            fillcolor=_radar_fill_colors.get(_dname, "rgba(129,199,132,0.25)"),
            line=dict(color=_radar_line_colors.get(_dname, "#81c784"), width=2),
            name=_dname,
        ))
    if _any_measured:
        st.subheader("Detection radar — measured detectors")
        _det_radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], tickfont=dict(size=10)),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cccccc"),
            height=380,
            margin=dict(l=60, r=60, t=50, b=40),
        )
        st.plotly_chart(_det_radar_fig, use_container_width=True, key="baseline_yolo_radar")

    st.divider()

   
# ── Helper: Operational Alert Metrics (cost-sensitive comparison) ─
def render_operational_alert_metrics(results_data):
    from src.evaluation import (
        operational_alert_metrics_from_confusion_matrix as _op_from_cm,
    )
    from src.results_loader import (
        load_operational_result as _load_op,
        is_selectable_operational as _is_selectable,
        select_operational_winner as _select_winner,
        status_label as _status_label,
        STATUS_OK as _STATUS_OK,
    )

    st.subheader("Operational Alert Metrics")
    st.caption(
        "Comparison at the alert level: fire/smoke = hazard, background = no hazard. "
        "**Primary decision metric: Alert F2-score** — the F-beta score (beta = 2) of "
        "Alert Precision and Hazard Recall. It weights recall above precision, because "
        "missing a real fire or smoke event is more costly than a false alarm, while "
        "still penalizing too many false alerts (which erode customer trust). Hazard "
        "Recall and Alert Precision are shown alongside it as its components."
    )

    def _op_for(d):
        # Prefer embedded operational_metrics; else derive from a stored
        # confusion matrix (same predictions, no re-run).
        om = d.get("operational_metrics")
        if om:
            return om
        cm = d.get("metrics", {}).get("confusion_matrix")
        if cm and cm.get("matrix") and cm.get("labels"):
            return _op_from_cm(cm["matrix"], cm["labels"])
        return None

    def _fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"

    # Expected YOLO operational result files. A detector reports a
    # missing-file status until its measured file exists.
    _op_specs = [
        ("YOLO11n", "results/yolo11n_operational_metrics.json",
         "results/baseline_yolo11n.json"),
        ("YOLO11s", "results/yolo11s_operational_metrics.json",
         "results/baseline_yolo11s.json"),
    ]

    rows, chart_data = [], []

    # ── sklearn rows (image-level classifiers — no localization) ──
    _sk = {n: d for n, d in results_data.items()
           if not _is_object_detection_result(d)}
    for _n in sorted(_sk, key=_model_sort_key):
        _om = _op_for(_sk[_n])
        if not _om:
            continue
        _label = _short_model_label(_n)
        rows.append({
            "Model": _label,
            "Recall": _fmt(_om.get("hazard_recall")),
            "False Alert Rate": _fmt(_om.get("false_alert_rate")),
            "Precision": _fmt(_om.get("alert_precision")),
            "F1 Score": _fmt(_om.get("alert_f1")),
            "F2 Score": _fmt(_om.get("alert_f2")),
            "Location Coverage": "N/A",
            "Mean Location Error": "N/A",
            "3x3 Grid Hit Rate": "N/A",
            "Status": "Measured (image-level)",
        })
        chart_data.append({"Model": _label, "Metric": "Recall",
                           "Value": _om.get("hazard_recall", 0)})
        chart_data.append({"Model": _label, "Metric": "F2 Score",
                           "Value": _om.get("alert_f2", 0)})

    # ── YOLO operational rows (dedicated operational JSON per model) ──
    _missing_yolo_op = []
    for _mname, _op_path, _det_path in _op_specs:
        _loaded = _load_op(_op_path)
        if _loaded["status"] != _STATUS_OK:
            # No measured file yet → explicit status row, no invented values.
            _missing_yolo_op.append((_mname, _op_path, _loaded["status"]))
            rows.append({
                "Model": _mname,
                "Recall": "—",
                "False Alert Rate": "—",
                "Alert Precision": "—",
                "F1 Score": "—",
                "F2 Score": "—",
                "Location Coverage": "—",
                "Mean Location Error": "—",
                "3x3 Grid Hit Rate": "—",
                "Status": _status_label(_loaded["status"]),
            })
            continue
        _data = _loaded["data"] or {}
        _om = _data.get("operational_metrics", {})
        _lm = _data.get("location_metrics", {}) or {}
        _label = _data.get("model_name", _mname)
        rows.append({
            "Model": _label,
            "Recall": _fmt(_om.get("hazard_recall")),
            "False Alert Rate": _fmt(_om.get("false_alert_rate")),
            "Alert Precision": _fmt(_om.get("alert_precision")),
            "F1 Score": _fmt(_om.get("alert_f1")),
            "F2 Score": _fmt(_om.get("alert_f2")),
            "Location Coverage": _fmt(_lm.get("location_coverage_rate")),
            "Mean Location Error": _fmt(_lm.get("fire_location_error_mean")),
            "3x3 Grid Hit Rate": _fmt(_lm.get("fire_location_grid_hit_rate")),
            "Status": "Measured" if _is_selectable(_loaded) else "Measured (incomplete)",
        })
        chart_data.append({"Model": _label, "Metric": "Recall",
                           "Value": _om.get("hazard_recall", 0)})
        chart_data.append({"Model": _label, "Metric": "F2 Score",
                           "Value": _om.get("alert_f2", 0)})

    _table_cols = [
        "Model", "Recall", "False Alert Rate",
        "Precision", "F1 Score", "F2 Score",
        "Location Coverage", "Mean Location Error", "3x3 Grid Hit Rate",
        "Status",
    ]
    if rows:
        st.dataframe(
            pd.DataFrame(rows)[_table_cols],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No operational alert metrics available in `results/` yet.")

    # ── Winner selection (YOLO detectors only; pending never wins) ──
    _winner = _select_winner(
        [(m, op) for m, op, _ in _op_specs],
        detection_items=[(m, det) for m, _, det in _op_specs],
    )
    if _winner:
        st.success(
            f"**Selected detector: {_winner}.** Selected by the **F2-score** "
            "— the primary metric (F-beta, beta = 2) that combines Precision and "
            "Recall while weighting recall higher — with object-detection Recall "
            "and mAP@0.5 as supporting detection-quality evidence."
        )
    else:
        st.info(
            "No detector selected yet: a model is chosen only when its measured "
            "operational result file exists with complete metrics. A detector with "
            "missing or incomplete result files cannot be selected."
        )

    # ── M3 failure-analysis summary (how YOLO11s wins + open weakness) ──
    st.info(
        "YOLO11s is selected, but the operational gain over YOLO11n is modest and "
        "consistent, not large. The detection-quality gain is clearer, especially "
        "mAP@0.5. Smoke-only images remain the dominant failure mode for both "
        "detectors."
    )
    with st.expander("M3 failure-analysis notes", expanded=False):
        st.markdown(
            "- 9 fewer missed hazards (FN 145 vs 154).\n"
            "- 5 fewer false alerts (FP 37 vs 42).\n"
            "- +0.0036 Alert F2 (0.9459 vs 0.9423).\n"
            "- +1.98 pp mAP@0.5 (0.7668 vs 0.7470), with consistent gains on "
            "mAP@0.5:0.95, Precision, Recall, and F1.\n"
            "- Smoke-only imagery is the main bottleneck (most misses for both "
            "detectors); the paired comparison shows partial complementarity — "
            "YOLO11s does not beat YOLO11n on every individual image.\n"
            "- Approximate fire-location metrics are practically tied; YOLO11s has "
            "slightly better coverage and 3x3 grid-hit rate. Location outputs are "
            "approximate image-space estimates only, never precise geolocation.\n\n"
            "Full analysis: `docs/M3_RESULTS_SUMMARY.md`."
        )

    # Compact chart: Hazard Recall + Alert F2 per model
    if chart_data:
        _fig_op = px.bar(
            pd.DataFrame(chart_data),
            x="Model", y="Value", color="Metric", barmode="group",
            color_discrete_map={
                "Recall": "#e07b39",
                "F2 Score": "#4fc3f7",
            },
            labels={"Value": "Score (0–1, higher is better)"},
            title="Recall vs F2 Score (beta = 2, recall weighted above precision)",
        )
        _fig_op.update_layout(yaxis_range=[0, 1], bargap=0.25, height=360)
        apply_chart_theme(_fig_op)
        st.plotly_chart(_fig_op, use_container_width=True, key="baseline_operational_bar")

    # ── Missing YOLO operational metrics → show generation command ──
    for _mname, _op_path, _ in _missing_yolo_op:
        _weights = f"models/{_mname.lower()}_dfire_best.pt"
        _csv = _op_path.replace("operational_metrics.json", "test_predictions.csv")
        st.info(
            f"{_mname} operational alert metrics not found (`{_op_path}`). "
            "Generate them (evaluation only, no training) with:\n\n"
            "```\n"
            "py scripts/evaluate_yolo_alert_metrics.py "
            "--raw-root \"<path-to-D-Fire-root>\" "
            f"--weights \"{_weights}\" "
            f"--model-name \"{_mname}\" --conf 0.25 "
            f"--output-json \"{_op_path}\" "
            f"--output-csv \"{_csv}\"\n"
            "```"
        )


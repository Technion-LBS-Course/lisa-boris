"""M3 Dashboard — Demo tab.

Upload one image and run the available fine-tuned D-Fire detectors. The detector
multiselect defaults to all available runnable detectors. Models load only on
demand (lazy), only fine-tuned `fire`/`smoke` checkpoints are used, pretrained
weights are never substituted, and missing checkpoints show a clear message.

Classification inference is offered only if a persisted, runnable sklearn artifact
exists; none do today. No classifier inference is faked and no model is trained
inside the app.
"""
import streamlit as st

from src.dashboards import model_helpers as mh


_OP_JSON = {
    "YOLO11n": "results/yolo11n_operational_metrics.json",
    "YOLO11s": "results/yolo11s_operational_metrics.json",
}


def _model_f2(model_name: str) -> str:
    """Return the measured Alert F2-score for a detector, or '—' if unavailable."""
    import json
    from pathlib import Path
    path = Path(_OP_JSON.get(model_name, ""))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        v = data.get("operational_metrics", {}).get("alert_f2")
        return f"{v:.4f}" if v is not None else "—"
    except Exception:
        return "—"


def render(confidence_threshold, confirmation_frames):
    from src import inference as _inf

    st.header("Demo")

    available = _inf.available_detectors()
    missing = [d for d in _inf.CHECKPOINTS if d not in available]

    # Classification inference is offered only if a persisted, runnable artifact exists.
    classifiers = mh.runnable_classification_models()

    for d in missing:
        if d == "YOLO11s":
            st.warning(_inf.MISSING_YOLO11S_MESSAGE)
        else:
            st.warning(
                "YOLO11n checkpoint not found. Add `models/yolo11n_dfire_best.pt` "
                "(e.g. `python scripts/YOLO11n_baseline.py --train`) to enable YOLO11n inference."
            )

    if not classifiers:
        st.info("Upload one image and run")

    if not available:
        st.info(
            "No fine-tuned detector checkpoints available yet. Add at least one of "
            "`models/yolo11n_dfire_best.pt` or `models/yolo11s_dfire_best.pt` to run inference."
        )
        return

    # Default selection = all available runnable detectors.
    selected = st.multiselect("Detectors to run", available, default=available)
    uploaded = st.file_uploader("Upload one image", type=["jpg", "jpeg", "png"])
    run = st.button(
        "Run demo", type="primary",
        disabled=(uploaded is None or not selected),
    )

    if uploaded is not None and run and selected:
        from PIL import Image as _PILImage

        image = _PILImage.open(uploaded).convert("RGB")
        cols = st.columns(len(selected))
        for col, mname in zip(cols, selected):
            with col:
                st.subheader(mname)
                try:
                    with st.spinner(f"Loading {mname} and running inference..."):
                        model = mh.load_detector_cached(mname)
                        res = _inf.run_detection(model, image, conf=confidence_threshold)
                    st.image(
                        res["annotated_png"],
                        caption=f"{mname} — {res['total_detections']} detection(s)",
                        use_container_width=True,
                    )
                    ic1, ic2 = st.columns(2)
                    ic1.metric("Fire detections", res["fire_count"])
                    ic2.metric("Smoke detections", res["smoke_count"])
                    ic3, ic4, ic5 = st.columns(3)
                    hc = res["max_confidence"]
                    ic3.metric("Highest confidence", f"{hc:.2f}" if hc is not None else "—")
                    ic4.metric("Inference time", f"{res['inference_ms']:.0f} ms")
                    ic5.metric("F2 Score", _model_f2(mname))
                except FileNotFoundError:
                    st.warning(
                        _inf.MISSING_YOLO11S_MESSAGE if mname == "YOLO11s"
                        else f"{mname} checkpoint not found. Add its fine-tuned D-Fire checkpoint."
                    )
                except ValueError as ve:
                    st.error(str(ve))
                except Exception as exc:
                    st.error(f"{mname} inference failed: {exc}")

    st.caption(
        f"Active confidence threshold: `{confidence_threshold}` · "
        f"Confirmation frames (N) for alert logic: `{confirmation_frames}` · "
        f"Classes: fire, smoke only."
    )
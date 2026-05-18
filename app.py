import streamlit as st

from src.data import get_primary_dataset_info
from src.model import get_model_plan, get_metrics_plan

st.set_page_config(page_title="PyroFinder", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("PyroFinder")
    st.caption("Fire detection and monitoring — existing cameras only")

    st.divider()

    mode = st.selectbox(
        "Dashboard mode",
        ["Operations & Learning Dashboard", "Central Control Dashboard"],
    )

    st.divider()

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    confirmation_frames = st.number_input(
        "Confirmation frames (N)",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Alert triggered only after fire/smoke detected in N consecutive frames.",
    )

    st.divider()
    model_plan = get_model_plan()
    st.caption(f"Primary model: **{model_plan['primary_model']}**")
    st.caption(f"Baseline: {model_plan['baseline_model']}")
    st.caption(f"Classes: {', '.join(model_plan['classes'])}")

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("PyroFinder")
st.subheader("Real-time fire outbreak detection and monitoring using existing customer cameras.")

if mode == "Operations & Learning Dashboard":
    tab_overview, tab_eda, tab_inference, tab_mapping, tab_alerts = st.tabs([
        "Overview",
        "Dataset & EDA",
        "Inference Demo",
        "Mapping Setup",
        "Alert Log",
    ])

    with tab_overview:
        st.header("System Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Planned model")
            st.json(get_model_plan())

        with col2:
            st.subheader("Evaluation metrics")
            for m in get_metrics_plan():
                st.markdown(f"- {m}")

        st.divider()
        st.subheader("Primary dataset")
        st.json(get_primary_dataset_info())

    with tab_eda:
        st.header("Dataset & EDA")
        st.info(
            "**Coming in Lecture 4–5 (M2):** "
            "Load D-Fire dataset, inspect class distribution, bounding box statistics, "
            "sample images, and compute baseline metrics. "
            "Place dataset files in `data/` (not tracked by Git)."
        )
        st.markdown("""
**EDA tasks planned:**
- Class distribution: fire vs smoke vs background
- Bounding box size and aspect ratio distributions
- Image resolution statistics
- Dataset split verification (train / val / test)
- Baseline metric (YOLO11n mAP@0.5 zero-shot or fine-tuned)
        """)

    with tab_inference:
        st.header("Inference Demo")
        st.info(
            "**Coming in Lecture 6–7 (M3):** "
            "Upload an image or video, run YOLO11s inference, "
            "and display bounding-box overlays with fire/smoke class labels and confidence scores."
        )
        st.markdown(f"""
**Settings active:**
- Confidence threshold: `{confidence_threshold}`
- Confirmation frames: `{confirmation_frames}`
- Primary model: `{model_plan['primary_model']}`
        """)

        uploaded = st.file_uploader(
            "Upload image or video (placeholder — inference not yet wired)",
            type=["jpg", "jpeg", "png", "mp4", "avi"],
            disabled=False,
        )
        if uploaded:
            st.warning("Model not loaded yet. Inference will be wired in M3.")

    with tab_mapping:
        st.header("Mapping Setup")
        st.info(
            "**Coming in M3:** "
            "Draw named polygons on camera images, link them to map areas, "
            "and configure camera GPS and metadata for approximate location output."
        )
        from src.mapping import get_mapping_modes
        st.markdown("**Supported mapping modes:**")
        for i, mode_name in enumerate(get_mapping_modes(), 1):
            st.markdown(f"{i}. {mode_name.capitalize()}")

        st.caption("All location outputs will be marked as approximate.")

    with tab_alerts:
        st.header("Alert Log")
        st.info(
            "**Coming in M3:** "
            "Confirmed fire/smoke alerts will appear here with camera ID, timestamp, "
            "apparent direction, approximate location, and status."
        )
        st.markdown("""
**Alert statuses:**
- `active` — detection confirmed, awaiting review
- `confirmed` — verified as real event
- `rejected` — dismissed by operator
- `false_alarm` — marked as false positive
        """)

elif mode == "Central Control Dashboard":
    st.header("Central Control Dashboard")
    st.info(
        "**Basic version — coming in M3:** "
        "Map view of all cameras, camera metadata table, alert history, "
        "and manual image polygon setup."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Camera metadata (placeholder)")
        import pandas as pd
        sample_cameras = pd.DataFrame({
            "camera_id": ["cam_001", "cam_002"],
            "site": ["north_boundary", "parking_area"],
            "status": ["active", "active"],
            "lat": [None, None],
            "lon": [None, None],
            "height_m": [None, None],
            "azimuth_deg": [None, None],
        })
        st.dataframe(sample_cameras, use_container_width=True)

    with col2:
        st.subheader("Map (placeholder)")
        st.caption("Map will display camera markers. Configure coordinates in camera metadata.")
        import folium
        from streamlit_folium import st_folium
        m = folium.Map(location=[32.0853, 34.7818], zoom_start=10)
        st_folium(m, height=300, use_container_width=True)

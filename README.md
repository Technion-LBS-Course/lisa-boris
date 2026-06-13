**Live app:** https://pyrofinder.streamlit.app/

If the link doesn't work, run locally: `streamlit run app.py`

---

# lisa-boris
ליסה + בוריס — פרויקט קורס LBS (016833)

# PyroFinder

**Real-time fire outbreak detection and monitoring using cameras that already exist at the customer site.**

---

## One-Liner

Private property owners in fire-prone areas suffer from delayed fire awareness, we will build PyroFinder — a real-time fire monitoring system that uses existing cameras, YOLO11s model for fire/smoke detection, and approximate map-based alerts.

---

## Problem Statement

Private property owners: homeowners, farm owners, ranch owners, agricultural facility managers, and private landowners, install security cameras at their properties, but those cameras are passive. Someone must actively watch every feed to notice smoke or fire. During dry seasons, fires can start at property edges, agricultural fields, forest borders, parking areas, or neighboring land. By the time the event is noticed, it may already be well-established.

Existing wildfire monitoring solutions require dedicated towers, sensors, drones, or public-sector infrastructure built for land agencies, not individual property owners. PyroFinder fills this gap by turning cameras the customer already owns into an automated fire/smoke monitoring layer,  with multi-frame confirmation, approximate location output, and operational visibility — without any new hardware.

---

## Target Audience

**Primary users and paying customers:** homeowners, farm owners, ranch owners, agricultural facility managers, and private landowners in fire-prone areas.

**Primary persona: Dani, farm owner, central Israel:**
Dani manages a 120-dunam farm with outdoor security cameras at boundary points. During dry summer months, fire risk from neighboring fields or agricultural equipment is high. PyroFinder monitors the feeds automatically and alerts Dani when fire or smoke is confirmed, including which camera triggered it and the approximate location within the frame.

**Main use case:** A fire ignites at the edge of Dani's property. PyroFinder detects smoke or fire, confirms it across multiple frames, raises an alert, and shows the approximate event location as a named image polygon (e.g., "north field") or image quadrant  so Dani can respond.

**Secondary users:** Municipalities, emergency response teams, rescue teams, and forest and park authorities may receive alerts with approximate location context — a named image zone, image quadrant, or approximate map location when mapping configuration is available — when a detected fire may affect public areas. The Emergency / Third-Party Viewer Dashboard is a future product anchor, not part of the first course MVP.

---

## Product Outputs

PyroFinder consists of three operational products and one internal product.

### 1. Central Control Dashboard
**For:** PyroFinder operator / admin. Manages all customers, sites, and cameras on a map. Displays camera health, active detections, alert history, and mapping status. Supports manual editing of camera location, height, azimuth, responsibility zones, and polygons. Supports alert review, confirmation, and false-alarm marking.

### 2. Mobile Customer App
**For:** End customer / property owner. Receives fire/smoke alerts with approximate location. Allows confirming or rejecting alerts and marking false alarms. Shows active events on a map relative to the customer's property. 

### 3. Emergency / Third-Party Viewer Dashboard
**For:** Firefighting teams, rescue services, municipalities, forest and park authorities. Read-only view of active alerts with approximate location and event status, filtered to areas under public or shared responsibility. 

### 4. Operations & Learning Dashboard
**For:** PyroFinder internal team. Internal tool for building, testing, evaluating, and improving the detection system. Capabilities: dataset loading and inspection; basic EDA; model comparison (YOLO11s vs YOLO11n); inference on uploaded images or videos with detection overlay; mAP, precision, recall, F1, false alarm rate, and inference speed; performance breakdown by conditions (day/night, smoke density, glare, fog, indoor/outdoor); false positive/negative review; model experiment tracking; manual image polygon definition; basic polygon-to-map linking; alert log from test runs.

---

## Data Source + Data Card

All datasets are normalized to a unified two-class schema: `fire` and `smoke`. Before any training or evaluation. Other labels (`human`, `vehicle`) are used only as background negatives, never as detection targets.

| Dataset | URL | Role | Format / Size | License / Notes | Known Gaps / Biases |
|---|---|---|---|---|---|
| **D-Fire** | [github.com/gaia-solutions-on-demand/DFireDataset](https://github.com/gaia-solutions-on-demand/DFireDataset) | Primary training and held-out test evaluation | 21,527 images; fire-only: 1,164; smoke-only: 5,867; fire+smoke: 4,658; background: 9,838; 14,692 fire boxes, 11,865 smoke boxes; YOLO-format | CC0 1.0 Universal | Limited night scenes, indoor fires, close-range agricultural fires; skews toward outdoor wildland fires |
| **Smart Fire System Dataset** | [github.com/mehmoodulhaq570/Smart-Fire-System-Yolov11n](https://github.com/mehmoodulhaq570/Smart-Fire-System-Yolov11n) | Supplementary training / external validation |32,603 images; train: 26,379, val: 4,394; classes: fire, smoke; YOLO-format| MIT License | Label format must be verified before use |
| **Aerial Rescue Object Detection** | [kaggle.com/datasets/julienmeine/rescue-object-detection](https://www.kaggle.com/datasets/julienmeine/rescue-object-detection) | Robustness validation | Classes: Fire, Vehicle, Human; Fire used for eval; Vehicle/Human as background negatives; Dataset is not splitted;  Extensive datasets - 33GB |Attribution 4.0 International (CC BY 4.0)| Aerial perspective differs from ground-level camera angles; Labels were created using ML, thus labels may not be accurate;|
| **Fire Detection in YOLO Format** | [kaggle.com/datasets/ankan1998/fire-detection-in-yolo-format](https://www.kaggle.com/datasets/ankan1998/fire-detection-in-yolo-format) | Supplementary training after class verification | YOLOv5-format; augmented; 270 images (train: 243, val: 16, test: 11) | GPL 2 | Small dataset; not realistic fires; class compatibility must be verified |
| **FURG Fire Dataset** | [github.com/steffensbola/furg-fire-dataset](https://github.com/steffensbola/furg-fire-dataset) | Video validation | 24 videos; per-video XML annotations; not pre-split; conversion to YOLO format required | CC0 1.0 | Smoke coverage to verify; used for temporal behavior and multi-frame tracking only |

**Dataset usage strategy:** Primary training: D-Fire. Supplementary training: Smart Fire System Dataset and Fire Detection in YOLO Format after label normalization. Robustness validation: Aerial Rescue OD (Fire class only; Vehicle/Human as negatives). Video validation: FURG. All labels normalized to `fire` / `smoke` before use.

---

## Map and Geo Data Strategy

Map and geo data are **not YOLO training data.** They are operational and configuration data for geolocation, alert display, and camera/site management.

Geo data types: customer property polygons; responsibility zones; camera GPS coordinates, height, and azimuth; indoor/outdoor flag; image-space polygons; linked map polygons, terrain cells, and map points; reference points (image ↔ map); optional basemap, DEM/topography, and GIS context layers (roads, buildings, fences, vegetation).

Candidate libraries (no paid provider required): Folium, pydeck, GeoPandas, Shapely, Rasterio, PyProj, Streamlit map components.

---

## Formal ML Problem Definition

**Task:** Two-class object detection — `fire` and `smoke`.

**Input X:** RGB images or sampled video frames from outdoor cameras, resized to 640 × 640 pixels.

**Output y:** Per-frame detections — bounding box `(x_center, y_center, width, height)` in normalized coordinates, class label ∈ {`fire`, `smoke`}, confidence score ∈ [0, 1].

**Main model:** Ultralytics YOLO11s (`yolo11s.pt`). Selected for near-real-time sampled-frame inference with higher accuracy than YOLO11n.

**Baseline / fallback:** Ultralytics YOLO11n (`yolo11n.pt`), fine-tuned on the same data. Speed baseline and fallback only, not an equal parallel model. YOLO11s is the improvement if it achieves higher mAP@0.5 and recall at acceptable inference speed; otherwise YOLO11n is the fallback.

**Loss:** Ultralytics YOLO detection loss,  bounding-box regression, classification loss, and distribution focal loss.

**Metrics:** mAP@0.5 (primary), mAP@0.5:0.95, Precision, Recall, F1-score, False Alarm Rate (FP per hour or per 1,000 sampled frames), Inference speed (FPS or ms/frame).

**KPI:** The model is object detection, the metric is recall, because missing a real fire is far more costly than a false alarm.

**Split:** D-Fire's provided split where available; otherwise reproducible 70/15/15 stratified by image category.

---

## Current Model Results

The fine-tuned **YOLO11n baseline** has been evaluated two complementary ways. Detection metrics evaluate the boxes and classes the model predicts; operational metrics reduce each image to *hazard detected / not detected*. They answer different questions and are not interchangeable. (YOLO11s, the planned primary detector, is not yet trained.)

**YOLO11n object-detection baseline** (D-Fire test split):

```text
mAP@0.5: 0.7470
mAP@0.5:0.95: 0.4249
Precision: 0.7397
Recall: 0.6825
F1: 0.7099
```

**YOLO11n operational alert evaluation** (D-Fire test split, evaluation only, confidence 0.25; `fire` and `smoke` both count as a hazard, and a missed hazard is weighted 10× a false alert):

```text
Hazard Recall: 0.9331
False Alert Rate: 0.0209
Alert Precision: 0.9808
Alert F1: 0.9563
Operational Alert Score: 0.9368
```

Approximate fire-location coverage: 0.9148 · 3×3 grid hit rate: 0.9559 (image-space estimates only, not precise geolocation).

Result files: `results/baseline_yolo11n.json` (detection), `results/yolo11n_operational_metrics.json` and `results/yolo11n_test_predictions.csv` (operational + per-image failure analysis).

### YOLO11s — training in progress

**YOLO11s, the planned primary detector, is currently training on Kaggle. No YOLO11s metrics are available yet, and no synthetic or placeholder values are used anywhere.** The app and model-comparison code are already prepared to load the real YOLO11s outputs as soon as they exist:

```text
models/yolo11s_dfire_best.pt              — fine-tuned checkpoint (local only, Git-ignored)
results/baseline_yolo11s.json             — object-detection metrics (mAP, precision, recall, F1)
results/results_yolo11s.csv               — per-epoch training curves
results/yolo11s_operational_metrics.json  — operational alert + approximate fire-location metrics
results/yolo11s_test_predictions.csv      — per-image alert outcome + fire-location error table
```

Until those files exist, the dashboard shows a clear **Training in progress** state for YOLO11s, never invents metrics, and never selects YOLO11s as the winning detector. The Inference Demo hides the YOLO11s option and shows: *"YOLO11s training is still in progress. Add models/yolo11s_dfire_best.pt after the Kaggle run completes."* When the real files arrive, they are loaded as measured results automatically (see `src/results_loader.py` and `src/inference.py`).

---

## Detection, Tracking, and Geolocation Logic

1. YOLO11s detects `fire` and `smoke` per sampled frame, producing bounding boxes, class labels, and confidence scores.
2. **Multi-frame confirmation** — a confirmed alert requires detection above the confidence threshold across `N` consecutive frames from the same camera. `N` is configurable.
3. Fire bounding box centroids estimate the **approximate fire location.** When camera GPS, height, and azimuth are registered, the centroid projects to approximate map coordinates. Otherwise, location is reported as a named image polygon or image quadrant.
4. Smoke centroid movement across frames estimates **apparent smoke direction**, usable as a rough wind-direction proxy only when validated.
5. The **mapping layer** resolves detections to a named polygon, image quadrant, or approximate map point.

Output wording is always approximate: "apparent direction in the camera frame," "image-plane spread direction," "approximate fire location based on camera projection." Geographic bearing is available only when compass orientation is registered. The MVP does not claim true physical fire-spread prediction.

---

## Mapping and Geolocation Strategy

Mapping is an **offline, pre-event setup stage** — not something solved during a live event. It translates image-space detections into approximate map or property locations before any fire occurs.

**Mode 1 — Responsibility zone definition:** Mark areas in the camera image as in-scope or out-of-scope.

**Mode 2 — Named polygon creation:** Draw and name polygons (e.g., "north field," "parking area," "forest edge") on the camera image.

**Mode 3 — Image-to-map polygon linking:** Link an image polygon to a map polygon, terrain cell, or map point. Detections inside generate an approximate map location.

**Mode 4 — Camera GPS setup:** Enter camera latitude/longitude manually or capture on-site.

**Mode 5 — Camera metadata setup:** Height, azimuth/compass bearing, indoor/outdoor flag, field of view, optional zoom state.

**Mode 6 — Reference-point mapping:** Mark visible fixed landmarks (road junctions, gates, buildings, fence corners) that appear in both the camera frame and the map, for future semi-automatic registration.

**Fully automatic image-to-map registration is a future feature and is not required for the course MVP.**

---

## Technical Architecture

```mermaid
flowchart TD
    A[Camera Feed / Uploaded Video / Image Set] --> B[Frame Sampling & Preprocessing\nOpenCV · resize to 640x640]
    B --> C[YOLO11s Detection\nfire / smoke · confidence · bbox]
    C --> D[Multi-Frame Confirmation\nN consecutive frames threshold]
    D --> E[Tracking & Direction Analysis\ncentroid movement · bbox area · apparent direction]
    E --> F[Mapping & Geolocation\nimage polygon lookup · camera metadata projection · reference points]
    F --> G[Alert Generation\nalert record · timestamp · class · location · direction]

    G --> H[Central Control Dashboard\noperator · all customers / sites / cameras]
    G --> I[Mobile Customer App\ncustomer alerts · future]
    G --> J[Emergency Viewer Dashboard\nread-only · future]

    B --> K[Operations & Learning Dashboard\ndataset EDA · inference · model comparison · experiment tracking]
    C --> K
    D --> K

    subgraph Storage
        L[ML Datasets · outside Git]
        M[Model Checkpoints & Experiment Logs]
        N[Camera & Site Config · Geo & Mapping Data]
        O[Alert History]
    end

    K --> M
    G --> O
    F --> N
```

- **Frame Sampling:** OpenCV reads from uploaded video or sample frames. Live RTSP ingestion is out of scope this semester.
- **YOLO11s Detection:** Fine-tuned two-class model. Confidence thresholding and NMS suppress weak and duplicate detections.
- **Multi-Frame Confirmation:** Alert issued only after `N` consecutive detections from the same camera.
- **Tracking & Direction Analysis:** Centroid-based tracking; bounding-box area change; apparent image-plane spread direction.
- **Mapping & Geolocation:** Image-space polygon lookup, camera metadata projection, reference-point linking.
- **Operations & Learning Dashboard:** Primary MVP deliverable — dataset inspection, EDA, model evaluation, inference, experiment tracking.

---

## Input / Output Schema

| Object | Key Fields |
|---|---|
| **Customer** | `customer_id`, `name`, `contact_info` |
| **Site / property** | `site_id`, `customer_id`, `name`, `location_polygon` (opt GeoJSON), `address` (opt) |
| **Camera** | `camera_id`, `site_id`, `name`, `status` (active / inactive / error) |
| **Camera metadata** | `camera_id`, `latitude` (opt), `longitude` (opt), `height_m` (opt), `azimuth_deg` (opt), `fov_horizontal_deg` (opt), `fov_vertical_deg` (opt), `indoor_outdoor`, `zoom_state` (opt) |
| **Image polygon** | `polygon_id`, `camera_id`, `name`, `vertices` (normalized), `polygon_type`, `linked_map_polygon_id` (opt) |
| **Map polygon** | `map_polygon_id`, `site_id`, `name`, `geometry` (GeoJSON), `polygon_type` |
| **Reference point** | `ref_point_id`, `camera_id`, `image_x`, `image_y`, `map_lat`, `map_lon`, `label` (opt) |
| **Frame input** | `timestamp`, `camera_id`, `image` (RGB array or video frame) |
| **Detection output** | `timestamp`, `camera_id`, `class` ∈ {`fire`, `smoke`}, `confidence`, `bbox` (normalized x_center, y_center, w, h) |
| **Tracking output** | `timestamp`, `camera_id`, `track_id`, `centroid`, `bbox_area`, `apparent_direction`, `matched_image_polygon_id` (opt), `approximate_map_location` (opt) |
| **Alert** | `alert_id`, `timestamp`, `camera_id`, `site_id`, `customer_id`, `detected_class`, `confidence`, `apparent_direction`, `image_polygon_name` (if avail), `approximate_lat` (opt), `approximate_lon` (opt), `geographic_bearing` (only if compass registered), `status` |
| **Model experiment** | `experiment_id`, `model_name`, `dataset`, `split`, `hyperparameters`, `metrics` (mAP, precision, recall, F1, FAR, speed), `notes`, `timestamp` |
| **Dataset record** | `dataset_id`, `name`, `source_url`, `num_images`, `classes`, `split_info`, `license`, `role` |
| **Evaluation run** | `run_id`, `experiment_id`, `dataset_id`, `split`, `metrics`, `timestamp` |

---

## User Stories

**Story 1 — Customer receives confirmed fire/smoke alert**
> As a property owner, I want to receive an alert the moment fire or smoke is confirmed in any of my camera feeds, so that I can respond immediately without monitoring footage manually.

*Acceptance criterion:* When YOLO11s detects fire or smoke above the configured threshold across `N` consecutive frames, the dashboard displays a confirmed alert including camera identifier, timestamp, and approximate location if available.

**Story 2 — Operator sees all customers and cameras on a central map**
> As a PyroFinder operator, I want to see all customers, sites, and cameras on a single map view, so that I can monitor system status across all installations from one screen.

*Acceptance criterion:* The Central Control Dashboard shows all registered cameras as map markers. Clicking a marker shows camera status, recent alerts, and metadata. Active fire events are visually distinguished from inactive cameras.

**Story 3 — Operator defines image polygons and links them to map areas**
> As a PyroFinder operator setting up a new customer, I want to draw named polygons on each camera image and link them to map areas, so that detections report an approximate geographic location.

*Acceptance criterion:* The operator can draw at least one named polygon on a camera image and link it to a map polygon or point. When a test detection falls inside the polygon, the alert includes the polygon name and linked map location.

---

## Related Work

| System | Approach | Why PyroFinder is different |
|---|---|---|
| **Pano AI** | Panoramic camera towers with cloud-based AI for public land monitoring | Requires dedicated hardware towers; targets public land agencies, not property owners |
| **FIREWAVE** | Acoustic sensors for fire-sound detection in forests | Requires specialized acoustic hardware; not camera-based |
| **CANDO** | Autonomous drone systems for security and public-safety operations | Requires drone operations and airspace coordination |

PyroFinder uses cameras the customer already owns — no new towers, sensors, drones, or public-sector infrastructure required.

---

## Risk Register

| # | Risk | Category | Likelihood | Impact | Mitigation |
|---|---|---|---|---|---|
| 1 | **Dataset domain gap** — D-Fire may not match private-property camera angles, lighting, or scenarios; datasets differ in formats and labels | Data | High | High | Normalize labels to `fire`/`smoke`; convert formats before training; validate on FURG and Aerial Rescue OD; apply augmentation |
| 2 | **False alarms** — reflections, sunsets, headlights, fog, dust, vehicles, and humans may trigger false detections | Technical | Medium | High | Multi-frame confirmation; tune thresholds per class; include D-Fire background images and rescue-scene negatives |
| 3 | **Poor camera calibration** — missing or incorrect height, azimuth, or FOV metadata leads to inaccurate geolocation | Technical | Medium | Medium | Mark all location outputs as approximate; allow manual correction in the dashboard; improve with reference-point matching |


---


## Repository Structure

<!-- Updated 2026-06-09: added YOLO11n baseline results, scripts/YOLO11n_baseline.py, models/ (local only) -->
<!-- Updated 2026-06-12: added src/evaluation.py, scripts/evaluate_yolo_alert_metrics.py, tests/test_evaluation.py, YOLO11n operational metrics -->

```text
project-root/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── .gitignore
├── .env.example
├── app.py
├── src/
│   ├── __init__.py
│   ├── data.py          ← dataset loading, inspection, Data Card utilities
│   ├── eda.py           ← EDA helpers: summary metrics, bbox stats, spatial analysis
│   ├── viz.py           ← on-the-fly YOLO box annotation (class map: 0=smoke, 1=fire)
│   ├── ui.py            ← shared UI palette, CAT_COLORS, CLASS_COLORS, chart theme
│   ├── model.py         ← model metadata, metrics plan, evaluation helpers
│   ├── detection.py     ← DetectionResult dataclass, class validation
│   ├── tracking.py      ← multi-frame confirmation, apparent direction estimation
│   ├── mapping.py       ← mapping modes, polygon helpers, location formatting
│   ├── alerts.py        ← alert record creation, status validation
│   └── evaluation.py    ← cost-sensitive operational alert metrics + approximate fire-location helpers (pure stdlib)
├── scripts/
│   ├── build_dfire_metadata.py  ← generates data/dfire_metadata.csv from raw D-Fire
│   ├── dummy_try.py             ← M3 sklearn baseline: D-Fire loading, feature extraction, DummyClassifier
│   ├── simple_baselines.py      ← M3: Logistic Regression and Random Forest baselines
│   ├── YOLO11n_baseline.py      ← M3: YOLO11n object-detection baseline runner (reproducible)
│   └── evaluate_yolo_alert_metrics.py  ← M3: evaluation-only operational alert + fire-location metrics for a YOLO checkpoint (no training)
├── results/
│   ├── baseline_dummy_classifier.json      ← DummyClassifier metrics
│   ├── baseline_logistic_regression.json   ← Logistic Regression metrics
│   ├── baseline_random_forest.json         ← Random Forest metrics
│   ├── baseline_yolo11n.json               ← YOLO11n detection metrics (mAP, P, R, F1)
│   ├── results_yolo11n.csv                 ← YOLO11n per-epoch training curves
│   ├── yolo11n_operational_metrics.json    ← YOLO11n operational alert + fire-location metrics
│   └── yolo11n_test_predictions.csv        ← YOLO11n per-image alert outcome + fire-location error table
├── models/                      ← local only, Git-ignored (model weights)
│   └── yolo11n_dfire_best.pt    ← YOLO11n best checkpoint from Kaggle training
├── data/
│   ├── dfire_metadata.csv        ← committed; app runs on a fresh clone using only this
│   ├── dfire_yolo11n.yaml        ← YOLO data config for YOLO11n training
│   ├── samples/
│   │   └── dfire/
│   │       ├── images/   ← 20 committed sample images
│   │       └── labels/   ← matching YOLO label files
│   └── market-survey/    ← competitor screenshots
├── docs/
│   ├── M2_DATA_EDA.md
│   ├── M2_dashboard.md
│   ├── M2_GAP_LIST.md
│   ├── AI_AGENT_SYSTEM.md
│   ├── Literature_review.md
│   └── market_survey_wildfire_existing_sensors.md
├── design_images/        ← persona, user-journey, and UI mockup assets
├── SprintPlan/
│   ├── SPRINT_PLAN.md
│   └── Sprint_Plan_PyroFinder_final_14Jul.xlsx
├── notebooks/
│   └── 01_eda.ipynb
└── tests/
    ├── test_smoke.py
    └── test_evaluation.py   ← unit tests for src/evaluation.py (alert confusion, cost weighting, location helpers)
```

---

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## M2 Data / EDA Progress (26/05/2026 checkpoint)

### Data location

`data/dfire_metadata.csv` is committed to Git and is the primary data source for all EDA
charts and metrics. The app runs fully on a fresh clone using only this CSV.

Raw D-Fire images and labels are **never committed** and must be stored locally if you
need to re-generate the CSV or view sample images from the full dataset.

Sample annotated images (20 images + YOLO labels) are committed to Git at:
```
data/samples/dfire/images/   ← raw sample images
data/samples/dfire/labels/   ← YOLO label files for annotation overlay
```

The EDA tab shows these committed samples when local raw paths are unavailable.

### Generate metadata CSV

```bash
python scripts/build_dfire_metadata.py \
  --raw-root "C:\Users\boris.azarov\OneDrive - Technion\Desktop\PyroFinder\RAW_DATA\D-Fire" \
  --output data/dfire_metadata.csv
```

To also generate annotated sample images:
```bash
python scripts/build_dfire_metadata.py \
  --raw-root "C:\Users\boris.azarov\OneDrive - Technion\Desktop\PyroFinder\RAW_DATA\D-Fire" \
  --output data/dfire_metadata.csv \
  --copy-samples data/samples/dfire \
  --sample-count 20
```

The script always overwrites the output CSV — safe to rerun.

### Run the Streamlit dashboard

```bash
streamlit run app.py
```

After a fresh clone (`git clone` + `pip install -r requirements.txt`), the app runs
without any local dataset installation. `data/dfire_metadata.csv` and
`data/samples/dfire/` are committed and provide all data the dashboard needs.

### What the Dataset & EDA tab currently shows

- **6 metrics:** total images, fire images, smoke images, background images, mean boxes/image, median boxes/image.
- **Filters:** image category, split, has_fire, has_smoke (all in sidebar).
- **Chart 1:** Bar chart — image count by category.
- **Chart 2:** Histogram — bounding box count per image.
- **Chart 3:** Bar chart — split distribution.
- **Table preview:** 20 rows with key columns.
- **EDA insight:** written, data-driven observation.
- **Sample images expander:** annotated thumbnails if generated locally.

### Actual metadata counts (full dataset — 21,527 images)

| Split | Images |
|---|---|
| train | 17,221 |
| test | 4,306 |

| Category | Images |
|---|---|
| background | 9,838 |
| smoke_only | 5,867 |
| fire_and_smoke | 4,658 |
| fire_only | 1,164 |

Fire boxes: 14,692 · Smoke boxes: 11,865

### Current EDA insight

D-Fire (21,527 images) is class-imbalanced: 45.7% background, 27.3% smoke-only, 21.6% fire+smoke, 5.4% fire-only. Fire recall must be the primary evaluation metric — a "predict background always" model would score ~46% accuracy but detect nothing. Weighted loss or oversampling of fire categories is recommended for YOLO11s fine-tuning.

### Run tests

```bash
python -m pytest
```

### Notes

- No YOLO11s training has been done yet. YOLO11s fine-tuning is next after M3.
- YOLO11n baseline is **complete** — see the M3 YOLO11n section below for results.
- The complete D-Fire dataset documentation is in `docs/M2_DATA_EDA.md`.
- Class mapping verified: D-Fire class 0 = smoke, class 1 = fire (confirmed against official category counts).

---

## M3 Sklearn Baseline Progress (2026-06-05)

**The model is an image-level classifier (fire / smoke / background). The primary metric is F1 macro, because the classes are imbalanced and both fire and smoke recall matter equally. Recall on fire is the minimum acceptable threshold — a model that never detects fire is useless.**

### Sklearn classifier pipeline

`scripts/dummy_try.py` implements the full sklearn baseline pipeline on the real D-Fire dataset:

- Loads D-Fire using the dataset's pre-existing train/test split (`train/` → training, `test/` → evaluation).
- Falls back to `data/samples/dfire/` on machines without raw data.
- Extracts a 60-value feature vector per image: RGB mean+std, HSV mean+std, 16-bin color histogram × 3 channels.
- Image-level label from YOLO boxes: fire if class 1 present, smoke if class 0 only, background if empty label file.

```bash
python scripts/dummy_try.py
```

### Dataset split (full D-Fire)

| Split | background | fire | smoke | total |
|---|---|---|---|---|
| train | 7,833 | 4,707 | 4,681 | 17,221 |
| test | 2,005 | 1,115 | 1,186 | 4,306 |

### Baseline results — DummyClassifier (most_frequent)

| Model | Accuracy | F1 macro | Fire recall | Smoke recall |
|---|---|---|---|---|
| DummyClassifier | 0.47 | 0.21 | 0.00 | 0.00 |

Baseline always predicts "background". Any real classifier must exceed F1 macro > 0.21 and achieve recall > 0 on both fire and smoke.

Full baseline metrics saved to `results/baseline_dummy_classifier.json`.

### Sklearn classifier results (full D-Fire)

| Model | Accuracy | F1 macro | Fire recall | Smoke recall |
|---|---|---|---|---|
| DummyClassifier | 0.47 | 0.21 | 0.00 | 0.00 |
| Logistic Regression | ~0.61 | ~0.62 | >0 | >0 |
| Random Forest | ~0.86 | ~0.85 | >0 | >0 |

Full metrics saved to `results/baseline_logistic_regression.json` and `results/baseline_random_forest.json`.

---

## M3 YOLO11n Object-Detection Baseline

**Training platform:** Kaggle Notebook, GPU Tesla T4

**Dataset:** D-Fire (primary training and held-out test evaluation)

| Split | Images |
|---|---|
| Train | 17,221 |
| Test | 4,306 |

**Classes:** `0 = smoke`, `1 = fire`

**Model:** Ultralytics YOLO11n  
**Image size:** 640 px  
**Epochs:** 30  
**Batch size:** 16

### Results

| Metric | Value |
|---|---|
| mAP@0.5 | **0.747** |
| mAP@0.5:0.95 | 0.4249 |
| Precision | 0.7397 |
| Recall | 0.6825 |
| F1 | 0.7099 |

### Result files

| File | Description |
|---|---|
| `results/baseline_yolo11n.json` | Detection metrics (committed to Git) |
| `results/results_yolo11n.csv` | Per-epoch training curves (committed to Git) |
| `models/yolo11n_dfire_best.pt` | Best checkpoint — **local only, Git-ignored** |

### Important distinctions

YOLO11n is an **object-detection** baseline, not an image-level sklearn classifier.
It predicts bounding boxes, class labels, and confidence scores for fire and smoke.
It should be compared to the future YOLO11s using detection metrics:
mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, and inference speed.

Do **not** compare YOLO11n mAP directly to sklearn accuracy or Macro F1 —
these measure different tasks at different granularities.

YOLO11n is the lightweight **baseline and fallback** model.
YOLO11s remains the planned primary detector for PyroFinder.

### Reproducibility

The training was run on Kaggle because local hardware GPU availability was limited.
To reproduce locally:

```bash
python scripts/YOLO11n_baseline.py --train --raw-root "<path-to-D-Fire>"
```

Ultralytics downloads starting weights (`yolo11n.pt`) automatically if not present.

### Operational alert evaluation (evaluation only)

Beyond detection metrics, YOLO11n was also evaluated with PyroFinder's cost-sensitive **operational alert** metrics — see the [Current Model Results](#current-model-results) section above for the numbers. This run performs inference only (no training) on the D-Fire test split and writes `results/yolo11n_operational_metrics.json` and `results/yolo11n_test_predictions.csv`:

```bash
python scripts/evaluate_yolo_alert_metrics.py \
  --raw-root "<path-to-D-Fire>" \
  --weights "models/yolo11n_dfire_best.pt" \
  --model-name "YOLO11n" --conf 0.25
```

---

*PyroFinder · Technion Course 016833 · Location-Based Services: Data Science · 2026*

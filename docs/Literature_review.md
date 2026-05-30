# PyroFinder — Literature Review

## Research Question

How can deep learning-based object detection be used to achieve accurate and fast real-time fire and wildfire detection from ordinary RGB civilian/security-camera video and images?

## Project Connection

PyroFinder is a real-time fire outbreak detection and monitoring system that uses cameras already installed at the customer site. The project focuses on detecting `fire` and `smoke`, confirming detections across consecutive frames, and producing approximate map-based alerts for private property owners. Therefore, the most relevant literature is work on visual fire/smoke detection, YOLO-style object detection, false-alarm reduction, real-time inference, and dataset generalization.

## Literature Review

Early fire and wildfire detection from ordinary RGB civilian and security cameras is increasingly studied as a practical alternative to sensor-based systems because cameras are already widely deployed and can provide continuous visual monitoring. The central challenge is to detect flames or smoke quickly enough to support intervention while keeping false alarms and computational cost low (Cheng et al., 2024; Saleh et al., 2024). For PyroFinder, this directly supports the product decision to use existing customer cameras instead of requiring new hardware.

Recent survey work shows that visual fire detection has moved from handcrafted image-processing pipelines toward deep learning models that automatically learn discriminative fire and smoke features from images and video. Cheng et al. argue that this shift is important because deep learning supports the three main visual fire tasks: classification, localization, and segmentation, while improving both accuracy and efficiency (Cheng et al., 2024). The lesson for PyroFinder is that classification alone is not enough; the system should localize the detected fire or smoke region so it can support approximate location and alert context.

Within this literature, object detection is especially relevant because it can both recognize the presence of fire and localize the fire or smoke region in the frame. The survey by Cheng et al. emphasizes that YOLO-style detectors are widely used for this purpose because they provide a strong balance between detection speed and accuracy, which is essential for real-time monitoring (Cheng et al., 2024). This influences PyroFinder by supporting the use of YOLO11s as the main object detector and YOLO11n as a speed baseline/fallback.

A practical example of this direction is the staged wildfire and smoke system proposed by Bahhar et al., which combines an ensemble CNN classifier with a YOLO detector. Their pipeline first checks whether a frame contains an abnormal event and then uses YOLO to localize smoke or fire, which helps reduce unnecessary detection work and improves robustness in complex scenes (Bahhar et al., 2023). This paper influences PyroFinder by showing that a staged pipeline can reduce false positives and computational load, even though PyroFinder will start with a simpler YOLO11s detection-first MVP.

Bahhar et al. report strong classification performance and solid detection results, but they also note that data quality remains a major limitation, especially the lack of good real-world UAV fire and smoke imagery. This point is important for RGB camera studies as well, because models trained on limited datasets often struggle to generalize to new camera views, weather conditions, and background clutter (Bahhar et al., 2023). The lesson for PyroFinder is to treat dataset quality and domain gap as a main risk, not as a secondary issue.

A related challenge is that wildfires occur in highly variable environments, and visual cues such as smoke, haze, clouds, and lighting changes can look similar. Wicaksono et al. show that YOLOv8 can detect fire and smoke with moderate success after preprocessing and training, but their results also reveal that limited dataset size and the absence of real-world testing constrain performance (Wicaksono et al., 2024). This paper influences PyroFinder by reinforcing the need to evaluate the model with real or realistic camera images, not only benchmark images.

That study still matters because it demonstrates that a modern YOLO detector can be trained for wildfire recognition using ordinary image data and can produce usable real-time predictions. The authors' mAP of 0.63, precision of 0.70, and recall of 0.57 suggest that there is room for improvement, especially if the system is intended for operational surveillance use (Wicaksono et al., 2024). The lesson for PyroFinder is to report mAP, precision, recall, and false-alarm behavior clearly rather than relying on visual demo examples only.

Survey evidence also suggests that the main way to improve YOLO-based fire detectors is by making them more lightweight, more attentive to small smoke targets, and better at fusing multiscale features. Das et al. describe recent YOLOv8-based fire and smoke detectors that use attention modules, lightweight necks, and edge-oriented optimization to improve speed and reduce false alarms (Das et al., 2026). This influences PyroFinder by making inference speed and model size part of the evaluation plan, not just accuracy.

Their survey also highlights an important deployment point: real-time wildfire detection is no longer just about model accuracy, but also about inference latency, energy use, and suitability for edge devices. This is especially relevant for civilian and security-camera systems, where models may need to run continuously on limited hardware rather than on powerful servers (Das et al., 2026). The lesson for PyroFinder is to compare YOLO11s against YOLO11n and document the accuracy/speed tradeoff.

The broader forest-fire surveillance review by Saleh et al. supports the same conclusion by showing that deep learning methods have largely outperformed classical approaches in forest-fire detection tasks. The review finds that many recent models exceed 90% accuracy and that YOLO-based detectors are among the most promising methods for real-time forest-fire surveillance, especially when combined with augmentation and lightweight backbones (Saleh et al., 2024). This paper influences PyroFinder by supporting the move from a descriptive dashboard to a predictive detection system.

Saleh et al. also note that forest-fire detection remains difficult because smoke can be thin, distant, or visually similar to clouds and other background structures. This means that real-world surveillance systems need not only a fast detector, but also a model designed to handle small objects, imbalanced data, and difficult visual conditions (Saleh et al., 2024). The lesson for PyroFinder is to include smoke-specific evaluation, background negatives, and false-positive review in the dashboard.

Overall, the literature suggests that accurate and fast real-time fire and wildfire detection from ordinary RGB cameras is best approached with YOLO-style object detection, possibly supported by a staging or hybrid pipeline. The strongest systems in the literature combine efficient real-time detectors with careful dataset design, augmentation, validation under difficult visual conditions, and architectural choices that improve robustness in complex scenes.

## Research Gap

Across these five papers, the key unresolved issue is transfer from benchmark datasets to real civilian or security-camera environments. Although deep learning and YOLO-based object detection have significantly improved fire and wildfire detection, the literature still shows limited real-world testing, dataset bias, weak generalization across scenes, and insufficient evaluation of false alarms in ordinary camera settings.

This leaves room for PyroFinder: a surveillance-focused system trained and evaluated for ordinary RGB camera feeds, with explicit model metrics, multi-frame confirmation, false-alarm review, and approximate location output.

## Paper Comparison and Project Lessons

| Paper | Type of Data | Data | Model | Result | Relation / Influence on PyroFinder | Lesson for PyroFinder |
|---|---|---|---|---|---|---|
| Bahhar et al. (2023) | UAV images, fire/smoke image datasets, video frames | Mixed fire/smoke datasets with limited high-quality real-world UAV images | Two-stage pipeline: ensemble CNN + YOLOv5s/YOLOv5l | Classification: accuracy 0.99, F1 0.95; detection: mAP@0.5 0.85 for smoke and 0.76 for combined detection | Shows that combining a classifier with a YOLO detector can reduce unnecessary detection work and improve robustness. This supports PyroFinder's future option for a staged pipeline after the MVP. | Data imbalance and smoke detection quality strongly affect performance; PyroFinder must track class balance, smoke-specific metrics, and false alarms. |
| Wicaksono et al. (2024) | RGB images | 3,104 annotated images from Roboflow Universe and web sources | YOLOv8 | mAP 0.63, precision 0.70, recall 0.57 | Demonstrates that a modern YOLO detector can identify fire and smoke from ordinary images, which supports PyroFinder's object-detection direction. | A small dataset and no real-world testing limit reliability; PyroFinder must validate beyond the training dataset and report limitations clearly. |
| Cheng et al. (2024) | Images and videos | Survey of public fire/smoke image and video benchmarks | Survey of deep learning methods, including YOLOv8 and improved variants | Concludes that YOLO-style detectors are fast and that attention/multiscale fusion can improve accuracy and reduce false alarms | Provides the strongest theoretical support for PyroFinder's two-class object-detection formulation: detect and localize `fire` and `smoke` instead of only classifying a frame. | Use detection metrics, not only accuracy: mAP, precision, recall, false alarm rate, and speed. |
| Saleh et al. (2024) | Images, videos, UAV, CCTV, and other surveillance sources | Review of 37 deep-learning forest-fire papers | Various CNN and YOLO-based detectors | Many studies report accuracy above 90%; YOLO-based methods are strong for real-time surveillance | Supports PyroFinder's move from passive camera viewing to automated detection using deep learning and a Streamlit monitoring dashboard. | Smoke can be small, distant, and visually similar to clouds/fog; PyroFinder needs background negatives, augmentation, and false-positive review. |
| Das et al. (2026) | UAV RGB, satellite, terrestrial RGB, edge-deployment datasets | Broad survey of RGB/UAV/satellite wildfire studies | YOLOv8 variants, hybrid CNN-Transformer models, lightweight detectors | Highlights tradeoff between accuracy, latency, and energy; reports improved YOLOv8 variants for small smoke and edge deployment | Influences PyroFinder's evaluation plan by making inference speed and deployability part of model selection, not only detection accuracy. | Benchmark YOLO11s against YOLO11n and document whether the main model is fast enough for near-real-time sampled-frame monitoring. |

## Practical Implications for PyroFinder

1. **Use object detection, not classification only.** PyroFinder needs bounding boxes because alerts require both detection and approximate location context.
2. **Keep the class schema simple.** The project should detect only `fire` and `smoke`; other objects can be used as background negatives but not as detection targets in the MVP.
3. **Measure both accuracy and operational performance.** The dashboard should report mAP@0.5, precision, recall, F1-score, false alarm rate, and inference speed.
4. **Handle false alarms explicitly.** Multi-frame confirmation, threshold tuning, and false-positive review are required because clouds, haze, glare, fog, dust, and lighting changes can resemble smoke or fire.
5. **Validate domain transfer.** D-Fire and supplementary datasets may not fully represent private-property cameras; PyroFinder should validate on additional images/videos and document known gaps.
6. **Compare YOLO11s and YOLO11n.** YOLO11s is the main model, but YOLO11n should be used as a speed baseline/fallback to understand the accuracy/speed tradeoff.

## Final Gap Statement

The literature provides strong evidence that YOLO-style deep learning object detection is a suitable foundation for real-time fire and smoke recognition. However, the main unsolved problem is reliable deployment in ordinary RGB civilian/security-camera conditions, where domain shift, false alarms, small smoke regions, and hardware constraints reduce reliability. PyroFinder addresses this gap by combining YOLO11s fire/smoke detection, D-Fire-based training, YOLO11n baseline comparison, multi-frame confirmation, false-positive review, and approximate map-based alerting for private property owners.

## References

Bahhar, C., Ksibi, A., Ayadi, M., Jamjoom, M. M., Ullah, Z., Soufiene, B. O., & Sakli, H. (2023). *Wildfire and Smoke Detection Using Staged YOLO Model and Ensemble CNN*. Electronics, 12(1), 228. https://doi.org/10.3390/electronics12010228

Cheng, G., Chen, X., Wang, C., Li, X., Xian, B., & Yu, H. (2024). *Visual fire detection using deep learning: A survey*. Neurocomputing, 596, 127975. https://doi.org/10.1016/j.neucom.2024.127975

Das, K., Poovvancheri, J., Flesca, S., Roberta Calidonna, C., & Chen, D. (2026). *Emerging Trends in Wildfire Detection Through the Lens of Computer Vision and Wildfire Emission Quantification: A Comprehensive Survey*. IEEE Access, 14, 20201-20228. https://doi.org/10.1109/ACCESS.2026.3660843

Saleh, A., Zulkifley, M. A., Harun, H. H., Gaudreault, F., Davison, I., & Spraggon, M. (2024). *Forest fire surveillance systems: A review of deep learning methods*. Heliyon, 10(1), e23127. https://doi.org/10.1016/j.heliyon.2023.e23127

Wicaksono, P., Yunanda, R., Arisaputra, P., & Izdihar, Z. N. (2024). *Deep Learning Wildfire Detection to Increase Fire Safety with YOLOv8*. International Journal of Intelligent Systems and Applications in Engineering, 12(3), 4383-4387.

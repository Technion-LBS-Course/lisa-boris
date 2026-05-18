# PyroFinder — Sprint Plan

**Course:** Technion 160833 — Location-Based Services: Data Science
**Team:** Lisa + Boris
**Semester:** Spring 2026

---

## Official Session Schedule

| Session | Date | Course Topic | PyroFinder Deliverable | Definition of Done |
|---|---|---|---|---|
| 3 — **M1** | 19/05 | Planning & README | All M1 files submitted to GitHub Classroom | README.md, CLAUDE.md, requirements.txt, .gitignore, .env.example, app.py, src/ skeleton, tests/, notebooks/ all committed and pushed; `streamlit run app.py` runs without errors |
| 4 | 26/05 | Data familiarity | D-Fire dataset loaded; class distribution visible in Streamlit | `app.py` Dataset & EDA tab shows real D-Fire class distribution bar chart (fire / smoke / background) and total image count from actual files in `data/` |
| 5 — **M2** | 02/06 | Pattern recognition | Full EDA dashboard live | At least 3 interactive visualizations in Streamlit: class distribution, bounding box size distribution, sample images with overlays; Data Card updated from actual file inspection; baseline metric (YOLO11n mAP@0.5) calculated and displayed |
| 6 | 09/06 | Value prediction | YOLO11n baseline fine-tuning complete | YOLO11n fine-tuned on D-Fire train split; mAP@0.5, Precision, Recall, F1, inference speed displayed in dashboard; experiment logged |
| 7 | 16/06 | Category prediction | YOLO11s fine-tuning + model comparison live | YOLO11s fine-tuned on D-Fire; side-by-side comparison of YOLO11s vs YOLO11n (mAP, Precision, Recall, F1, FAR, speed) visible in Streamlit; YOLO11s exceeds YOLO11n on mAP@0.5 |
| 8 — **M3** | 23/06 | Publish app | Public Streamlit Cloud URL live with real inference | URL active; upload an image → YOLO11s returns bounding-box overlay with fire/smoke labels and confidence scores; multi-frame confirmation logic wired; basic alert log shown |
| 9 | 30/06 | Dev marathon 1 | UX improvements + mentor feedback addressed | Mentor feedback from M3 reviewed and at least 2 UX/quality improvements committed; apparent direction estimation visible on inference output |
| 10 | 07/07 | Dev marathon 2 | Tests, validation, mapping prototype | All `pytest tests/` passing; camera metadata table live; at least one image polygon drawable on camera frame; approximate location shown in alert log |
| 11 | 14/07 | Dress rehearsal | Presentation ready + rehearsed in class | 5-slide deck finalized; demo script written; public URL stable; README finalized |
| 12 — **Final** | 21/07 | Demo Day | Full classroom presentation and defense | Live demo runs without errors; all milestones met; final report submitted |

---

## Milestone Exit Criteria

### M1 — The Pitch (19/05/2026)
- README.md complete in GitHub with all required sections
- Sprint Plan with weekly DoD committed
- GitHub repo active with correct folder structure
- CLAUDE.md ready
- 3-minute pitch prepared (4–5 slides)

### M2 — Data (02/06/2026)
- Streamlit dashboard loads real D-Fire data
- Dataset inspection and basic EDA working
- At least 3 visualizations in the dashboard
- Data Card updated based on actual file inspection
- Baseline metric (YOLO11n mAP@0.5) calculated and displayed

### M3 — Published Prototype (23/06/2026)
- Public Streamlit Cloud URL active
- YOLO11s model live with real inference on uploaded images/video
- YOLO11s performance above YOLO11n baseline on at least mAP@0.5
- Basic tests under `tests/` passing
- Alert/map prototype visible in dashboard

### Final — Demo Day (21/07/2026)
- Documented code
- Stable public demo URL
- Final report
- Presentation ready for classroom defense

---

*PyroFinder · Technion Course 160833 · Location-Based Services: Data Science · 2026*

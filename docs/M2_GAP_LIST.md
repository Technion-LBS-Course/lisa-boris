# PyroFinder — M2 Dashboard Audit Report

**Audit date:** 2026-05-28  
**Auditor role:** Shadow Mentor (D16) + UX Flow Agent (B6)  
**Sources of truth:** `docs/M2_dashboard.md`, `docs/AI_AGENT_SYSTEM.md`, `PROJECT_CONTEXT.md`, `CLAUDE.md`

---

## 4.1 — Tab-by-Tab Status Table

### Tab 1 — Problem Learning (`tab_problem` in M2 Course Dashboard mode)

| Required Element | Status | Notes |
|---|---|---|
| Problem in one non-technical sentence | ⚠️ Partial | Three-paragraph narrative present; no single isolated sentence |
| Stakeholder map | ❌ Missing | Replaced by a target-audience bullet list — not a map |
| Main persona with name and context | ✅ Present | Dani persona is clear and correctly framed |
| User journey **before** the solution | ❌ Missing | No "before" flow — use-case steps describe the ideal post-solution path |
| User journey **after** the solution | ❌ Missing | Same issue — steps describe the solution, not a before/after contrast |
| Value proposition in one sentence | ❌ Missing | Multiple explanatory paragraphs, no isolated VP sentence |
| `st.graphviz_chart()` flow diagram (suggested) | ❌ Missing | Not implemented |
| Before/After sub-tabs (suggested) | ❌ Missing | Not implemented |
| Persona card with optional image (suggested) | ⚠️ Partial | Bullet-point card present, no image |

---

### Tab 2 — Literature Review (`tab_lit`)

| Required Element | Status | Notes |
|---|---|---|
| Research question | ❌ Missing | Tab body is a single `st.info("Coming soon …")` |
| 3–5 academic sources with APA citations | ❌ Missing | Nothing |
| Comparison table (dataset / model / result) | ❌ Missing | Nothing |
| Research gap statement | ❌ Missing | Nothing |
| One-line project impact per source | ❌ Missing | Nothing |

---

### Tab 3 — Market Review (`tab_market`)

| Required Element | Status | Notes |
|---|---|---|
| 3–5 competitors with comparison table | ❌ Missing | Tab body is a single `st.info("Coming soon …")`. A competitor table exists in Tab 1, not here |
| Pricing / target audience columns | ❌ Missing | — |
| Competitor screenshots | ❌ Missing | — |
| Positioning map (e.g., 2×2 or scatter) | ❌ Missing | — |
| Design insights (adopt / replace / avoid) | ❌ Missing | — |
| SCAMPER exercise on one competitor | ❌ Missing | — |
| Project differentiation statement | ❌ Missing | — |

---

### Tab 4 — Dataset & EDA (`tab_eda_story`)

| Required Element | Status | Notes |
|---|---|---|
| Data Card — source URL | ❌ Missing | Caption mentions class mapping; no formal card |
| Data Card — license | ❌ Missing | — |
| Data Card — access date | ❌ Missing | — |
| Data Card — size and structure | ⚠️ Partial | Mentioned in chart captions/code comments, not as a card |
| Data Card — file format and schema | ❌ Missing | — |
| Data Card — known gaps / biases | ❌ Missing | — |
| Variable types (df.info() style) | ❌ Missing | — |
| Descriptive statistics (df.describe() style) | ❌ Missing | — |
| Missing values analysis | ❌ Missing | — |
| Dataset preview (df.head()) | ❌ Missing | The Operations tab has a preview; M2 story tab does not |
| **Viz 1 — Target distribution** | ✅ Present | Chart 1: category bar chart with `st.info()` insight |
| **Viz 2 — Correlation heatmap** | ❌ Missing | Correlation explorer is in Operations tab only, not in M2 tab |
| **Viz 3 — Domain-specific (CV)** | ✅ Present | Charts 3–5: pixel stats, bbox area, smoke-fire scatter — all with insights |
| Written insight below each chart | ✅ Present | All 5 charts have `st.info()` insight blocks |
| Outlier boxplot | ⚠️ Partial | Violin plots cover this partially; explicit outlier framing absent |

---

### Tab 5 — KPI & Metrics (`tab_kpi`)

| Required Element | Status | Notes |
|---|---|---|
| Model output type stated (regression vs classification) | ❌ Missing | Not explicitly addressed in KPI tab |
| Error cost — false positive defined | ❌ Missing | Not shown |
| Error cost — false negative defined | ❌ Missing | Not shown |
| KPI selection logic (3 questions from deck) | ❌ Missing | Not implemented |
| Chosen primary KPI (one metric chosen above others) | ❌ Missing | All 7 metrics are shown as equal; no one is chosen as the primary KPI |
| `st.success()` with "The model is ___, metric is ___, because ___" | ❌ Missing | — |
| All 7 metrics displayed | ✅ Present | Full metric grid rendered with placeholder cards |
| Metrics clearly marked as placeholder | ✅ Present | "N/A — awaits M3 training" |
| Metric interpretations | ✅ Present | `st.caption()` for each metric |

**README.md — Required KPI line:**  
`"The model is ___, the metric is ___, because ___."` → ❌ **Not present** in README.md

---

## 4.2 — M2 Definition of Done Checklist (AI_AGENT_SYSTEM.md §13)

- [x] Streamlit dashboard loads without errors on `streamlit run app.py`
- [ ] **Tab 1** — Problem, Dani persona, no forbidden wording → ⚠️ Persona ✅, no forbidden wording ✅, but before/after and VP missing
- [ ] **Tab 2** — 3 paper summaries and differentiation table → ❌ Tab is empty
- [ ] **Tab 3** — Market gap and competitor comparison → ❌ Tab is empty
- [x] **Tab 4** — 3+ visualizations each with written insight → ✅ Has 5 charts with insights
- [ ] **Tab 5** — All 7 metrics as placeholders → ✅ Present, but KPI selection logic and README line missing
- [x] D-Fire counts match CLAUDE.md (21,527 images, class 0=smoke, class 1=fire)
- [ ] `python -m pytest` passes — unverified in this audit (needs a live run)
- [x] No raw image paths hardcoded (uses `METADATA_PATH` variable)
- [x] No model weights loaded at import time
- [ ] `.gitignore` correctness — unverified in this audit (CLAUDE.md states it was updated in commit `83d4aff`)
- [ ] README.md M2 section updated → ✅ M2 progress section present, but KPI line missing
- [ ] `git tag m2-final` created → ❌ Not yet (expected at submission)

---

## 4.3 — Critical Gaps (P0/P1)

### P0 — Blocks Submission

**Gap 1: Literature Review tab is completely empty**
- What: No research question, no academic sources, no comparison table, no gap statement.
- File: `app.py:914-915` — replace `st.info("Coming soon …")` with real content.
- What to add: Research question, 3–5 papers (real DOI/arXiv links), comparison table with columns `Source | Dataset | Model | Metric/Result | Main Limitation | Impact on PyroFinder`, and a research gap statement.

**Gap 2: Market Review tab is completely empty**
- What: No competitor table, no positioning, no SCAMPER, no differentiation.
- File: `app.py:918-919` — replace `st.info("Coming soon …")` with real content.
- What to add: 3–5 competitors in a table (`Competitor | Type | Features | Audience | Pricing | Gap | What We Learn`), a 2×2 positioning chart, and a SCAMPER walkthrough for one competitor (Pano AI is the best candidate).

**Gap 3: Required KPI README line missing**
- What: `docs/M2_dashboard.md §7` requires one line in `README.md`: `"The model is ___, the metric is ___, because ___."` — not present.
- File: `README.md` — add after the Formal ML Problem section.
- What to add: `"The model is a two-class object detection model (YOLO11s), the metric is Recall, because missing a confirmed fire event is more dangerous than triggering a false alarm."`

---

### P1 — Important for Grade Quality

**Gap 4: Correlation heatmap missing from M2 EDA tab**
- What: `docs/M2_dashboard.md §6` requires exactly three visualizations including a correlation heatmap. The Operations tab has a Pearson correlation explorer, but `tab_eda_story` does not.
- File: `app.py` — in `tab_eda_story`, add a compact correlation heatmap using the same columns already used in the Operations tab correlation explorer (`total_boxes`, `fire_bbox_coverage`, `smoke_bbox_coverage`, `mean_brightness`, `dark_pixel_ratio`).

**Gap 5: Data Card missing from EDA tab**
- What: The minimum EDA checklist requires source URL, license, access date, size, format, known gaps, and biases as a formal card — not scattered in captions.
- File: `app.py` — add a `st.subheader("Data Card")` + `st.markdown()` block at the top of `tab_eda_story` with all fields from `docs/M2_dashboard.md §6`.

**Gap 6: Before/after user journey and value proposition missing from Problem tab**
- What: Required elements — stakeholder map, before-solution journey, after-solution journey, value proposition sentence — are absent. The tab has good content but skips these structural M2 requirements.
- File: `app.py` — in `tab_problem`, add `st.graphviz_chart()` for before/after flow, and a `st.success()` or `st.metric()` for the one-sentence value proposition.

---

## 4.4 — Improvement Recommendations

### What should be focused / sharpened (content exists but is too broad)

- **KPI tab:** The 7-metric grid is good implementation but misses the instructor's intent — the KPI tab should demonstrate a *decision*, not a list. Add the three KPI selection questions, answer them for PyroFinder, and declare one primary metric (Recall) with an explicit rationale. The other metrics can remain as supporting context.

- **Problem tab opening:** The problem statement is currently a multi-paragraph block. Distill it to one non-technical sentence at the very top (`st.subheader("The Problem")` → `st.success("One sentence.")`), then expand into paragraphs below. This is the first thing the instructor reads.

- **Problem tab competitive table:** The `comp_df` table in the Problem tab is good, but it belongs in the Market Review tab, not here. The Problem tab should focus on the user and the pain — move competitive content to Tab 3 and clean up Tab 1.

### What should be reduced / cut (adds noise)

- **Problem tab — Risks table:** The risks section (`risk_df`, lines 900–911) is M3/technical information. The instructor's M2 rubric does not include risks in the Problem tab. Consider moving it to an expander or removing it from the primary view.

- **M2 mode is buried in a sidebar dropdown:** The M2 Course Dashboard is one of three modes in a sidebar selector. At a classroom demo, the instructor needs to arrive at the right dashboard instantly. Consider making M2 Course Dashboard the default `mode` value, or restructuring so it is the landing page.

### What should be improved (exists but below standard)

- **Persona card:** The Dani persona is in a plain `st.markdown()` bullet list on the left column. `docs/M2_dashboard.md` suggests a persona card with an optional image. Even without an image, a bordered `st.container()` with a metric header ("Dani — Farm Owner") and structured bullets is visually much stronger at projector resolution.

- **User journey:** The step-by-step expanders (lines 823–832) show the post-solution flow. This is useful, but the M2 rubric calls for a visible contrast: *what happens today without PyroFinder* vs. *what happens after*. A simple `st.tabs(["Before", "After"])` within Tab 1, each with a `st.graphviz_chart()`, would satisfy the requirement efficiently.

### What should be added (completely missing)

- Literature Review content (academic papers, table, gap) — see Gap 1 above.
- Market Review content (competitors, positioning, SCAMPER) — see Gap 2 above.
- Data Card in EDA tab — see Gap 5 above.
- Correlation heatmap in EDA tab — see Gap 4 above.
- KPI selection logic with three decision questions in KPI tab.
- README KPI line — see Gap 3 above.

---

## 4.5 — Recommended Build Order

| Priority | Task | Agent | File(s) | Effort |
|---|---|---|---|---|
| 1 | Build Literature Review tab content | A2 — Literature Review Agent | `app.py:914-915` | Large |
| 2 | Build Market Review tab content | A3 — Market Review Agent | `app.py:918-919` | Medium |
| 3 | Add README KPI line | A5 — KPI Agent | `README.md` | Small |
| 4 | Add before/after journeys + value prop to Problem tab | A1 — Product & Problem Agent | `app.py:771-911` | Medium |
| 5 | Add Data Card section to EDA tab | A4 — Data & EDA Agent | `app.py:922-930` | Small |
| 6 | Add correlation heatmap to EDA tab | A4 — Data & EDA Agent | `app.py:930-1086` | Small |
| 7 | Add KPI selection logic + chosen metric to KPI tab | A5 — KPI Agent | `app.py:1087-1183` | Small |
| 8 | Move competitor table from Problem tab → Market tab | B6 — UX Flow + B8 Layout | `app.py:875-894` | Small |
| 9 | Make M2 Course Dashboard the default landing mode | B8 — Streamlit Layout Agent | `app.py:46-54` | Small |
| 10 | Run full test suite + verify `.gitignore` | C13 — Testing & QA Agent | `tests/test_smoke.py` | Small |

---

## 4.6 — Conflicts or Ambiguities

| # | Conflict | M2_dashboard.md says | AI_AGENT_SYSTEM.md says | Recommended authority |
|---|---|---|---|---|
| 1 | Visualization count | "Use **exactly three** focused visualizations" (§6) | "At least 3 visualizations" (§13) | **M2_dashboard.md spirit, not the letter.** The current 5 charts with clear insights exceed the minimum and meet the spirit. Do not cut charts — keep all 5. Add the missing correlation heatmap to reach the required three types. |
| 2 | Market Review content | Requires positioning map, SCAMPER, screenshots, pricing column | Only requires "market gap description and competitor comparison" (§13) | **M2_dashboard.md.** The instructor's deck is the exam rubric. |
| 3 | Literature Review vs. Related Work | Requires 3–5 *academic papers* with APA + DOI | Mentions "differentiation table comparing Pano AI, FIREWAVE, CANDO" — which are products, not papers | **M2_dashboard.md.** Tab 2 needs real academic papers (arXiv/DOI); Pano AI/FIREWAVE/CANDO belong in Tab 3 (Market Review). Do not conflate the two. |
| 4 | Dashboard structure | Implies a flat 5-tab app as the primary view | Treats M2 Course Dashboard as one of three sidebar modes | **M2_dashboard.md for the demo.** The current sidebar-mode architecture is fine for development, but the M2 Course Dashboard should be the default or clearly primary for submission. |

---

## Executive Summary

**Current M2 readiness: Medium-Low.** The foundation is solid — the EDA tab has five strong charts with real data-driven insights, the Dani persona is well-presented, the UI theme is applied consistently, and the codebase is clean. However, two of the five required tabs (Literature Review and Market Review) are completely empty placeholders, and the Problem and KPI tabs are missing specific structural elements that appear on the grading rubric: before/after user journeys, a value proposition sentence, KPI selection logic, and the required README line.

**Top 3 actions before submission (due 2026-06-02):**

1. **Build Literature Review** — 3 real academic papers with DOI/arXiv links, comparison table, research gap. This is the single largest gap and requires human research for verified citations.
2. **Build Market Review** — 3+ competitors in a structured table, 2×2 positioning chart, SCAMPER on Pano AI, project differentiation.
3. **Add the README KPI line + KPI selection logic in the tab** — one sentence in `README.md` and three decision questions in the KPI tab convert a metrics list into an evidence-based argument.

**Confidence that M2 can be submitted on time: Medium.** Items 2 and 3 above are achievable in a single focused session each. Item 1 (Literature Review) requires real academic sources with verified DOIs, which is human research work that cannot be generated — this is the primary risk factor for the timeline.

---

*PyroFinder · M2 Gap Audit · Technion Course 016833 · Audited: 2026-05-28*

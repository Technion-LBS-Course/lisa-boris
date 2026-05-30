# M2 Dashboard — Requirements for Claude / Claude Code

Source deck: `l4_Dashboard.pptx`  
Course: 00160833 — Location-Based Services: Data Science, Technion  
Lecture: Meeting 04 — M2 Dashboard  
Instructor: Dr. Eli Safra  
Date in deck: 26 May 2026  
M2 due date in deck: 02 June 2026

---

## 1. Core Message

> Before solving the problem, prove that you understand it.  
> The dashboard is the evidence — not a visualization exercise.

The M2 dashboard is **not only EDA**. It is the opportunity to prove that the team understands the project problem end-to-end before training or improving a model.

The dashboard should answer four questions:

1. **Problem** — Who suffers from what, and what do they do today without the solution?
2. **Literature** — What has already been studied, which approaches were tested, and where is the gap?
3. **Market** — Who is already trying to solve the problem, and how is this project different?
4. **Data / EDA** — What data do we have, and what does it reveal about the problem?

---

## 2. Required M2 Dashboard Structure

Recommended Streamlit structure:

```python
import streamlit as st

st.set_page_config(page_title="M2 Dashboard", layout="wide")

problem_tab, literature_tab, market_tab, eda_tab, kpi_tab = st.tabs([
    "Problem Learning",
    "Literature Review",
    "Market Review",
    "Dataset & EDA",
    "KPI"
])
```

Minimum required tabs for submission: **4 tabs**.  
Recommended tabs for clarity: **5 tabs**, with KPI separated.

---

## 3. Tab 1 — Problem Learning

### Goal

Show that the team understands the real-world weakness / pain point before jumping into technical implementation.

This is the section students often want to skip, but it separates a real-world project from a purely academic or technical exercise.

### Required Content

Include:

- Problem in one non-technical sentence.
- Stakeholder map.
- Main persona with name and context.
- User journey **before** the solution.
- User journey **after** the solution.
- Value proposition in one sentence.

### Suggested Streamlit Implementation

Use:

- `st.tabs()` to separate **Before** and **After** user journeys.
- `st.graphviz_chart()` for a simple flow diagram.
- Persona card with short bullet points and, optionally, an AI-generated persona image.

Example layout:

```python
with problem_tab:
    st.header("Problem Learning")
    st.subheader("Problem in One Sentence")
    st.write("...")

    st.subheader("Main Persona")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("path/to/persona.png", caption="Primary persona")
    with col2:
        st.markdown("""
        **Name:** ...  
        **Role:** ...  
        **Pain:** ...  
        **Current workaround:** ...
        """)

    before, after = st.tabs(["Before", "After"])
    with before:
        st.graphviz_chart("""
        digraph {
            User -> Problem -> Manual_Workaround -> Delay
        }
        """)
    with after:
        st.graphviz_chart("""
        digraph {
            User -> System -> Alert -> Faster_Response
        }
        """)
```

### Useful Prompt for Claude

```text
Analyze my project problem using a before/after user journey.
Include stakeholder map, main persona, current workaround, post-solution journey,
and a one-sentence value proposition.
```

---

## 4. Tab 2 — Literature Review

### Goal

Show what has already been researched so the project does not reinvent the wheel, and so the project can connect to existing academic work.

### Required Content

Include:

- Research question: what exactly is being investigated?
- 3–5 academic sources with APA citations.
- Comparison table: dataset / model / result.
- The gap that no one fully solved.
- One line explaining how each source affects this project.

### Where to Search

Recommended sources:

- Google Scholar — default academic search.
- arXiv — especially for ML topics.
- Semantic Scholar.
- ResearchGate.
- Technion Library — for access to paid academic databases.

### Important Warning

AI models may invent citations.  
Every academic source must have a real DOI, publisher page, arXiv link, or other verifiable URL.

### Suggested Streamlit Implementation

```python
with literature_tab:
    st.header("Literature Review")
    st.subheader("Research Question")
    st.write("...")

    st.subheader("Academic Sources")
    st.dataframe(literature_df, use_container_width=True)

    st.subheader("Research Gap")
    st.info("...")

    st.subheader("Impact on Our Project")
    st.write("...")
```

Suggested comparison table columns:

| Source | Dataset | Model / Method | Metric / Result | Main Limitation | Impact on Our Project |
|---|---|---|---|---|---|

---

## 5. Tab 3 — Market Review

### Goal

Show what exists in the market today: competitors, products, startups, and partial solutions.

This section is not about academic research. It is about real products and practical alternatives.

### Required Content

Include:

- 3–5 competitors, including partial solutions.
- Comparison table: features / price / target audience.
- Screenshots of central features when available.
- Positioning: where this project sits on the map.
- Design insights: what to adopt, what to replace, what to avoid.

### SCAMPER Exercise

Use one competitor and ask:

| Letter | Question |
|---|---|
| S — Substitute | What should we replace? |
| C — Combine | What should we combine? |
| A — Adapt | What should we adapt? |
| M — Modify | What should we modify? |
| P — Put to another use | How can we use it differently? |
| E — Eliminate | What should we remove? |
| R — Reverse / Rearrange | What should we rearrange? |

### Suggested Streamlit Implementation

```python
with market_tab:
    st.header("Market Review")
    st.subheader("Competitor Comparison")
    st.dataframe(market_df, use_container_width=True)

    st.subheader("Positioning Map")
    st.plotly_chart(positioning_fig, use_container_width=True)

    st.subheader("SCAMPER Insights")
    st.markdown("""
    - **Substitute:** ...
    - **Combine:** ...
    - **Adapt:** ...
    - **Modify:** ...
    - **Put to another use:** ...
    - **Eliminate:** ...
    - **Rearrange:** ...
    """)
```

Suggested comparison table columns:

| Competitor | Product Type | Core Features | Target Audience | Pricing / Business Model | Weakness / Gap | What We Learn |
|---|---|---|---|---|---|---|

---

## 6. Tab 4 — Dataset & EDA

### Goal

Show what the dataset contains and what it reveals about the problem. The EDA should be focused, not overloaded.

Quality is more important than quantity. Every chart must lead to an insight that helps choose a model, metric, feature, threshold, or validation strategy. A beautiful chart without an insight is noise.

### Minimum EDA Checklist

Every project must include:

- Data source: link, license, and access date.
- Size and structure: number of rows, columns, files, images, or records.
- File format and schema.
- Variable types, similar to `df.info()`.
- Descriptive statistics, similar to `df.describe()`.
- Missing values: how many, where, and why they matter.
- Target variable distribution — required.
- Outliers: boxplot for numeric features when relevant.
- Correlations: heatmap for numeric features when relevant.

### Three Required Visualizations

Use **exactly three focused visualizations** for the core M2 dashboard. Each one must lead to a decision.

#### Visualization 1 — Target Distribution

Use:

- Histogram for regression targets.
- Bar chart for classification targets.

Purpose: Check whether the target is balanced or imbalanced. This affects metric choice, baseline, and modeling strategy.

#### Visualization 2 — Correlation Heatmap

Use for numeric variables.

Purpose: Identify multicollinearity and estimate which features may be useful.

#### Visualization 3 — Domain-Specific Visualization

Choose based on project domain:

- Geographic project → map, e.g. `folium`.
- Time-based project → time series.
- Text project → word cloud or term-frequency view.
- Computer vision project → sample image grid, bounding-box examples, class / object-size distribution.

Purpose: Show what makes the project unique.

### Suggested Streamlit Implementation

```python
with eda_tab:
    st.header("Dataset & EDA")

    st.subheader("Data Card")
    st.markdown("""
    - **Source:** ...
    - **License:** ...
    - **Access date:** ...
    - **Size:** ...
    - **Format:** ...
    - **Known gaps / biases:** ...
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Target Distribution")
    st.plotly_chart(target_distribution_fig, use_container_width=True)
    st.info("Insight: ...")

    st.subheader("Correlation Heatmap")
    st.plotly_chart(correlation_fig, use_container_width=True)
    st.info("Insight: ...")

    st.subheader("Domain-Specific Visualization")
    st.plotly_chart(domain_fig, use_container_width=True)
    st.info("Insight: ...")
```

---

## 7. Tab 5 — KPI and Metric Choice

### Goal

Translate the problem into one measurable success criterion.

The dashboard should make clear which metric will be used to prove model success.

### KPI Selection Logic

Ask three questions:

1. **What is the model output?**
   - Number → regression.
   - Category → classification.
2. **What is the cost of an error?**
   - False positive: false alarm.
   - False negative: missed event.
3. **What does the target distribution look like?**
   - Balanced → Accuracy may be acceptable.
   - Imbalanced → F1 or Recall may be better.

### Common Metrics

| Problem Type | Metric | When to Use |
|---|---|---|
| Regression | RMSE | Error in the same units as the target, such as ₪, degrees, or millimeters |
| Regression | MAE | Robust to outliers and easy to explain to users |
| Classification | Accuracy | Only when classes are balanced, roughly 50/50 |
| Classification | F1 | Imbalanced classes; balances Precision and Recall |
| Classification | Recall | When missing a real case is dangerous, such as safety or medicine |

### Required README Line

Add one clear line to `README.md`:

```text
The model is ___, the metric is ___, because ___.
```

Example:

```text
The model is a classification model, the metric is Recall, because missing a real safety event is more dangerous than a false alarm.
```

### Suggested Streamlit Implementation

```python
with kpi_tab:
    st.header("KPI Selection")
    st.subheader("Model Output")
    st.write("...")

    st.subheader("Error Cost")
    st.write("False Positive cost: ...")
    st.write("False Negative cost: ...")

    st.subheader("Chosen KPI")
    st.success("The model is ___, the metric is ___, because ___.")
```

---

## 8. M2 Submission Checklist

M2 due date from the deck: **02/06/2026**.

Before submission, verify that the dashboard includes:

- [ ] **Problem Learning** — persona, before/after story, and value proposition in one sentence.
- [ ] **Literature Review** — 3 cited papers, comparison table, and one lesson from each.
- [ ] **Market Review** — 3 competitors in a table, positioning map, and project differentiation.
- [ ] **EDA** — minimum checklist, 3 visualizations, and 3 insights.
- [ ] **KPI Definition** — one README line: `The model is X, the metric is Y, because Z`.
- [ ] **Streamlit App** — 4 tabs minimum, either public URL or local demo.

---

## 9. Recommended Claude Code Prompt

Use this prompt when asking Claude Code to implement or update the M2 dashboard.

```text
Use PROJECT_CONTEXT.md as the source of truth for the project.
Use M2_dashboard.md as the source of truth for the M2 dashboard requirements.

Goal:
Update the Streamlit app so it satisfies the M2 dashboard requirements.
The dashboard is not only EDA. It must prove that we understand the problem, literature, market, data, and KPI.

Required Streamlit tabs:
1. Problem Learning
2. Literature Review
3. Market Review
4. Dataset & EDA
5. KPI

For each tab:
- Add clear headings and concise explanations.
- Prefer tables, cards, and focused visualizations.
- Every visualization must include a written insight.
- Do not invent citations, dataset facts, or competitor claims.
- Add TODO placeholders only where real information is still missing.

EDA requirements:
- Show Data Card: source, license, date, size, format, known gaps, possible biases.
- Show target distribution.
- Show one correlation or feature relationship visualization when applicable.
- Show one domain-specific visualization.
- Include exactly 3 core insights from the data.

KPI requirements:
- Explain model type.
- Explain FP vs FN cost.
- Explain chosen metric.
- Add or update the README line: "The model is X, the metric is Y, because Z."

Constraints:
- Keep code modular.
- Do not load heavy ML models on import.
- Do not require unavailable local datasets to run the app.
- If data is missing, show a clear placeholder and keep the app running.
- Make sure `python -m pytest tests` still passes.
```

---

## 10. Workshop Flow from the Deck

The workshop in the deck suggests building the M2 dashboard skeleton in pairs. The output is a plan, not necessarily full Streamlit implementation during class.

Workflow:

1. **Problem Learning** — persona, before/after, value proposition.
2. **3 Literature Titles** — use Perplexity / Scholar; do not over-read at this stage.
3. **3 Competitors** — quick search; names + 3 features.
4. **From Data to Visualization** — identify where the data is and what the unique visual should be.
5. **Short Share** — 30 seconds from one or two teams.

---

## 11. Slide-by-Slide Content Summary

### Slide 1 — Title

M2 Dashboard: What does your dashboard tell? Understand the problem before building a model.

### Slide 2 — Main Message

Before solving, make sure you understand the problem. The dashboard is evidence of that understanding, not a visualization exercise.

### Slide 3 — Four Pillars

The dashboard is not just EDA. It proves understanding across: problem learning, literature review, market review, and EDA.

### Slide 4 — Problem Learning

Required: non-technical problem sentence, stakeholder map, persona, before journey, after journey, and value proposition. Suggested Streamlit tools: tabs, graphviz, persona card.

### Slide 5 — Literature Review

Required: research question, 3–5 academic sources in APA, comparison table, research gap, and impact on project. Search in Scholar, arXiv, Semantic Scholar, ResearchGate, and Technion Library. Do not trust AI-generated citations without DOI or real link.

### Slide 6 — Market Review

Required: 3–5 competitors, comparison table, screenshots, positioning, and design insights. Use SCAMPER to analyze competitors.

### Slide 7 — EDA Checklist

Required: descriptive statistics, target distribution, outliers, correlations, source, size, structure, variable types, and missing values. Every chart must produce a useful insight.

### Slide 8 — Three Required Visualizations

Target distribution, correlation heatmap, and one domain-specific visualization such as map, time series, word cloud, or computer-vision sample view.

### Slide 9 — Workshop

Build a dashboard skeleton: problem learning, literature titles, competitors, data-to-visualization plan, and short sharing.

### Slide 10 — KPI Selection

Move from problem to one number. Decide model output, error cost, and target distribution. Metric depends on regression/classification, FP/FN cost, and balance.

### Slide 11 — Common Metrics

Regression: RMSE, MAE.  
Classification: Accuracy, F1, Recall.

### Slide 12 — M2 Assignment Checklist

Problem learning, literature review, market review, EDA, KPI, and Streamlit tabs. Due 02/06/2026.

### Slide 13 — Takeaway

Measure before training. M2 dashboard proves problem understanding. Clear KPI is the promise of how success will be proven.

# Assistant Working Rules

## Communication Style

- Be direct, precise, and critical when needed.
- Do not flatter the user without reason.
- Do not invent facts, citations, metrics, datasets, links, or implementation details.
- If information is missing, say what is missing.
- If a suggestion is risky, unnecessary, or out of scope, say so clearly.

## Language Rules

- Conversation with the user should be in Hebrew unless the user asks otherwise.
- Commands, prompts, filenames, code comments, application text, documentation text, UI text, and website/software copy must be in English.
- Do not mix Hebrew into code, file names, prompts, markdown files intended for the project, Streamlit UI text, or documentation content.

## Source and Accuracy Rules

- Prefer project source files over memory.
- Use README.md, PROJECT_CONTEXT.md, CLAUDE.md, and AI_AGENT_SYSTEM.md as source-of-truth files.
- Do not cite or reference sources unless they are real, relevant, and verified.
- When using academic papers, use only real papers with valid publication details, DOI, URL, or uploaded PDF evidence.
- Mark uncertain claims as uncertain.

## PyroFinder Project Rules

- PyroFinder uses existing cameras; do not introduce new dedicated hardware unless explicitly marked as future/out of scope.
- Use YOLO11s as the current primary detector.
- Use YOLO11n only as the lightweight speed baseline/fallback.
- Do not write generic “YOLO” when the model version matters.
- Classes are strictly `fire` and `smoke`.
- Location outputs are approximate only.
- Do not describe PyroFinder as an “early warning system.”
- Do not claim precise geolocation, automatic image-to-map registration, emergency dispatch integration, or real fire-spread prediction in the MVP.

## Coding Rules

- Keep Python code modular, readable, and testable.
- Use English names for files, variables, functions, comments, and UI text.
- Do not load heavy ML models at import time.
- Do not commit large datasets, model weights, secrets, or local machine paths.
- After code changes, run:
  python -m pytest tests
- After Streamlit layout changes, run:
  streamlit run app.py

## Session Rules

- At the start of a task, inspect the relevant project files first.
- Before making broad changes, identify the exact files that should change.
- Prefer small, safe steps over large rewrites.
- Tell the user when a source file appears outdated and should be updated.
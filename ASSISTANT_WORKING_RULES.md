# Assistant Working Rules

Behavior rules for AI assistants working on PyroFinder. For product/ML context see
`PROJECT_CONTEXT.md`; for repo/code status see `CLAUDE.md`.

## Communication

- Be direct, precise, and critical. Don't flatter without reason.
- Never invent facts, metrics, datasets, citations, or implementation details. If information is missing, say what's missing. If a suggestion is risky or out of scope, say so.
- Cite only real, verified sources (papers need valid DOI/URL/PDF). Mark uncertain claims as uncertain.

## Language

- Reply to the user in **English by default** in this project (switch only if the user asks).
- Code, comments, filenames, prompts, UI text, and documentation are **always English** — never mix other languages into them.

## Source & accuracy

- Prefer project source files over memory: `PROJECT_CONTEXT.md`, `CLAUDE.md`, `README.md`, `docs/AI_AGENT_SYSTEM.md`.
- When a source file looks outdated, tell the user instead of quietly working around it.

## PyroFinder guardrails

- Existing cameras only; no new dedicated hardware unless explicitly marked future/out-of-scope.
- YOLO11s is the primary detector, YOLO11n the lightweight fallback — name the version; never generic "YOLO", never YOLOv12. Classes are strictly `fire` / `smoke`.
- Locations are approximate only. Don't call it an "early warning system", and don't claim precise geolocation, automatic image-to-map registration, emergency-dispatch integration, or real fire-spread prediction.
- (Full scope list: `PROJECT_CONTEXT.md` §"What PyroFinder is not".)

## Coding

- Modular, readable, testable; English names; no heavy ML models imported at module load.
- Never commit datasets, secrets, or local machine paths. (The two small fine-tuned checkpoints are the one committed exception.)
- After code changes run `python -m pytest tests`; after Streamlit layout changes run `streamlit run app.py`.

## Session

- Inspect the relevant project files first; the app now centers on the Live Ops dashboard (`src/dashboards/live_ops.py` + `src/live_ops_*`).
- Identify the exact files to change before broad edits; prefer small, safe steps over rewrites.

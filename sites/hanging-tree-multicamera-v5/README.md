# PyroFinder Hanging Tree multi-camera site

This is the isolated vinext/Sites implementation for the PyroFinder multi-camera
prototype. Hanging Tree 1 is the default camera and Thunder Valley West remains
the second selectable camera.

Production URL:

- https://pyrofinder-hanging-tree-v5.lisaborisclark.chatgpt.site

The separate Sites project is declared in `.openai/hosting.json`. Do not replace
that project ID with the original Sites v4 project ID, and do not deploy this
source over the original public PyroFinder site.

## Local development

Requirements: Node.js 22.13 or newer.

```powershell
npm install
npm run dev
npm run lint
npm test
```

The application uses two production secrets, configured in Sites rather than in
Git:

- `OPENWEATHER_API_KEY` for synchronized current weather and wind
- `GROQ_API_KEY` for live Ops chat responses

When a service is unavailable, the application uses an explicit prepared or safe
fallback and never commits a secret locally.

## Project structure

- `app/`: multi-camera UI, map, weather API, chat API, and isolated camera state
- `app/data/`: prepared camera mapping and detection result JSON
- `public/`: camera references and frame sequences
- `tests/`: SSR and implementation guardrail tests
- `tools/mock-fire-remotion/`: deterministic Hanging Tree simulated-fire frame generator

## Detector provenance

The original PyroFinder project selected an Ultralytics YOLO11s checkpoint that
was fine-tuned on D-Fire. In the original repository the checkpoint is stored at
`models/yolo11s_dfire_best.pt`.

This Sites application does not run that PyTorch checkpoint in the browser or at
the edge. It replays prepared per-frame result JSON. The Hanging Tree simulated
fire detections are scenario-aligned prepared results, not a fresh inference run
of the checkpoint. The UI states this distinction.

## Embedded location in the original repository

The shared copy lives at `sites/hanging-tree-multicamera-v5/`. Run all Node
commands from that directory. Its dependencies, build output, hosting metadata,
and deployment remain isolated from the original Streamlit application at the
repository root.

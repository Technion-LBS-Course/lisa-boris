# PyroFinder Hanging Tree multi-camera site

This is the isolated vinext/Sites implementation for the PyroFinder multi-camera
prototype. Hanging Tree 1 is the default camera and Thunder Valley West remains
the second selectable camera.

Production URL:

- https://pyrofinder-hanging-tree-v5.lisaborisclark.chatgpt.site

## Canonical release and archive policy

The release tagged `sites-canonical-2026-07-19` is the only version intended for
presentation. The production URL above points to that release. Its isolated copy
in the original PyroFinder repository is
`sites/pyrofinder-hanging-tree-live/`.

Earlier Sites versions are retained only as read-only rollback artifacts. They
are not alternate presentation links and should not be edited or deployed as the
current app. Their exact source commits are preserved by the Git tags
`archive/sites-v01` through `archive/sites-v11`; the Sites version history keeps
the corresponding deployable archives.

## Viewing detection results

In the application, finish the three setup steps and open **Live**. Press play or
use the frame buttons. Smoke markers are amber, fire markers are orange, and a
combined marker keeps the stronger fire treatment with a smoke accent. Select
**Hanging Tree 1** or **Thunder Valley West** in the camera picker to inspect
each isolated sequence. Confidence sliders filter the verified offline model
candidates; lowering a slider to 5% shows every candidate collected by the
offline inference run. Resolved alerts appear under **History** for the selected
camera.

The machine-readable results and provenance are in:

- `app/data/hanging-tree-yolo11s-results.json` (28 frames)
- `app/data/thunder-valley-yolo11s-results.json` (26 frames)

The public URL above reflects only the last deployed revision. Branch changes
must pass review and be deployed separately; this repository does not describe
precomputed inference as live browser-side YOLO.

The separate Sites project is declared in `.openai/hosting.json`. Do not replace
that project ID with the original Sites v4 project ID, and do not deploy this
source over the original public PyroFinder site.

## Local development

Requirements: Node.js 22.13 or newer. Regenerating detector outputs additionally
requires Python and the pinned packages in `tools/yolo-inference/requirements.txt`.

```powershell
npm install
npm run dev
npm run lint
npm test
```

`npm test` first creates a production build and then runs 23 behavior and
integration tests. The suite executes detection priority and temporal logic,
exact incident guidance, camera-to-map geometry, wind and fire-risk calculations,
the built SSR/API worker, SHA-256 correspondence for every displayed frame, and
the measured fire progression in the authentic YOLO output. It does not treat
source-code string searches as evidence that a feature works.

The two checked-in model result files can also be verified independently:

```powershell
py tools/yolo-inference/validate_results.py --results app/data/hanging-tree-yolo11s-results.json --frames public/cameras/hanging-tree-1/frames
py tools/yolo-inference/validate_results.py --results app/data/thunder-valley-yolo11s-results.json --frames public/frames
```

The application uses two production secrets, configured in Sites rather than in
Git:

- `OPENWEATHER_API_KEY` for synchronized current weather and wind
- `GROQ_API_KEY` for live Ops chat responses

When a service is unavailable, the application uses an explicit prepared or safe
fallback and never commits a secret locally.

## Project structure

- `app/`: multi-camera UI, map, weather API, chat API, and isolated camera state
- `app/data/`: camera mapping and verified YOLO11s result JSON
- `public/`: camera references and frame sequences
- `tests/`: behavior, built-runtime, geometry, weather/risk, and YOLO artifact tests
- `tools/mock-fire-remotion/`: deterministic Hanging Tree simulated-fire frame generator
- `tools/yolo-inference/`: reproducible inference generator and pinned dependency

## Detector provenance

The original PyroFinder project selected an Ultralytics YOLO11s checkpoint that
was fine-tuned on D-Fire. In the original repository the tracked checkpoint is
stored once at `models/yolo11s_dfire_best.pt`.

Both displayed camera sequences have been processed with that exact checkpoint.
The committed JSON contains authentic model boxes and confidence values together
with the checkpoint SHA-256, frame hashes, class mapping, inference settings, and
runtime versions. The browser filters those verified outputs at the selected
smoke/fire thresholds; it never fabricates a detection or reruns inference when a
slider changes.

Sites cannot execute the PyTorch `.pt` checkpoint in its browser or edge runtime,
so the verified inference step is deliberately reproducible and offline. See
`tools/yolo-inference/README.md` for the exact regeneration commands. The weight
file is not duplicated into the public web bundle.

## Embedded location in the original repository

The canonical shared copy lives at `sites/pyrofinder-hanging-tree-live/`. Run all
Node commands from that directory. Its dependencies, build output, hosting
metadata, and deployment remain isolated from the original Streamlit application
at the repository root. The former `sites/hanging-tree-multicamera-v5/` copy is
archive-only.

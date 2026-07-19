# Verified YOLO11s result generation

This tool runs the exact PyroFinder checkpoint on every Hanging Tree frame and
writes the authentic detections consumed by the Sites UI. The checkpoint remains
in the original repository at `models/yolo11s_dfire_best.pt`; it is not copied
into the public web bundle.

From the original PyroFinder repository root:

```powershell
py -m pip install -r sites/hanging-tree-multicamera-v5/tools/yolo-inference/requirements.txt
py sites/hanging-tree-multicamera-v5/tools/yolo-inference/generate_results.py `
  --weights models/yolo11s_dfire_best.pt `
  --frames sites/hanging-tree-multicamera-v5/public/cameras/hanging-tree-1/frames `
  --output sites/hanging-tree-multicamera-v5/app/data/hanging-tree-yolo11s-results.json
```

Generate the Thunder Valley outputs with the same checkpoint:

```powershell
py sites/hanging-tree-multicamera-v5/tools/yolo-inference/generate_results.py `
  --weights models/yolo11s_dfire_best.pt `
  --frames sites/hanging-tree-multicamera-v5/public/frames `
  --output sites/hanging-tree-multicamera-v5/app/data/thunder-valley-yolo11s-results.json `
  --camera-id TVW --expected-frames 26
```

The generator rejects a checkpoint whose SHA-256 or class mapping differs from
the selected model. It collects candidates at confidence `0.05` with IoU NMS
`0.50`; the web sliders then apply per-class thresholds to those stored model
outputs. Changing a slider does not rerun inference, and `N` remains the temporal
confirmation setting.

Do not hand-edit the generated detection coordinates or confidence values.

Validate committed outputs against their source bytes (add `--weights` when the
checkpoint is locally available to validate its bytes too):

```bash
python tools/yolo-inference/validate_results.py --frames public/cameras/hanging-tree-1/frames --results app/data/hanging-tree-yolo11s-results.json
python tools/yolo-inference/validate_results.py --frames public/frames --results app/data/thunder-valley-yolo11s-results.json
```

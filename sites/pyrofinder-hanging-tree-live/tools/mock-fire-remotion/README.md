# Hanging Tree mock-fire frame generator

This Remotion composition preserves the first ten source frames, introduces a tiny ignition on frame 10, and expands one consistent mature fire corridor through frame 27. The mask radius and opacity change on every fire frame, avoiding static plateaus and directional jumps between unrelated fire lines. The v2 keyframes use brighter flame cores and clearer orange-red edges while the masks keep edits local to the hillside, so the camera overlays, farm, and wider landscape remain unchanged.

Run from this directory:

```powershell
npm install
npm run render
```

The renderer stages the current sequence, restores frames 10 through 27 from
the committed clean source bases, and renders 28 candidate JPEG frames into
`out`. This default command does not change the app's camera frames. After the
candidate sequence has passed visual and YOLO checks, install it explicitly:

```powershell
npm run render:install
```

The install command replaces only
`public/cameras/hanging-tree-1/frames/frame_10.jpg` through `frame_27.jpg`.
Repeated runs remain deterministic instead of compositing over a previous fire
result.

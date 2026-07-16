# Hanging Tree mock-fire frame generator

This Remotion composition preserves the first ten source frames, introduces a tiny ignition on frame 10, and blends three image-edited fire stages through frame 27. The masks keep edits local to the hillside so the camera overlays, farm, and wider landscape remain unchanged.

Run from this directory:

```powershell
npm install
npm run render
```

The renderer stages the current sequence, restores frames 10 through 27 from
the committed clean source bases, renders 28 JPEG frames, and replaces only
`public/cameras/hanging-tree-1/frames/frame_10.jpg` through `frame_27.jpg`.
Repeated runs therefore remain deterministic instead of compositing over a
previous fire result.

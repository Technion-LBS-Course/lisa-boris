"""Pre-compute the Live Ops detection cache (``data/live_demo/cache/detections.json``).

Runs the fine-tuned YOLO detector once per demo frame at the DEFAULT per-class
confidence thresholds and writes a small detection manifest. The Live Ops tab loads
this manifest in default mode, so the demo plays without re-running YOLO — no
flicker, instant first play, and valid on a fresh clone.

Re-run this whenever the demo frames or the default thresholds change. (The app
also rebuilds automatically when it detects a fingerprint mismatch; this script is
for producing/committing the cache offline.)

Usage:
    python scripts/build_live_ops_cache.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Allow running as a plain script (add repo root to sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import inference, live_ops_cache as lc, live_ops_config as lo  # noqa: E402


def main() -> int:
    settings = lo.load_settings()
    items, source = lo.demo_sequence_items(settings)
    if not items:
        print("No demo frames found; nothing to cache.")
        return 1
    frames = lc.build_sequence_frames(items)

    legacy = float(settings.get("confidence_threshold", 0.50))
    thresholds = {
        "smoke": round(float(settings.get("smoke_confidence_threshold", legacy)), 4),
        "fire": round(float(settings.get("fire_confidence_threshold", legacy)), 4),
    }

    model_name = (
        "YOLO11s" if inference.checkpoint_exists("YOLO11s")
        else "YOLO11n" if inference.checkpoint_exists("YOLO11n") else None
    )
    if model_name is None:
        print("No fine-tuned checkpoint found (e.g. models/yolo11s_dfire_best.pt). Aborting.")
        return 1

    from PIL import Image

    model = inference.load_detector(model_name)

    def detect(frame_bytes: bytes) -> dict:
        return inference.run_detection(
            model, Image.open(io.BytesIO(frame_bytes)).convert("RGB"),
            conf=min(thresholds.values()), conf_by_class=thresholds)

    per_frame = lc.build(frames, detect)
    fingerprint = lc.build_fingerprint(
        lc.frames_fingerprint(items), thresholds["smoke"], thresholds["fire"],
        model_name, len(frames))
    lc.save_manifest(fingerprint, per_frame)

    hits = sum(1 for f in per_frame if f["fire_count"] or f["smoke_count"])
    print(
        f"Cached {len(per_frame)} frames ({source}) with {model_name} at "
        f"smoke≥{thresholds['smoke']:.2f} / fire≥{thresholds['fire']:.2f}; "
        f"{hits} frame(s) with a detection.\n-> {lc.MANIFEST_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate reproducible Hanging Tree detections with the real YOLO11s checkpoint.

The public Sites runtime cannot execute a PyTorch ``.pt`` checkpoint. This tool
runs the exact checkpoint offline, records its checksum and inference settings,
and writes the authentic per-frame outputs consumed by the web UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_CHECKPOINT_SHA256 = "6AA0C7DCD60E3572F85F02EDC05293266822F9394944479337BEBB8D178B6903"
EXPECTED_CLASSES = {0: "smoke", 1: "fire"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("public/cameras/hanging-tree-1/frames"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("app/data/hanging-tree-yolo11s-results.json"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-frames", type=int, default=28)
    parser.add_argument("--camera-id", default="HT1")
    return parser.parse_args()


def normalized_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def existing_sources(output: Path) -> dict[str, str]:
    if not output.exists():
        return {}
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        frame["name"]: frame.get("source", frame["name"])
        for frame in payload.get("frames", [])
        if isinstance(frame, dict) and isinstance(frame.get("name"), str)
    }


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    frames_dir = args.frames.resolve()
    output = args.output.resolve()

    if not weights.is_file():
        raise SystemExit(f"Checkpoint not found: {weights}")
    checkpoint_sha256 = sha256_file(weights)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit(
            "Checkpoint checksum mismatch. "
            f"Expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha256}."
        )

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_paths) != args.expected_frames:
        raise SystemExit(
            f"Expected {args.expected_frames} frames in {frames_dir}, found {len(frame_paths)}."
        )

    random.seed(0)
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    np.random.seed(0)
    torch.manual_seed(0)

    model = YOLO(str(weights))
    class_names = normalized_names(model.names)
    if class_names != EXPECTED_CLASSES:
        raise SystemExit(
            f"Unexpected checkpoint classes {class_names}; expected {EXPECTED_CLASSES}."
        )

    results = model.predict(
        source=[str(path) for path in frame_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
        save=False,
    )

    source_names = existing_sources(output)
    frame_set_digest = hashlib.sha256()
    output_frames: list[dict[str, Any]] = []

    for frame_path, result in zip(frame_paths, results, strict=True):
        frame_sha256 = sha256_file(frame_path)
        frame_set_digest.update(f"{frame_path.name}\0{frame_sha256}\n".encode())
        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is not None:
            for box, confidence, class_id in zip(
                boxes.xywhn.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.cls.cpu().tolist(),
                strict=True,
            ):
                detections.append(
                    {
                        "class": result.names[int(class_id)],
                        "confidence": round(float(confidence), 6),
                        "bbox_norm": [round(float(value), 6) for value in box],
                    }
                )
        detections.sort(key=lambda detection: detection["confidence"], reverse=True)
        output_frames.append(
            {
                "name": frame_path.name,
                "source": source_names.get(frame_path.name, frame_path.name),
                "sha256": frame_sha256,
                "inference_ms": round(float(result.speed.get("inference", 0.0)), 3),
                "detections": detections,
            }
        )

    payload = {
        "fingerprint": {
            "version": 2,
            "camera_id": args.camera_id,
            "model": "YOLO11s",
            "output_kind": "verified-ultralytics-inference",
            "checkpoint": "models/yolo11s_dfire_best.pt",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": weights.stat().st_size,
            "classes": {str(key): value for key, value in class_names.items()},
            "imgsz": args.imgsz,
            "collection_confidence": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "device": args.device,
            "python_version": platform.python_version(),
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "n_frames": len(output_frames),
            "frame_set_sha256": frame_set_digest.hexdigest().upper(),
            "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        },
        "frames": output_frames,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    smoke = sum(detection["class"] == "smoke" for frame in output_frames for detection in frame["detections"])
    fire = sum(detection["class"] == "fire" for frame in output_frames for detection in frame["detections"])
    print(f"Wrote {output}")
    print(f"Checkpoint SHA-256: {checkpoint_sha256}")
    print(f"Frames: {len(output_frames)}; smoke candidates: {smoke}; fire candidates: {fire}")


if __name__ == "__main__":
    main()

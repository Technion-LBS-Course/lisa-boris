"""Validate checked-in YOLO artifacts against the exact source-frame bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from generate_results import EXPECTED_CHECKPOINT_SHA256, EXPECTED_CLASSES, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    fingerprint = payload["fingerprint"]
    if fingerprint["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("Result checkpoint checksum is not the approved YOLO11s checkpoint")
    if {int(key): value for key, value in fingerprint["classes"].items()} != EXPECTED_CLASSES:
        raise SystemExit("Result class mapping must be exactly {0: smoke, 1: fire}")
    if fingerprint["output_kind"] != "verified-ultralytics-inference":
        raise SystemExit("Results do not claim authentic Ultralytics inference")
    if args.weights and sha256_file(args.weights) != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("Supplied checkpoint checksum mismatch")

    frame_paths = sorted(args.frames.glob("frame_*.jpg"))
    frames = payload["frames"]
    if len(frames) != len(frame_paths) or fingerprint["n_frames"] != len(frame_paths):
        raise SystemExit("Source-frame/result count mismatch")

    frame_set = hashlib.sha256()
    counts = {"smoke": 0, "fire": 0}
    for index, (source, result) in enumerate(zip(frame_paths, frames, strict=True)):
        expected_name = f"frame_{index:02d}.jpg"
        digest = sha256_file(source)
        if source.name != expected_name or result["name"] != expected_name or result["sha256"] != digest:
            raise SystemExit(f"Frame correspondence/hash mismatch at index {index}")
        frame_set.update(f"{expected_name}\0{digest}\n".encode())
        for detection in result["detections"]:
            kind, confidence, box = detection["class"], detection["confidence"], detection["bbox_norm"]
            if kind not in counts or not math.isfinite(confidence) or not fingerprint["collection_confidence"] <= confidence <= 1:
                raise SystemExit(f"Invalid class/confidence in {expected_name}")
            if len(box) != 4 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in box) or box[2] <= 0 or box[3] <= 0:
                raise SystemExit(f"Invalid normalized xywh in {expected_name}")
            counts[kind] += 1
    if frame_set.hexdigest().upper() != fingerprint["frame_set_sha256"]:
        raise SystemExit("Frame-set checksum mismatch")
    print(f"Validated {len(frames)} frames: smoke={counts['smoke']}, fire={counts['fire']}")
    print(f"Checkpoint SHA-256: {EXPECTED_CHECKPOINT_SHA256}")


if __name__ == "__main__":
    main()

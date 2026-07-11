"""Pre-computed detection cache for the Live Ops demo.

Running YOLO on every frame during autoplay is slow and makes the preview flicker.
This module persists a small per-frame detection manifest computed once at the
DEFAULT confidence thresholds, so the Live tab can play the demo by loading cached
detections and redrawing the boxes (instant, flicker-free) instead of re-running
the detector. On a fresh clone the committed manifest is already valid, so the demo
is fast with no first-run YOLO pass.

The manifest is fingerprinted by (frame content + default smoke/fire thresholds +
model). If the demo frames change or the default thresholds change, the fingerprint
no longer matches and the cache is rebuilt automatically.

Import-safe: only ``hashlib`` / ``json`` at module level. ``PIL`` is imported
lazily inside :func:`annotate` / :func:`build_sequence_frames`. NO ML imports —
detection is injected as a callable so this module never imports ultralytics/torch.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

# Repo root: src/live_ops_cache.py -> repo root (two levels up).
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "live_demo" / "cache"
MANIFEST_PATH = CACHE_DIR / "detections.json"

# Bump when the manifest schema changes so stale caches rebuild.
CACHE_VERSION = 1

# Box colours for the redrawn overlay (fire = red, smoke = blue).
_CLASS_COLORS = {"fire": (255, 70, 50), "smoke": (90, 160, 255)}


# ── Frame resize (shared with the dashboard so cached + live frames match) ─────


def build_sequence_frames(items: list[tuple[str, bytes]]) -> list[dict]:
    """Decode ``(name, raw-bytes)`` pairs and resize all to one common size.

    A zone is drawn once and reused across the whole sequence, so every frame is
    resized to the first frame's size and re-encoded as JPEG. Returns
    ``[{name, bytes, size}]`` in order. Skips unreadable files.
    """
    from PIL import Image  # lazy

    decoded = []
    for name, raw in items:
        try:
            decoded.append((name, Image.open(io.BytesIO(raw)).convert("RGB")))
        except Exception:
            continue
    if not decoded:
        return []
    target = decoded[0][1].size
    frames = []
    for name, img in decoded:
        if img.size != target:
            img = img.resize(target)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        frames.append({"name": name, "bytes": buf.getvalue(), "size": target})
    return frames


# ── Fingerprint ────────────────────────────────────────────────────────────────


def frames_fingerprint(raw_items: list[tuple[str, bytes]]) -> str:
    """Content hash of the raw source frames (order-independent by name)."""
    h = hashlib.sha1()
    for name, raw in sorted(raw_items, key=lambda t: t[0]):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha1(raw).digest())
    return h.hexdigest()


def build_fingerprint(
    frames_hash: str, smoke_thr: float, fire_thr: float, model_name: str, n_frames: int
) -> dict:
    """Fingerprint identifying the cache: frames + default thresholds + model."""
    return {
        "version": CACHE_VERSION,
        "smoke": round(float(smoke_thr), 4),
        "fire": round(float(fire_thr), 4),
        "model": model_name,
        "n_frames": int(n_frames),
        "frames_sha1": frames_hash,
    }


# ── Manifest load / save / validate ────────────────────────────────────────────


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict | None:
    """Return the cached manifest dict, or ``None`` if absent/unreadable/malformed."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_valid(manifest: dict | None, fingerprint: dict) -> bool:
    """True when the manifest matches the expected fingerprint and frame count."""
    if not manifest:
        return False
    frames = manifest.get("frames")
    return (
        manifest.get("fingerprint") == fingerprint
        and isinstance(frames, list)
        and len(frames) == fingerprint.get("n_frames")
    )


def save_manifest(
    fingerprint: dict, per_frame: list[dict], path: Path | str = MANIFEST_PATH
) -> None:
    """Write the manifest (fingerprint + per-frame detection summaries) to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"fingerprint": fingerprint, "frames": per_frame}, indent=2),
        encoding="utf-8",
    )


# ── Build (detection injected — no ML import here) ─────────────────────────────


def build(frames: list[dict], detect_fn) -> list[dict]:
    """Run ``detect_fn`` on each frame's bytes, returning per-frame summaries.

    ``detect_fn(frame_bytes) -> run_detection-shaped dict``. Only the small,
    JSON-serialisable fields are kept (no images — the overlay is redrawn on load).
    """
    per_frame: list[dict] = []
    for fr in frames:
        result = detect_fn(fr["bytes"]) or {}
        per_frame.append({
            "name": fr.get("name", ""),
            "detections": result.get("detections", []),
            "smoke_count": int(result.get("smoke_count", 0)),
            "fire_count": int(result.get("fire_count", 0)),
            "max_confidence": result.get("max_confidence"),
        })
    return per_frame


def result_from_summary(summary: dict, frame_bytes: bytes, model_name: str) -> dict:
    """Rebuild a ``run_detection``-shaped result from a cached summary + the frame.

    Boxes are redrawn on the frame from the cached normalized detections, so no
    detector runs. ``inference_ms`` is 0 and ``from_cache`` marks the source.
    """
    detections = summary.get("detections", [])
    fire = int(summary.get("fire_count", 0))
    smoke = int(summary.get("smoke_count", 0))
    return {
        "annotated_png": annotate(frame_bytes, detections),
        "fire_count": fire,
        "smoke_count": smoke,
        "total_detections": fire + smoke,
        "max_confidence": summary.get("max_confidence"),
        "inference_ms": 0.0,
        "detections": detections,
        "model_name": model_name,
        "from_cache": True,
    }


# ── Overlay (redraw cached boxes; PIL only, no ML) ─────────────────────────────


def annotate(frame_bytes: bytes, detections: list[dict]) -> bytes:
    """Draw fire/smoke boxes + labels on a frame from cached detections (PNG bytes)."""
    from PIL import Image, ImageDraw  # lazy

    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    for d in detections:
        box = d.get("bbox_norm")
        if not box or len(box) != 4:
            continue
        cls = str(d.get("class", "")).lower()
        conf = float(d.get("confidence", 0.0))
        xc, yc, bw, bh = box
        x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
        x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
        color = _CLASS_COLORS.get(cls, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{cls} {conf * 100:.0f}%"
        ty = max(0.0, y1 - 16)
        try:
            tb = draw.textbbox((x1, ty), label)
            draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color + (220,))
        except AttributeError:
            pass
        draw.text((x1, ty), label, fill=(10, 12, 18))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

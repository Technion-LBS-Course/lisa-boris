"""Unit tests for src/live_ops_cache — pre-computed Live Ops detection cache.

Pure/IO logic: no ML, no network. Detection is injected as a fake callable, so no
YOLO/ultralytics import. PIL-dependent helpers skip cleanly if Pillow is absent.
"""

import io

import pytest

from src import live_ops_cache as lc


def _png_bytes(size=(8, 8), color=(10, 20, 30)) -> bytes:
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


# ── fingerprint ────────────────────────────────────────────────────────────────


def test_frames_fingerprint_stable_and_order_independent():
    a = [("f1.jpg", b"aaa"), ("f2.jpg", b"bbb")]
    b = [("f2.jpg", b"bbb"), ("f1.jpg", b"aaa")]  # different order, same content
    assert lc.frames_fingerprint(a) == lc.frames_fingerprint(b)


def test_frames_fingerprint_changes_on_content_or_name():
    base = [("f1.jpg", b"aaa")]
    assert lc.frames_fingerprint(base) != lc.frames_fingerprint([("f1.jpg", b"aab")])
    assert lc.frames_fingerprint(base) != lc.frames_fingerprint([("f9.jpg", b"aaa")])


def test_build_fingerprint_fields():
    fp = lc.build_fingerprint("hash123", 0.4, 0.4, "YOLO11s", 12)
    assert fp["smoke"] == 0.4 and fp["fire"] == 0.4
    assert fp["model"] == "YOLO11s" and fp["n_frames"] == 12
    assert fp["frames_sha1"] == "hash123" and fp["version"] == lc.CACHE_VERSION


# ── manifest load / save / validate ───────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "detections.json"
    fp = lc.build_fingerprint("h", 0.4, 0.4, "YOLO11s", 1)
    per_frame = [{"name": "f1.jpg", "detections": [], "smoke_count": 0, "fire_count": 0,
                  "max_confidence": None}]
    lc.save_manifest(fp, per_frame, path=path)
    loaded = lc.load_manifest(path=path)
    assert loaded["fingerprint"] == fp
    assert loaded["frames"] == per_frame


def test_load_manifest_missing_or_malformed(tmp_path):
    assert lc.load_manifest(path=tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert lc.load_manifest(path=bad) is None


def test_is_valid_matches_fingerprint_and_count():
    fp = lc.build_fingerprint("h", 0.4, 0.4, "YOLO11s", 2)
    good = {"fingerprint": fp, "frames": [{}, {}]}
    assert lc.is_valid(good, fp) is True
    # fingerprint mismatch (different threshold)
    assert lc.is_valid(good, lc.build_fingerprint("h", 0.5, 0.4, "YOLO11s", 2)) is False
    # frame-count mismatch
    assert lc.is_valid({"fingerprint": fp, "frames": [{}]}, fp) is False
    assert lc.is_valid(None, fp) is False


# ── build (detection injected) ─────────────────────────────────────────────────


def test_build_runs_detect_per_frame_and_keeps_small_fields():
    frames = [{"name": "a.jpg", "bytes": b"a"}, {"name": "b.jpg", "bytes": b"b"}]
    calls = []

    def fake_detect(frame_bytes):
        calls.append(frame_bytes)
        return {
            "detections": [{"class": "fire", "confidence": 0.7, "bbox_norm": [0.5, 0.5, 0.2, 0.2]}],
            "fire_count": 1, "smoke_count": 0, "max_confidence": 0.7,
            "annotated_png": b"IGNORED", "inference_ms": 12.3,  # dropped from the summary
        }

    per_frame = lc.build(frames, fake_detect)
    assert calls == [b"a", b"b"]
    assert per_frame[0]["name"] == "a.jpg"
    assert per_frame[0]["fire_count"] == 1
    assert per_frame[0]["detections"][0]["class"] == "fire"
    assert "annotated_png" not in per_frame[0]  # only small fields persisted


def test_build_handles_detect_fn_returning_none():
    # detect_fn may return None for a frame (e.g. an upstream decode failure); the
    # `or {}` guard must yield an empty, zero-count summary rather than crashing on
    # None.get(...).
    per_frame = lc.build([{"name": "a.jpg", "bytes": b"a"}], lambda _b: None)
    assert per_frame[0]["name"] == "a.jpg"
    assert per_frame[0]["detections"] == []
    assert per_frame[0]["smoke_count"] == 0
    assert per_frame[0]["fire_count"] == 0
    assert per_frame[0]["max_confidence"] is None


# ── annotate / result_from_summary (PIL) ───────────────────────────────────────


def test_annotate_returns_png_bytes():
    Image = pytest.importorskip("PIL.Image")
    frame = _png_bytes(size=(60, 40))
    dets = [{"class": "fire", "confidence": 0.6, "bbox_norm": [0.5, 0.5, 0.4, 0.4]}]
    out = lc.annotate(frame, dets)
    assert isinstance(out, bytes) and out[:8] == b"\x89PNG\r\n\x1a\n"

    # Prove a box is actually drawn: annotating with a detection must change the
    # image relative to annotating with none (a deleted draw loop would make the two
    # pixel-identical), and the exact fire box colour must appear (PNG is lossless).
    drawn_px = list(Image.open(io.BytesIO(out)).convert("RGB").getdata())
    blank_px = list(Image.open(io.BytesIO(lc.annotate(frame, []))).convert("RGB").getdata())
    assert drawn_px != blank_px
    assert lc._CLASS_COLORS["fire"] in drawn_px
    assert lc._CLASS_COLORS["fire"] not in blank_px


def test_annotate_skips_malformed_detections():
    Image = pytest.importorskip("PIL.Image")
    frame = _png_bytes(size=(60, 40))
    dets = [
        {"class": "fire", "confidence": 0.6, "bbox_norm": [0.5, 0.5, 0.4, 0.4]},  # valid
        {"class": "smoke", "confidence": 0.5},                                     # no bbox_norm
        {"class": "smoke", "confidence": 0.5, "bbox_norm": [0.1, 0.1]},            # wrong length
        {"class": "smoke", "confidence": 0.5, "bbox_norm": None},                  # None
    ]
    out = lc.annotate(frame, dets)
    assert out[:8] == b"\x89PNG\r\n\x1a\n"  # malformed entries don't crash the draw loop

    px = set(Image.open(io.BytesIO(out)).convert("RGB").getdata())
    assert lc._CLASS_COLORS["fire"] in px       # the one valid box is drawn ...
    assert lc._CLASS_COLORS["smoke"] not in px  # ... every malformed smoke entry is skipped


def test_result_from_summary_shape():
    pytest.importorskip("PIL")
    summary = {
        "detections": [{"class": "smoke", "confidence": 0.5, "bbox_norm": [0.4, 0.4, 0.2, 0.2]}],
        "fire_count": 0, "smoke_count": 1, "max_confidence": 0.5,
    }
    res = lc.result_from_summary(summary, _png_bytes(), "YOLO11s")
    assert res["from_cache"] is True
    assert res["smoke_count"] == 1 and res["total_detections"] == 1
    assert res["detections"] == summary["detections"]
    assert isinstance(res["annotated_png"], bytes)
    assert res["inference_ms"] == 0.0


# ── build_sequence_frames (PIL) ────────────────────────────────────────────────


def test_build_sequence_frames_resizes_to_common_size():
    pytest.importorskip("PIL")
    items = [("f1.jpg", _png_bytes((20, 10))), ("f2.jpg", _png_bytes((40, 20)))]
    frames = lc.build_sequence_frames(items)
    assert len(frames) == 2
    assert frames[0]["size"] == frames[1]["size"] == (20, 10)  # all match the first
    assert all(isinstance(f["bytes"], bytes) for f in frames)


def test_build_sequence_frames_skips_unreadable():
    pytest.importorskip("PIL")
    frames = lc.build_sequence_frames([("ok.jpg", _png_bytes()), ("bad.jpg", b"not an image")])
    assert len(frames) == 1 and frames[0]["name"] == "ok.jpg"

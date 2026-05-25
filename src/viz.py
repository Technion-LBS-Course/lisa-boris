"""Visualization helpers for PyroFinder — on-the-fly image annotation."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw

_BOX_COLORS: dict[str, tuple[int, int, int]] = {
    "fire": (255, 80, 0),
    "smoke": (120, 120, 200),
}
_CLASS_MAP: dict[int, str] = {0: "smoke", 1: "fire"}


def draw_yolo_boxes(image_path: str | Path, label_path: str | Path | None) -> bytes:
    """Return PNG bytes of an image with YOLO bounding boxes drawn on it.

    Reads the stored JPEG/PNG from image_path and overlays labelled rectangles
    using the D-Fire class map (0=smoke, 1=fire). Safe to call even when
    label_path is None or the label file is missing — returns the plain image.
    """
    with Image.open(image_path).convert("RGB") as img:
        if label_path:
            lp = Path(label_path)
            if lp.exists():
                draw = ImageDraw.Draw(img)
                w, h = img.size
                for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])
                    except ValueError:
                        continue
                    cls_name = _CLASS_MAP.get(cls_id, "unknown")
                    color = _BOX_COLORS.get(cls_name, (255, 255, 0))
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    draw.text((x1 + 2, y1 + 2), cls_name, fill=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

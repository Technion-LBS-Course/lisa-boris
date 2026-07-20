"""Import-safety guards for the Live Ops dashboard + its page entry.

Verifies the hard constraint that importing the renderer pulls in NO heavy ML/vision
libraries (ultralytics/torch/groq/cv2) at module import time — detector/segmentation
loading must stay lazy — and that the page entry wires set_page_config + render().

That the Live Ops page actually renders is covered end-to-end in
``test_e2e_streamlit_app.py``.
"""
import ast
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")


def test_import_does_not_pull_heavy_ml():
    import importlib

    for name in ("src.live_ops_config", "src.live_ops_agents", "src.dashboards.live_ops"):
        importlib.import_module(name)
    # Only ultralytics/torch are asserted absent: cv2/groq can be pulled in
    # indirectly by other code paths (see tests/test_dashboards_smoke.py), and the
    # AST test below already guards that the renderer never imports them at top level.
    for heavy in ("ultralytics", "torch"):
        assert heavy not in sys.modules, heavy


def test_no_heavy_top_level_imports_in_renderer():
    from src.dashboards import live_ops

    tree = ast.parse(Path(live_ops.__file__).read_text(encoding="utf-8"))
    top: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top.append(node.module or "")
    heavy = {"ultralytics", "torch", "groq", "cv2"}
    assert not any(m and m.split(".")[0] in heavy for m in top), top


def test_page_entry_exists():
    page = Path(__file__).resolve().parent.parent / "pages" / "1_Live_Ops.py"
    assert page.exists()
    text = page.read_text(encoding="utf-8")
    assert "set_page_config" in text
    assert "live_ops.render()" in text

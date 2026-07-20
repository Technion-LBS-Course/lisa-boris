"""End-to-end tests: render the REAL PyroFinder Streamlit app.

These use ``streamlit.testing.v1.AppTest`` to execute ``app.py`` and the Live Ops
page through Streamlit's own script runner — not mocks — so they exercise the full
render path (sidebar dispatch, dashboard rendering, data/EDA/plotly/folium, mapping)
and catch render-time regressions that unit and integration tests cannot.

They are the top of the unit -> integration -> e2e pyramid and are the slowest layer
(the heavier dashboards recompute EDA and charts on each run). Marked ``e2e`` so the
fast inner loop can skip them::

    pytest -m "not e2e"     # unit + integration only
    pytest -m e2e           # just these

No model weights are loaded here (detector loading stays lazy behind a button click),
so these render without running YOLO.
"""
import os
import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

# Ensure the repo root is importable so the app script's ``from src...`` imports
# resolve when AppTest execs it in-process (pytest normally provides this already).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_APP = os.path.join(_REPO_ROOT, "app.py")
_LIVE_OPS = os.path.join(_REPO_ROOT, "pages", "1_Live_Ops.py")

pytestmark = pytest.mark.e2e

_DASHBOARD_MODES = [
    "M4 Dashboard",
    "M3 Dashboard",
    "M2 Dashboard",
    "Operations & Learning Dashboard",
    "Central Control Dashboard",
]


def _mode_selectbox(at):
    """Return the sidebar 'Dashboard mode' selectbox.

    Some dashboards render their own selectboxes in the main area (e.g. M4's
    'Indoor / Outdoor'), so select the shell's picker by its label rather than index.
    """
    return next(sb for sb in at.selectbox if sb.label == "Dashboard mode")


def _run_classic_shell():
    """Run app.py as the classic multi-dashboard shell.

    New sessions are redirected to the Live Ops page via ``st.switch_page``; setting
    the one-shot guard flag first keeps us on the classic shell so the dashboard
    picker renders.
    """
    at = AppTest.from_file(_APP, default_timeout=180)
    at.session_state["lo_landing_done"] = True
    at.run()
    return at


def test_app_boots_classic_shell_with_dashboard_picker():
    at = _run_classic_shell()
    assert not at.exception
    # The shell rendered its dashboard picker with exactly the five known modes.
    assert _mode_selectbox(at).options == _DASHBOARD_MODES


@pytest.mark.parametrize("mode", _DASHBOARD_MODES)
def test_each_dashboard_mode_renders_without_exception(mode):
    at = _run_classic_shell()
    _mode_selectbox(at).set_value(mode).run()
    assert not at.exception, f"{mode} raised: {at.exception}"
    if mode == "M3 Dashboard":
        # The "upload an image to test" feature lives in M3 -> Demo tab; with the
        # committed checkpoints present it renders its 'Run demo' trigger button.
        import src.inference as inference

        if inference.available_detectors():
            assert "Run demo" in [b.label for b in at.button]


def test_live_ops_page_renders_without_exception():
    at = AppTest.from_file(_LIVE_OPS, default_timeout=180).run()
    assert not at.exception
    # The page produced UI (it is the default landing surface, not a blank page).
    assert at.title or at.header or at.markdown


def test_new_session_redirects_to_live_ops_landing():
    # A fresh session (guard flag unset) marks the landing done and calls
    # st.switch_page to the Live Ops default landing surface — without error.
    at = AppTest.from_file(_APP, default_timeout=180).run()
    assert not at.exception
    assert at.session_state["lo_landing_done"] is True

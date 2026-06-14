"""Dashboard renderers for the PyroFinder Streamlit app.

Each module exposes a ``render(...)`` function called by ``app.py``. Splitting the
dashboards out of ``app.py`` keeps the Streamlit entry point a thin shell. Heavy
ML libraries are never imported at module import time — detector loading stays
lazy inside ``src.inference`` and inside the inference-demo code paths.
"""

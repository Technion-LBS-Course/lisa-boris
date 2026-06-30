"""Groq LLM client helper for operational text (alert summaries, EDA explanations).

Not used for detection — YOLO11s/YOLO11n remain the detectors. The Groq key is
read from st.secrets (Streamlit Cloud / local .streamlit/secrets.toml) with an
environment-variable fallback. The key is never hard-coded.
"""

from __future__ import annotations

import os

# Use the OS certificate store (Windows / Streamlit Cloud) so TLS verification
# works behind networks that intercept HTTPS with their own root CA. Best-effort:
# if truststore is unavailable, fall back to the default certifi bundle.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

import streamlit as st
from groq import Groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _get_api_key() -> str:
    """Return the Groq API key from st.secrets, falling back to the environment."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    key = key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Add it to .streamlit/secrets.toml locally "
            "or to Settings -> Secrets on Streamlit Cloud."
        )
    return key


@st.cache_resource
def get_client() -> Groq:
    """One cached Groq client reused across Streamlit reruns."""
    return Groq(api_key=_get_api_key())


def ask(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a single prompt to Groq and return the reply text."""
    resp = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

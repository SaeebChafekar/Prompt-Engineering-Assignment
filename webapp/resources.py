"""Shared setup for the pages: configuration, model backends, and one client per browser session.

The three module-level hooks below are what the pages call to get configuration and
backends. Tests replace them with fakes, so the whole interface can be exercised without
Ollama, a GPU or an internet connection.
"""

from __future__ import annotations

import streamlit as st

from common.backends import ChatBackend, create_chat_client, create_embedder
from common.config import Config
from common.errors import ChatbotError

config_loader = Config.from_env
chat_factory = create_chat_client
embedder_factory = create_embedder


def load_config() -> Config | None:
    """The current configuration, or None after showing a clear error on the page."""
    try:
        return config_loader()
    except ChatbotError as exc:
        st.error(f"Configuration problem: {exc}")
        return None


def backend_label(cfg: Config) -> str:
    if cfg.backend == "gemini":
        return f"Gemini (cloud): {cfg.gemini_model}"
    return f"Ollama (local): {cfg.chat_model}"


def get_chat(cfg: Config) -> ChatBackend:
    """The chat client for this browser session; recreated if the backend or model setting changes."""
    key = backend_label(cfg)
    if st.session_state.get("_chat_key") != key:
        old = st.session_state.pop("_chat_client", None)
        if old is not None:
            old.close()
        st.session_state["_chat_client"] = chat_factory(cfg)
        st.session_state["_chat_key"] = key
    return st.session_state["_chat_client"]

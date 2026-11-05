"""The sidebar: page choice, which models are in use, and a connection check."""

from __future__ import annotations

import streamlit as st

from common.errors import ChatbotError

from . import resources

PAGE_REVIEW = "Project 1: Code reviewer"
PAGE_REPO = "Project 2: Codebase Q&A"


def render() -> str:
    """Draw the sidebar and return the chosen page."""
    with st.sidebar:
        st.title("Code chatbots")
        page = st.radio("Choose a chatbot", [PAGE_REVIEW, PAGE_REPO], key="page")
        cfg = resources.load_config()
        if cfg is None:
            return page
        st.divider()
        with st.container(border=True):
            st.caption("Models in use")
            st.write(f"**Chat:** {resources.backend_label(cfg)}")
            st.write(f"**Embeddings:** {cfg.embed_model} (local)")
            if st.button("Check connection", key="check_backend", use_container_width=True):
                _check_connection(cfg)
        st.caption(
            "💡 **Tip:** Switch to the cloud model by starting the app with `CHATBOT_BACKEND=gemini` "
            "and `GEMINI_API_KEY` set. Everything else stays the same."
        )
    return page


def _check_connection(cfg) -> None:
    with st.spinner("Checking the models ..."):
        try:
            chat = resources.get_chat(cfg)
            chat.health_check()
            st.success(f"Chat model ready: {chat.model_name}")
        except ChatbotError as exc:
            st.error(f"Chat model problem ({type(exc).__name__}): {exc}")
        try:
            embedder = resources.embedder_factory(cfg)
            try:
                embedder.health_check()
                st.success(f"Embedding model ready: {embedder.model_name}")
            finally:
                embedder.close()
        except ChatbotError as exc:
            st.error(f"Embedding model problem ({type(exc).__name__}): {exc}")

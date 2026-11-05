"""The web interface for both chatbots.

    streamlit run app.py

Project 1 (code reviewer) and Project 2 (codebase Q&A) live on two pages chosen in the sidebar.
Models are configured with environment variables (see README.md); the default is local Ollama.
"""

import streamlit as st

from webapp import repo_page, review_page, sidebar
from webapp.theme import inject_custom_theme

st.set_page_config(page_title="Code chatbots", page_icon="🧭", layout="wide")
inject_custom_theme()

page = sidebar.render()
if page == sidebar.PAGE_REVIEW:
    review_page.render()
else:
    repo_page.render()

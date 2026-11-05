"""Design system and custom CSS for a clean, simplistic, user-friendly interface."""

from __future__ import annotations

import streamlit as st

CUSTOM_CSS = """
<style>
/* ---------------- Clean typography ---------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

code, pre, [class*="stCode"] {
    font-family: 'JetBrains Mono', SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}

/* ---------------- Severity Badges ---------------- */
.sev-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.sev-critical {
    background: rgba(239, 68, 68, 0.12);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.sev-high {
    background: rgba(249, 115, 22, 0.12);
    color: #f97316;
    border: 1px solid rgba(249, 115, 22, 0.3);
}

.sev-medium {
    background: rgba(234, 179, 8, 0.14);
    color: #ca8a04;
    border: 1px solid rgba(234, 179, 8, 0.35);
}

.sev-low {
    background: rgba(14, 165, 233, 0.12);
    color: #0284c7;
    border: 1px solid rgba(14, 165, 233, 0.3);
}

.sev-style {
    background: rgba(139, 92, 246, 0.12);
    color: #8b5cf6;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

/* ---------------- Citations & Sources ---------------- */
.citation-chip {
    display: inline-flex;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(16, 185, 129, 0.1);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.3);
    margin: 2px 4px 2px 0;
}

.citation-chip-invalid {
    background: rgba(239, 68, 68, 0.1);
    color: #dc2626;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.conf-pill {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 4px;
}
.conf-certain { background: rgba(16, 185, 129, 0.15); color: #059669; }
.conf-likely { background: rgba(14, 165, 233, 0.15); color: #0284c7; }
.conf-guess { background: rgba(245, 158, 11, 0.15); color: #d97706; }

/* ---------------- Metric styling ---------------- */
div[data-testid="stMetric"] {
    background: var(--background-secondary, rgba(255, 255, 255, 0.03));
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 8px;
    padding: 10px 14px;
}
</style>
"""


def inject_custom_theme() -> None:
    """Inject custom modern CSS into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def severity_pill(severity: str) -> str:
    """HTML pill for issue severity."""
    sev = severity.lower()
    icons = {"critical": "🔥", "high": "⚠️", "medium": "🟡", "low": "ℹ️", "style": "🎨"}
    icon = icons.get(sev, "•")
    return f'<span class="sev-badge sev-{sev}">{icon} {severity.upper()}</span>'


def confidence_pill(confidence: str) -> str:
    """HTML pill for graph confidence."""
    conf = confidence.lower()
    return f'<span class="conf-pill conf-{conf}">{confidence}</span>'

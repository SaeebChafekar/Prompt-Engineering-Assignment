"""Project 1 page: paste code, get a structured review, then chat about it."""

from __future__ import annotations

import streamlit as st

from code_review import CodeReviewer, SEVERITIES
from code_review.render import SEVERITY_LABELS, issue_markdown, review_markdown
from common.errors import ChatbotError

from . import resources
from .theme import severity_pill

LANGUAGES = ["auto", "python", "javascript", "typescript", "java", "c", "cpp", "go", "rust",
             "csharp", "ruby", "php", "shell", "sql", "kotlin"]
UPLOAD_TYPES = ["py", "js", "jsx", "ts", "tsx", "java", "c", "h", "cpp", "go", "rs", "cs", "rb", "php", "sh", "sql", "kt", "txt"]
EXAMPLE_CODE = '''def average(numbers):
    total = 0
    for i in range(len(numbers) + 1):
        total += numbers[i]
    return total / len(numbers)


def find_user(db, name):
    query = "SELECT * FROM users WHERE name = '" + name + "'"
    return db.execute(query)
'''


def _get_effort(raw: str | None) -> str:
    if not raw:
        return "high"
    lowered = raw.lower()
    if "high" in lowered:
        return "high"
    if "low" in lowered:
        return "low"
    return "medium"


def _load_example() -> None:
    st.session_state["review_code"] = EXAMPLE_CODE
    st.session_state["review_problem"] = "average([]) crashes"


def render() -> None:
    st.header("Project 1: Code reviewer")
    st.caption(
        "A prompt-based chatbot. Paste code and get issues grouped by severity, each with an actionable minimal fix. "
        "The reply is forced into a strict JSON format and validated before it is shown."
    )
    cfg = resources.load_config()
    if cfg is None:
        return

    with st.container(border=True):
        st.markdown("**Review Setup & Code Input**")
        top_a, top_b = st.columns([1, 1])
        with top_a:
            st.selectbox(
                "Analysis effort",
                ["High (Deep analysis & root cause)", "Medium (Balanced)", "Low (Fast & concise)"],
                index=0,
                key="review_effort",
                help="High effort prompts the model for comprehensive root cause analysis, edge cases, failure scenarios, and complete fixes.",
            )
        with top_b:
            active_effort = _get_effort(st.session_state.get("review_effort"))
            st.write("")
            st.caption(f"**Depth mode:** {active_effort.capitalize()} effort active")

        left, right = st.columns([3, 2])
        with left:
            st.text_area("Your code", key="review_code", height=320, placeholder="Paste the code you want reviewed ...")
            st.file_uploader("... or upload a file", type=UPLOAD_TYPES, key="review_file")
        with right:
            st.selectbox("Language", LANGUAGES, key="review_language")
            st.text_input("What is going wrong? (optional)", key="review_problem",
                          placeholder="an error message, or what you expected to happen")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.button("Load an example", key="review_example", on_click=_load_example, use_container_width=True)
            with btn_col2:
                run = st.button("Review code", type="primary", key="review_btn", use_container_width=True)

    if run:
        _run_review(cfg)

    result = st.session_state.get("review_result")
    if result is not None:
        _show_result(result)
        _followup_chat(cfg, result)


def _run_review(cfg) -> None:
    uploaded = st.session_state.get("review_file")
    code, filename = st.session_state.get("review_code", ""), None
    if uploaded is not None:
        code, filename = uploaded.getvalue().decode("utf-8", errors="replace"), uploaded.name
        st.info(f"Reviewing the uploaded file {filename} instead of the text box.")
    effort = _get_effort(st.session_state.get("review_effort"))
    try:
        with st.spinner(f"Reviewing code with {effort.upper()} effort ... (local models take ~30-60s)"):
            reviewer = CodeReviewer(resources.get_chat(cfg), cfg)
            result = reviewer.review(
                code, language=st.session_state.get("review_language"),
                problem=st.session_state.get("review_problem"), filename=filename, effort=effort,
            )
    except (ChatbotError, ValueError) as exc:
        st.session_state.pop("review_result", None)
        st.error(f"{type(exc).__name__}: {exc}")
        return
    st.session_state["review_result"] = result
    st.session_state["review_chat"] = []  # a new review starts a new conversation


def _show_result(result) -> None:
    review = result.review
    st.subheader("Result")
    with st.container(border=True):
        st.info(f"**Summary:** {review.summary}")
        if review.verdict:
            st.write(f"**Verdict:** {review.verdict}")
        counts = review.counts()
        for column, severity in zip(st.columns(len(SEVERITIES)), SEVERITIES):
            column.metric(SEVERITY_LABELS[severity], counts[severity])
        notes = [f"{result.line_count} lines of {result.language} reviewed."]
        if result.parser_note:
            notes.append(f"Automatic check: {result.parser_note}")
        if result.attempts > 1:
            notes.append("The model needed a second try to produce a valid review.")
        st.caption(" ".join(notes))
        if not review.issues:
            st.success("No issues found.")
        for severity in SEVERITIES:
            group = [i for i in review.issues if i.severity == severity]
            if group:
                with st.expander(f"{SEVERITY_LABELS[severity]} ({len(group)})", expanded=severity in ("critical", "high")):
                    for index, issue in enumerate(group):
                        if index:
                            st.divider()
                        st.markdown(issue_markdown(issue, result.language))
        st.download_button("Download the review (Markdown)", review_markdown(result), file_name="review.md",
                           mime="text/markdown", key="review_download")


def _followup_chat(cfg, result) -> None:
    st.subheader("Ask about this review")
    with st.container(border=True):
        effort = _get_effort(st.session_state.get("review_effort"))
        st.caption(f"Active discussion effort: **{effort.capitalize()}**")
        messages = st.session_state.setdefault("review_chat", [])
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("e.g. why is the first issue critical? show me a fixed version", key="review_chat_input")
        if not question:
            return
        history = [  # failed exchanges are shown on screen but never sent back to the model
            (messages[i]["content"], messages[i + 1]["content"])
            for i in range(0, len(messages) - 1, 2)
            if not messages[i + 1].get("error")
        ]
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                reviewer = CodeReviewer(resources.get_chat(cfg), cfg)
                answer = st.write_stream(reviewer.followup_stream(result, question, history, effort=effort))
            except (ChatbotError, ValueError) as exc:
                message = f"{type(exc).__name__}: {exc}"
                st.error(message)
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": f"**Error.** {message}", "error": True})
                return
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

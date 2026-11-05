"""Turn a review into Markdown (for download and for the web page)."""

from __future__ import annotations

from .reviewer import ReviewResult
from .schema import SEVERITIES

SEVERITY_LABELS = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low", "style": "Style"}


def issue_markdown(issue, language: str) -> str:
    lines = [f"**{issue.lines} · {issue.category} · {issue.title}**", "", issue.problem]
    if issue.why_it_matters:
        lines += ["", f"*Why it matters:* {issue.why_it_matters}"]
    if issue.minimal_fix:
        lines += ["", "*Minimal fix:*", f"```{language if language != 'code' else ''}", issue.minimal_fix, "```"]
    if issue.source == "parser":
        lines += ["", "_Found by the automatic syntax check, not by the model._"]
    return "\n".join(lines)


def review_markdown(result: ReviewResult, title: str = "Code review") -> str:
    review = result.review
    counts = review.counts()
    header = ", ".join(f"{counts[s]} {s}" for s in SEVERITIES if counts[s]) or "no issues"
    out = [f"# {title}", "", f"**Summary:** {review.summary}", "", f"**Verdict:** {review.verdict or '-'}",
           "", f"_{result.line_count} lines of {result.language} reviewed; {header}._"]
    if result.parser_note:
        out += ["", f"_Automatic check: {result.parser_note}_"]
    for severity in SEVERITIES:
        group = [i for i in review.issues if i.severity == severity]
        if group:
            out += ["", f"## {SEVERITY_LABELS[severity]} ({len(group)})"]
            for issue in group:
                out += ["", issue_markdown(issue, result.language)]
    return "\n".join(out) + "\n"

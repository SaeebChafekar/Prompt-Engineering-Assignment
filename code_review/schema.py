"""The review output contract: a JSON schema for the model, and validation of what comes back.

The schema is sent to the model (Ollama and Gemini both constrain output to it), but a model
can still return values that fit the schema yet make no sense (line 900 in a 20-line file,
severity "major"). Review.from_payload therefore validates and normalizes everything: it
repairs what can be repaired safely, clamps line numbers into the file, orders issues by
severity, and raises InvalidOutputError when the reply is unusable so the caller can retry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from common.errors import InvalidOutputError

SEVERITIES = ("critical", "high", "medium", "low", "style")
SEVERITY_RANK = {name: i for i, name in enumerate(SEVERITIES)}
ISSUE_CATEGORIES = ("bug", "security", "performance", "error-handling", "readability", "maintainability", "other")
# Words models use instead of our severity names.
SEVERITY_ALIASES = {
    "blocker": "critical", "severe": "critical", "major": "high", "error": "high", "serious": "high",
    "moderate": "medium", "warning": "medium", "minor": "low", "info": "low", "nit": "style",
    "suggestion": "style", "trivial": "style",
}
MAX_ISSUES = 25
MAX_TEXT = {"title": 120, "problem": 700, "why_it_matters": 500, "minimal_fix": 900, "summary": 500, "verdict": 200}
MAX_TEXT_BY_EFFORT = {
    "low": {"title": 120, "problem": 450, "why_it_matters": 350, "minimal_fix": 600, "summary": 350, "verdict": 150},
    "medium": MAX_TEXT,
    "high": {"title": 160, "problem": 3500, "why_it_matters": 2500, "minimal_fix": 4000, "summary": 1500, "verdict": 350},
}

_ISSUE_PROPERTIES = {
    "severity": {"type": "string", "enum": list(SEVERITIES)},
    "category": {"type": "string", "enum": list(ISSUE_CATEGORIES)},
    "line_start": {"type": "integer"},
    "line_end": {"type": "integer"},
    "title": {"type": "string"},
    "problem": {"type": "string"},
    "why_it_matters": {"type": "string"},
    "minimal_fix": {"type": "string"},
}
REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {"type": "object", "properties": _ISSUE_PROPERTIES, "required": list(_ISSUE_PROPERTIES)},
        },
        "verdict": {"type": "string"},
        "patched_code": {"type": "string"},
    },
    "required": ["summary", "issues", "verdict"],
}


@dataclass(frozen=True)
class Issue:
    severity: str
    category: str
    line_start: int
    line_end: int
    title: str
    problem: str
    why_it_matters: str
    minimal_fix: str
    source: str = "model"  # "parser" for issues found by a deterministic check

    @property
    def lines(self) -> str:
        return f"L{self.line_start}" if self.line_start == self.line_end else f"L{self.line_start}-{self.line_end}"


@dataclass(frozen=True)
class Review:
    summary: str
    issues: tuple[Issue, ...] = ()
    verdict: str = ""
    patched_code: str = ""

    def counts(self) -> dict[str, int]:
        counts = {name: 0 for name in SEVERITIES}
        for issue in self.issues:
            counts[issue.severity] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        data = {"summary": self.summary, "issues": [asdict(i) for i in self.issues], "verdict": self.verdict}
        if self.patched_code:
            data["patched_code"] = self.patched_code
        return data

    def with_issue(self, issue: Issue) -> "Review":
        """A copy with one more issue, kept in severity/line order."""
        return Review(self.summary, _sorted((*self.issues, issue)), self.verdict, self.patched_code)

    @classmethod
    def from_payload(cls, payload: Any, line_count: int, effort: str = "medium") -> "Review":
        """Validate and normalize a model reply. Raises InvalidOutputError if it is unusable."""
        if not isinstance(payload, dict):
            raise InvalidOutputError(f"The review must be a JSON object, got {type(payload).__name__}.")
        raw_issues = payload.get("issues")
        if not isinstance(raw_issues, list):
            raise InvalidOutputError("The review has no 'issues' list.")
        limits = MAX_TEXT_BY_EFFORT.get(effort, MAX_TEXT)
        summary = _text(payload.get("summary"), "summary", limits)
        verdict = _text(payload.get("verdict"), "verdict", limits)
        if not summary:
            raise InvalidOutputError("The review has no summary.")

        issues: list[Issue] = []
        seen: set[tuple[str, int, str]] = set()
        for position, raw in enumerate(raw_issues[:MAX_ISSUES], 1):
            if not isinstance(raw, dict):
                raise InvalidOutputError(f"Issue {position} is not an object.")
            issue = _issue(raw, line_count, limits)
            if issue is None:  # an empty shell with no title and no problem text: drop it
                continue
            key = (issue.severity, issue.line_start, issue.title.lower())
            if key not in seen:
                seen.add(key)
                issues.append(issue)
        raw_patched = payload.get("patched_code")
        patched_code = raw_patched.strip() if isinstance(raw_patched, str) else ""
        return cls(summary, _sorted(issues), verdict, patched_code)


def _text(value: Any, field_name: str, limits: dict[str, int] = MAX_TEXT) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise InvalidOutputError(f"'{field_name}' must be text, got {type(value).__name__}.")
    value = value.strip()
    limit = limits.get(field_name, MAX_TEXT.get(field_name, 500))
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def _line(value: Any, fallback: int, line_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return fallback
    try:
        number = int(float(value))
    except ValueError:
        return fallback
    return min(max(number, 1), max(line_count, 1))


def _issue(raw: dict[str, Any], line_count: int, limits: dict[str, int] = MAX_TEXT) -> Issue | None:
    severity = str(raw.get("severity", "")).strip().lower()
    severity = SEVERITY_ALIASES.get(severity, severity)
    if severity not in SEVERITY_RANK:
        severity = "medium"  # an unknown label is kept visible rather than dropped
    category = str(raw.get("category", "")).strip().lower().replace("_", "-").replace(" ", "-")
    if category not in ISSUE_CATEGORIES:
        category = "other"
    title = _text(raw.get("title"), "title", limits)
    problem = _text(raw.get("problem"), "problem", limits)
    if not title and not problem:
        return None
    start = _line(raw.get("line_start"), 1, line_count)
    end = _line(raw.get("line_end"), start, line_count)
    if end < start:
        start, end = end, start
    return Issue(
        severity=severity, category=category, line_start=start, line_end=end,
        title=title or problem[:80], problem=problem or title,
        why_it_matters=_text(raw.get("why_it_matters"), "why_it_matters", limits),
        minimal_fix=_text(raw.get("minimal_fix"), "minimal_fix", limits),
    )


def _sorted(issues: Any) -> tuple[Issue, ...]:
    return tuple(sorted(issues, key=lambda i: (SEVERITY_RANK[i.severity], i.line_start, i.title)))

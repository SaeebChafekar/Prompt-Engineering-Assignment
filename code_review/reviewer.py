"""The code reviewer: validate input, run a cheap automatic check, ask the model, validate the reply.

    code (+ optional problem description) --> numbered listing --> model (JSON mode)
        --> validated Review (retry once on a bad reply) --> + parser-found issues

For Python, `ast.parse` runs first. A syntax error is a certainty, not an opinion, so it is
added to the review as a critical issue by the program itself (source="parser") and the model
is told about it. Everything else is the model's judgement, clearly separated from that.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Iterator, Sequence

from common.backends import ChatBackend
from common.config import Config
from common.errors import InvalidOutputError

from .prompts import (
    CORRECTION_MESSAGE,
    FOLLOWUP_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_followup_message,
    build_review_message,
    get_followup_system_prompt,
    get_review_system_prompt,
)
from .schema import REVIEW_SCHEMA, Issue, Review

MAX_CODE_CHARS = 20_000
MAX_CODE_LINES = 600
MAX_PROBLEM_CHARS = 1_000
MAX_FOLLOWUP_CHARS = 1_000
LOCAL_MAX_TOKENS = 2_048  # JSON reviews are longer than chat answers; Gemini uses its own larger default

_EXTENSION_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".sh": "shell", ".sql": "sql", ".kt": "kotlin",
}
# (pattern, weight): a clear signal such as `def name(` counts double. A language needs a total of 2+.
_LANGUAGE_HINTS: dict[str, list[tuple[str, int]]] = {
    "python": [(r"^\s*def \w+\(", 2), (r"^\s*class \w+.*:\s*$", 2), (r"^\s*(import \w+|from [\w.]+ import )", 2),
               (r"^\s*if __name__", 2), (r"\bprint\(", 1), (r"^\s*(elif|except|finally)\b.*:", 1), (r"\bself\b", 1),
               (r"^\s*for \w+ in .+:\s*$", 1), (r"^\s*pass\s*$", 1), (r"\b(None|True|False)\b", 1)],
    "javascript": [(r"\bfunction\b", 2), (r"console\.log", 2), (r"=>", 1), (r"\b(const|let|var) \w+", 1),
                   (r"===", 1), (r"require\(", 1)],
    "java": [(r"\bpublic (static )?(class|void)", 2), (r"System\.out", 2), (r"\bString\[\] args", 1)],
    "c": [(r"#include\s*<", 2), (r"\bint main\(", 2), (r"\bprintf\(", 1)],
    "go": [(r"^package \w+", 2), (r"\bfunc \w+\(", 2), (r":= ", 1)],
    "rust": [(r"\bfn \w+\(", 2), (r"\blet mut\b", 2), (r"println!", 2)],
    "shell": [(r"^#!/bin/(ba)?sh", 2), (r"\becho \$", 1), (r"\bfi$", 1)],
    "sql": [(r"\bSELECT\b.*\bFROM\b", 2), (r"\bINSERT INTO\b", 2), (r"\bWHERE\b", 1)],
}


_COMMENT_FIX_PREFIX = re.compile(
    r"^\s*(?:#|//|/\*)\s*(?:fix|fixed|correction|note|solution|changed|remediation)\b",
    re.IGNORECASE,
)
_INLINE_FIX_COMMENT = re.compile(
    r"\s+(?:#|//)\s*(?:fix|fixed|correction|note|solution|changed|remediation)\b.*$",
    re.IGNORECASE,
)


def _clean_fix_text(text: str) -> str:
    """Strip explanatory fix comments added by models (e.g. '# Fix: bounds check')."""
    cleaned = []
    for line in text.splitlines():
        if _COMMENT_FIX_PREFIX.match(line):
            continue
        cleaned.append(_INLINE_FIX_COMMENT.sub("", line))
    return "\n".join(cleaned)


@dataclass(frozen=True)
class ReviewResult:
    review: Review
    code: str  # normalized source that was reviewed
    numbered_code: str
    language: str
    line_count: int
    parser_note: str | None = None
    attempts: int = 1  # 2 means the first model reply was unusable and a retry succeeded
    effort: str = "medium"
    focus: str = "general"
    filename: str | None = None

    def patched_code(self) -> str:
        """Return the source code with all recommended fixes applied (clean code, no added comments)."""
        if self.review.patched_code and self.review.patched_code.strip():
            raw = self.review.patched_code.strip()
            if raw.startswith("```"):
                lines_p = raw.splitlines()
                if len(lines_p) >= 2 and lines_p[-1].strip() == "```":
                    raw = "\n".join(lines_p[1:-1]).strip()
            return _clean_fix_text(raw)
        lines = self.code.splitlines(keepends=True)
        applicable_issues = [
            i for i in self.review.issues
            if i.minimal_fix and 1 <= i.line_start <= len(lines)
        ]
        if not applicable_issues:
            return self.code

        sorted_issues = sorted(applicable_issues, key=lambda i: (-i.line_start, -i.line_end))
        patched_lines = list(lines)
        for issue in sorted_issues:
            start_idx = max(0, issue.line_start - 1)
            end_idx = min(len(patched_lines), issue.line_end)
            fix_content = _clean_fix_text(issue.minimal_fix).strip("\r\n")
            if not fix_content:
                continue
            orig_line = lines[start_idx] if start_idx < len(lines) else ""
            indent = orig_line[:len(orig_line) - len(orig_line.lstrip())]
            if indent and not fix_content.startswith((" ", "\t")):
                fix_content = "\n".join(indent + l if l.strip() else l for l in fix_content.splitlines())
            if not fix_content.endswith("\n"):
                fix_content += "\n"
            patched_lines[start_idx:end_idx] = [fix_content]
        return "".join(patched_lines)

    def unified_diff(self) -> str:
        """Return unified diff comparing original code to patched code."""
        patched = self.patched_code()
        if not patched or patched.strip() == self.code.strip():
            return ""
        import difflib
        fname = self.filename or f"code.{self.language}"
        diff = difflib.unified_diff(
            self.code.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{fname}",
            tofile=f"b/{fname}",
        )
        return "".join(diff)


def detect_language(code: str, filename: str | None = None, hint: str | None = None) -> str:
    """The language to tell the model: an explicit hint, the file extension, or a guess from the text."""
    if hint and hint.strip().lower() not in ("", "auto"):
        return hint.strip().lower()
    if filename:
        found = _EXTENSION_LANGUAGES.get(PurePosixPath(filename.lower()).suffix)
        if found:
            return found
    scores = {
        lang: sum(weight for pattern, weight in patterns if re.search(pattern, code, re.MULTILINE | re.IGNORECASE
                                                                       if lang == "sql" else re.MULTILINE))
        for lang, patterns in _LANGUAGE_HINTS.items()
    }
    best, score = max(scores.items(), key=lambda item: item[1])
    return best if score >= 2 else "code"


def number_lines(code: str) -> str:
    lines = code.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(lines, 1))


def normalize_code(code: str) -> str:
    """Validate pasted code and normalize its line endings. Raises ValueError with a helpful message."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Paste some code to review.")
    if "\x00" in code:
        raise ValueError("That does not look like text (it contains NUL bytes). Paste source code.")
    text = code.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").rstrip("\n")
    if len(text) > MAX_CODE_CHARS:
        raise ValueError(
            f"The code is too long ({len(text):,} characters; the limit is {MAX_CODE_CHARS:,}). "
            "Review one function or file at a time."
        )
    if text.count("\n") + 1 > MAX_CODE_LINES:
        raise ValueError(f"The code has more than {MAX_CODE_LINES} lines. Review one file or function at a time.")
    return text


def python_syntax_issue(code: str) -> tuple[str | None, Issue | None]:
    """Run the Python parser. Returns (note for the model, a ready-made critical issue) or (None, None)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
    except SyntaxError as exc:
        line = max(exc.lineno or 1, 1)
        message = exc.msg or "invalid syntax"
        note = f"Python's parser reports a syntax error at line {line}: {message}."
        issue = Issue(
            severity="critical", category="bug", line_start=line, line_end=line,
            title="Syntax error", problem=f"Python cannot parse this code: {message}.",
            why_it_matters="The code will not run at all until this is fixed.",
            minimal_fix="Fix the syntax on this line (check brackets, colons and indentation).",
            source="parser",
        )
        return note, issue
    except (ValueError, RecursionError, MemoryError):
        return None, None
    return "Python's parser accepted the code (no syntax errors).", None


class CodeReviewer:
    def __init__(self, chat: ChatBackend, cfg: Config) -> None:
        self.chat = chat
        self.cfg = cfg

    def review(
        self,
        code: str,
        *,
        language: str | None = None,
        problem: str | None = None,
        filename: str | None = None,
        effort: str = "medium",
        focus: str = "general",
    ) -> ReviewResult:
        """Review pasted code. Raises ValueError for bad input and ChatbotError subclasses for
        model, network or timeout failures (InvalidOutputError if the model never produces a usable review)."""
        text = normalize_code(code)
        problem = (problem or "").strip() or None
        if problem and len(problem) > MAX_PROBLEM_CHARS:
            raise ValueError(f"The problem description is too long ({len(problem)} characters; the limit is {MAX_PROBLEM_CHARS}).")
        lang = detect_language(text, filename, language)
        line_count = text.count("\n") + 1
        numbered = number_lines(text)

        parser_note, parser_issue = (None, None)
        if lang == "python":
            parser_note, parser_issue = python_syntax_issue(text)

        messages = [{"role": "user", "content": build_review_message(lang, numbered, problem, parser_note)}]
        review, attempts = self._ask_model(messages, line_count, effort=effort, focus=focus)
        if parser_issue is not None and not any(
            i.line_start == parser_issue.line_start and "syntax" in (i.title + i.problem).lower() for i in review.issues
        ):
            review = review.with_issue(parser_issue)
        return ReviewResult(
            review, text, numbered, lang, line_count, parser_note, attempts,
            effort=effort, focus=focus, filename=filename,
        )

    def _ask_model(
        self, messages: list[dict[str, str]], line_count: int, effort: str = "medium", focus: str = "general"
    ) -> tuple[Review, int]:
        system_prompt = get_review_system_prompt(effort, focus)
        if self.chat.backend_name == "ollama":
            max_tokens = 4096 if effort == "high" else LOCAL_MAX_TOKENS
        else:
            max_tokens = 4096 if effort == "high" else None
        last_error: InvalidOutputError | None = None
        for attempt in (1, 2):
            try:
                payload = self.chat.chat_json(messages, REVIEW_SCHEMA, system=system_prompt, max_tokens=max_tokens)
                return Review.from_payload(payload, line_count, effort=effort), attempt
            except InvalidOutputError as exc:  # unusable reply: tell the model what was wrong, once
                last_error = exc
                messages = [*messages[:1], {"role": "user", "content": CORRECTION_MESSAGE.format(error=str(exc)[:300])}]
        raise InvalidOutputError(
            f"The model did not produce a usable review after 2 attempts ({last_error}). "
            "Try again, or use a shorter piece of code."
        )

    # ------------------------------------------------------------ follow-up chat
    def followup_stream(
        self,
        result: ReviewResult,
        question: str,
        history: Sequence[tuple[str, str]] = (),
        effort: str = "medium",
        focus: str = "general",
    ) -> Iterator[str]:
        """Answer a question about a finished review, streaming the reply."""
        text = question.strip()
        if not text:
            raise ValueError("The question is empty.")
        if len(text) > MAX_FOLLOWUP_CHARS:
            raise ValueError(f"The question is too long ({len(text)} characters; the limit is {MAX_FOLLOWUP_CHARS}).")
        message = build_followup_message(
            result.numbered_code, json.dumps(result.review.to_dict(), ensure_ascii=False), list(history), text
        )
        system_prompt = get_followup_system_prompt(effort, focus)
        max_tokens = 4096 if effort == "high" else None
        return self.chat.chat_stream([{"role": "user", "content": message}], system=system_prompt, max_tokens=max_tokens)

    def followup(
        self,
        result: ReviewResult,
        question: str,
        history: Sequence[tuple[str, str]] = (),
        on_token: Callable[[str], None] | None = None,
        effort: str = "medium",
        focus: str = "general",
    ) -> str:
        pieces: list[str] = []
        for piece in self.followup_stream(result, question, history, effort=effort, focus=focus):
            pieces.append(piece)
            if on_token:
                on_token(piece)
        answer = "".join(pieces).strip()
        if not answer:
            raise InvalidOutputError("The model returned an empty answer. Try asking again.")
        return answer
        return answer

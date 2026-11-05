"""The evaluation question set.

Each question lists its GOLD evidence: the code that answers it, as "path::symbol" (a function,
method or class) or just "path". Questions with answerable=false ask about something that does
not exist in the repository; the right reply is that the code does not contain it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from repo_qa.index.index import RepoIndex

CATEGORIES = ("locate", "explain", "relationship", "unanswerable")


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    gold: tuple[str, ...]
    must_include: tuple[str, ...]
    answerable: bool = True


def parse_gold(entry: str) -> tuple[str, str | None]:
    """'app/models.py::User.check' -> ('app/models.py', 'User.check'); 'app/x.py' -> ('app/x.py', None)."""
    path, sep, symbol = entry.partition("::")
    return path.strip(), (symbol.strip() or None) if sep else None


def load_questions(path: str | Path) -> list[Question]:
    """Load and validate a question file. Raises ValueError naming the problem and question id."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read the question file {path}: {exc}") from exc
    raw = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not raw:
        raise ValueError("The question file must be an object with a non-empty 'questions' list.")
    questions: list[Question] = []
    seen: set[str] = set()
    for position, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Question #{position} is not an object.")
        qid = str(item.get("id", "")).strip()
        label = qid or f"#{position}"
        if not qid or qid in seen:
            raise ValueError(f"Question {label}: the id is missing or duplicated.")
        seen.add(qid)
        text = str(item.get("question", "")).strip()
        if not text:
            raise ValueError(f"Question {qid}: the question text is empty.")
        category = str(item.get("category", ""))
        if category not in CATEGORIES:
            raise ValueError(f"Question {qid}: category must be one of {', '.join(CATEGORIES)}.")
        answerable = item.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"Question {qid}: 'answerable' must be true or false.")
        gold, terms = item.get("gold", []), item.get("must_include", [])
        if not isinstance(gold, list) or not all(isinstance(g, str) and g.strip() for g in gold):
            raise ValueError(f"Question {qid}: 'gold' must be a list of 'path' or 'path::symbol' strings.")
        if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
            raise ValueError(f"Question {qid}: 'must_include' must be a list of strings.")
        if answerable and not gold:
            raise ValueError(f"Question {qid}: an answerable question needs at least one gold location.")
        if not answerable and gold:
            raise ValueError(f"Question {qid}: an unanswerable question must not list gold locations.")
        questions.append(Question(qid, category, text, tuple(g.strip() for g in gold), tuple(terms), answerable))
    return questions


def validate_against_index(questions: list[Question], index: RepoIndex) -> list[str]:
    """Problems that make a gold location impossible to find in this index (typos, renamed files)."""
    paths = {c.path for c in index.chunks}
    symbols = {(c.path, c.symbol) for c in index.chunks if c.symbol}
    problems: list[str] = []
    for q in questions:
        for entry in q.gold:
            path, symbol = parse_gold(entry)
            if path not in paths:
                problems.append(f"{q.id}: file {path!r} is not in the index")
            elif symbol is not None and (path, symbol) not in symbols:
                problems.append(f"{q.id}: no function or class {symbol!r} in {path}")
    return problems

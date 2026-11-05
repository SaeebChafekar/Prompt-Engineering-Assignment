"""Run the questions against the system under different conditions and record the results.

A condition is a search mode, optionally with the graph: "keyword", "vector", "hybrid",
"hybrid+graph". Comparing "hybrid" with "hybrid+graph" isolates what the knowledge graph adds.
Results are appended to JSONL files one row at a time, so a long run on a slow machine can be
interrupted and resumed without losing finished work.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Iterable

from common.errors import ChatbotError
from repo_qa.index.index import RepoIndex
from repo_qa.index.search import MODES
from repo_qa.rag.answer import Answerer
from repo_qa.rag.retrieve import Retriever

from .dataset import Question
from .metrics import answer_scores, retrieval_scores


def parse_condition(name: str) -> tuple[str, bool]:
    """'hybrid+graph' -> ('hybrid', True); 'keyword' -> ('keyword', False)."""
    mode, plus, flag = name.strip().partition("+")
    if mode not in MODES or (plus and flag != "graph"):
        raise ValueError(f"Unknown condition {name!r}. Use a mode ({', '.join(MODES)}), optionally with +graph.")
    return mode, bool(plus)


def default_conditions(index: RepoIndex, for_answers: bool = False) -> list[str]:
    if index.has_vectors:
        return ["hybrid", "hybrid+graph"] if for_answers else ["keyword", "vector", "hybrid", "hybrid+graph"]
    return ["keyword", "keyword+graph"]


@dataclass
class RetrievalRow:
    question_id: str
    category: str
    condition: str
    recall: float
    file_recall: float
    first_rank: int | None
    found: list[str]
    missing: list[str]
    n_items: int
    tokens: int
    dropped: int


@dataclass
class AnswerRow:
    question_id: str
    category: str
    condition: str
    answerable: bool
    answer: str
    cited: list[str]
    invalid_citations: list[str]
    refused: bool
    gold_cited: float
    terms: float
    n_items: int
    tokens: int
    seconds: float
    error: str | None = None
    gold: list[str] = field(default_factory=list)
    question: str = ""


def row_from_dict(cls, data: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def append_jsonl(path: Path, row) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path, cls) -> list:
    if not Path(path).exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(row_from_dict(cls, json.loads(line)))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"{path}, line {number}: unreadable result row ({exc})") from exc
    return rows


def run_retrieval(
    retriever: Retriever,
    questions: Iterable[Question],
    conditions: list[str],
    *,
    k: int | None = None,
    on_row: Callable[[RetrievalRow], None] | None = None,
) -> list[RetrievalRow]:
    """Retrieval-only evaluation of every answerable question. Needs no chat model."""
    parsed = [(c, *parse_condition(c)) for c in conditions]
    rows: list[RetrievalRow] = []
    for q in questions:
        if not q.answerable:
            continue
        for name, mode, use_graph in parsed:
            result = retriever.retrieve(q.question, use_graph=use_graph, mode=mode, k=k)
            s = retrieval_scores(result.items, q.gold)
            row = RetrievalRow(q.id, q.category, name, s["recall"], s["file_recall"], s["first_rank"],
                               s["found"], s["missing"], len(result.items), result.tokens, result.dropped)
            rows.append(row)
            if on_row:
                on_row(row)
    return rows


def run_answers(
    answerer: Answerer,
    questions: Iterable[Question],
    conditions: list[str],
    *,
    skip: set[tuple[str, str]] = frozenset(),
    limit: int | None = None,
    on_row: Callable[[AnswerRow], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[AnswerRow]:
    """Ask every question under every condition (no conversation history). A failed question is
    recorded with its error and the run continues; (question id, condition) pairs in `skip` are not repeated."""
    parsed = [(c, *parse_condition(c)) for c in conditions]
    rows: list[AnswerRow] = []
    selected = list(questions)[:limit] if limit else list(questions)
    for q in selected:
        for name, mode, use_graph in parsed:
            if (q.id, name) in skip:
                continue
            started = clock()
            try:
                answer = answerer.ask(q.question, (), use_graph=use_graph, mode=mode)
            except (ChatbotError, ValueError) as exc:
                row = AnswerRow(q.id, q.category, name, q.answerable, "", [], [], False, 0.0, 0.0, 0, 0,
                                clock() - started, f"{type(exc).__name__}: {exc}", list(q.gold), q.question)
            else:
                s = answer_scores(q, answer)
                row = AnswerRow(q.id, q.category, name, q.answerable, answer.text, answer.cited,
                                answer.invalid_citations, s["refused"], s["gold_cited"], s["terms"],
                                len(answer.retrieval.items), answer.retrieval.tokens, clock() - started,
                                None, list(q.gold), q.question)
            rows.append(row)
            if on_row:
                on_row(row)
    return rows

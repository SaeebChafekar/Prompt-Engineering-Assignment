"""Scoring: did retrieval find the gold code, and did the answer use it correctly?"""

from __future__ import annotations

from typing import Sequence

from repo_qa.ingest.models import Chunk
from repo_qa.rag.answer import Answer
from repo_qa.rag.retrieve import ContextItem

from .dataset import Question, parse_gold


def chunk_matches(chunk: Chunk, gold: str) -> bool:
    """True if the chunk is (part of) the gold location: same file and, if named, same symbol."""
    path, symbol = parse_gold(gold)
    return chunk.path == path and (symbol is None or chunk.symbol == symbol)


def retrieval_scores(items: Sequence[ContextItem], gold: Sequence[str]) -> dict:
    """How much of the gold evidence appears among the retrieved chunks.

    recall ........ share of gold locations found (symbol-level when the gold names a symbol)
    file_recall ... share of gold FILES found at all (a more lenient view)
    first_rank .... position of the first retrieved chunk that is gold (None if none), for MRR
    """
    found = [g for g in gold if any(chunk_matches(it.chunk, g) for it in items)]
    gold_files = {parse_gold(g)[0] for g in gold}
    files_found = {p for p in gold_files if any(it.chunk.path == p for it in items)}
    first_rank = next((i for i, it in enumerate(items, 1) if any(chunk_matches(it.chunk, g) for g in gold)), None)
    return {
        "recall": len(found) / len(gold) if gold else 0.0,
        "file_recall": len(files_found) / len(gold_files) if gold_files else 0.0,
        "first_rank": first_rank,
        "found": found,
        "missing": [g for g in gold if g not in found],
    }


def answer_scores(question: Question, answer: Answer) -> dict:
    """Automatic checks on an answer. These are signals, not a verdict: grading by hand is still needed."""
    text = answer.text.lower()
    cited_paths = {c.rsplit(":", 1)[0] for c in answer.cited}
    gold_files = {parse_gold(g)[0] for g in question.gold}
    gold_cited = (
        sum(1 for p in gold_files if any(c == p or c.endswith("/" + p) or p.endswith("/" + c) for c in cited_paths))
        / len(gold_files)
        if gold_files else 0.0
    )
    terms = (
        sum(1 for t in question.must_include if t.lower() in text) / len(question.must_include)
        if question.must_include else 1.0
    )
    return {
        "refused": answer.not_found,
        "n_cited": len(answer.cited),
        "n_invalid": len(answer.invalid_citations),
        "gold_cited": gold_cited,
        "terms": terms,
    }

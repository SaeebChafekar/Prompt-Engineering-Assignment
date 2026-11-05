"""Turn raw results into tables for the report: Markdown, plus a CSV for grading answers by hand."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .runner import AnswerRow, RetrievalRow, read_jsonl

GRADE_COLUMN = "human_score (0 wrong, 1 partly right, 2 correct)"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _by_condition(rows: Sequence) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        grouped[row.condition].append(row)
    return dict(grouped)


def retrieval_report(rows: Sequence[RetrievalRow]) -> str:
    if not rows:
        return "_No retrieval results._"
    grouped = _by_condition(rows)
    overview = []
    for cond, group in grouped.items():
        mrr = _mean([1 / r.first_rank if r.first_rank else 0.0 for r in group])
        overview.append([cond, str(len(group)), _pct(_mean([r.recall for r in group])),
                         _pct(_mean([r.file_recall for r in group])),
                         _pct(sum(1 for r in group if r.recall == 1.0) / len(group)),
                         f"{mrr:.2f}", f"{_mean([r.n_items for r in group]):.1f}",
                         f"{_mean([r.tokens for r in group]):.0f}"])
    out = ["### Retrieval: did the right code reach the model?", "",
           _table(["Condition", "Questions", "Gold found (symbol)", "Gold files found", "All gold found",
                   "MRR", "Avg chunks", "Avg tokens"], overview)]

    categories = sorted({r.category for r in rows})
    by_cat = []
    for cat in categories:
        by_cat.append([cat] + [_pct(_mean([r.recall for r in g if r.category == cat])) for g in grouped.values()])
    out += ["", "**Gold found by question type**", "", _table(["Type"] + list(grouped), by_cat)]
    out += ["", _graph_effect(grouped)]
    return "\n".join(out)


def _graph_effect(grouped: dict[str, list[RetrievalRow]]) -> str:
    sections = []
    for cond in grouped:
        if cond.endswith("+graph") and cond[: -len("+graph")] in grouped:
            base = {r.question_id: r for r in grouped[cond[: -len("+graph")]]}
            better, worse, same = [], [], 0
            for r in grouped[cond]:
                b = base.get(r.question_id)
                if b is None:
                    continue
                if r.recall > b.recall:
                    better.append(f"{r.question_id} ({_pct(b.recall)} to {_pct(r.recall)})")
                elif r.recall < b.recall:
                    worse.append(f"{r.question_id} ({_pct(b.recall)} to {_pct(r.recall)})")
                else:
                    same += 1
            sections.append(
                f"**What the graph changed ({cond} vs {cond[:-len('+graph')]})**: improved {len(better)}, "
                f"unchanged {same}, worse {len(worse)}.\n"
                f"- Improved: {', '.join(better) or 'none'}\n- Worse: {', '.join(worse) or 'none'}"
            )
    return "\n\n".join(sections)


def _valid_citations(row: AnswerRow) -> bool:
    return row.error is None and bool(row.cited) and not row.invalid_citations


def answer_report(rows: Sequence[AnswerRow]) -> str:
    if not rows:
        return "_No answer results._"
    table = []
    for cond, group in _by_condition(rows).items():
        answerable = [r for r in group if r.answerable and r.error is None]
        unanswerable = [r for r in group if not r.answerable and r.error is None]
        table.append([
            cond, str(len(group)), str(sum(1 for r in group if r.error)),
            _pct(_mean([1.0 if _valid_citations(r) else 0.0 for r in answerable])),
            _pct(_mean([r.gold_cited for r in answerable])),
            _pct(_mean([r.terms for r in answerable])),
            _pct(_mean([1.0 if r.refused else 0.0 for r in answerable])),
            _pct(_mean([1.0 if r.refused else 0.0 for r in unanswerable])) if unanswerable else "n/a",
            f"{_mean([r.seconds for r in group if r.error is None]):.1f}s",
        ])
    return "\n".join(["### Answers: automatic checks (not a substitute for grading by hand)", "",
                      _table(["Condition", "Answers", "Errors", "Citations all valid", "Gold files cited",
                              "Key terms present", "Wrongly refused", "Correctly refused", "Avg time"], table)])


def write_grading_csv(rows: Sequence[AnswerRow], path: Path) -> None:
    """One row per answer, with a blank column to fill in by hand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "category", "condition", "question", "expected_code", "answer", "citations",
                         "invalid_citations", "gold_files_cited", "key_terms", "refused", GRADE_COLUMN, "notes"])
        for r in rows:
            writer.writerow([r.question_id, r.category, r.condition, r.question, "; ".join(r.gold),
                             r.answer or f"[ERROR] {r.error}", "; ".join(r.cited), "; ".join(r.invalid_citations),
                             f"{r.gold_cited:.2f}", f"{r.terms:.2f}", r.refused, "", ""])


def human_scores(path: Path) -> dict[str, dict]:
    """Summarize the hand-graded CSV: mean score and share fully correct, per condition."""
    scores: dict[str, list[int]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get(GRADE_COLUMN) or "").strip()
            if raw in ("0", "1", "2"):
                scores[row["condition"]].append(int(raw))
            elif raw:
                raise ValueError(f"Question {row.get('id')}: the grade must be 0, 1 or 2, got {raw!r}.")
    return {c: {"graded": len(v), "mean": _mean(v), "correct": sum(1 for s in v if s == 2) / len(v)}
            for c, v in scores.items() if v}


def human_report(path: Path) -> str:
    summary = human_scores(path)
    if not summary:
        return "_No hand grades filled in yet._"
    rows = [[c, str(s["graded"]), f"{s['mean']:.2f} / 2", _pct(s["correct"])] for c, s in summary.items()]
    return "\n".join(["### Answers graded by hand", "", _table(["Condition", "Graded", "Mean score", "Fully correct"], rows)])


METHOD_NOTES = """### How to read these numbers
- **Gold found (symbol)**: share of the expected functions/classes that appear among the chunks given to the model.
- **Gold files found**: the same, counting a file as found if any chunk of it was retrieved (more lenient).
- **MRR**: mean of 1/rank of the first correct chunk (1.0 = the first chunk is always correct).
- **Avg tokens**: size of the context sent to the model; the graph adds related code, so this shows its cost.
- **Citations all valid**: the answer cites at least one location and every cited file and line range was in the context.
- **Wrongly / correctly refused**: how often the answer said the code does not contain it, for answerable and for
  unanswerable questions. Unanswerable questions ask about things verified to be absent from the repository."""


def build_report(out_dir: Path) -> str:
    out_dir = Path(out_dir)
    parts = ["# Evaluation report", "", retrieval_report(read_jsonl(out_dir / "retrieval.jsonl", RetrievalRow)), "",
             answer_report(read_jsonl(out_dir / "answers.jsonl", AnswerRow))]
    graded = out_dir / "answers_for_grading.csv"
    if graded.exists():
        parts += ["", human_report(graded)]
    parts += ["", METHOD_NOTES]
    return "\n".join(parts) + "\n"

"""Evaluation commands for the report.

    python -m repo_qa.eval validate  <index> [--questions FILE]       check the gold locations exist
    python -m repo_qa.eval retrieval <index> [--conditions a,b] [--k 8]   did the right code reach the model?
    python -m repo_qa.eval answers   <index> [--conditions a,b] [--limit N] [--resume]   ask every question
    python -m repo_qa.eval report    <results-dir>                    tables for the report (report.md)
    python -m repo_qa.eval sample-calls <index> <calls.csv> --n 40    sample call edges to label by hand
    python -m repo_qa.eval calls-precision <labelled.csv>             graph accuracy from your labels

<index> is a folder made by `python -m repo_qa.index build`. Conditions are search modes
(keyword, vector, hybrid), optionally with +graph, e.g. hybrid and hybrid+graph.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

from common.backends import ChatBackend, Embedder, create_chat_client, create_embedder
from common.config import Config
from common.errors import ChatbotError
from repo_qa.index.index import RepoIndex
from repo_qa.index.search import HybridSearcher
from repo_qa.rag.retrieve import Retriever
from repo_qa.service import RepoChat

from .calls import calls_precision, precision_text, read_calls_csv, sample_calls, write_sample
from .dataset import load_questions, validate_against_index
from .report import answer_report, build_report, retrieval_report, write_grading_csv
from .runner import (
    AnswerRow, append_jsonl, default_conditions, parse_condition, read_jsonl, run_answers, run_retrieval,
)

DEFAULT_QUESTIONS = Path(__file__).resolve().parents[2] / "eval" / "microblog_questions.json"


def _conditions(text: str | None, index: RepoIndex, for_answers: bool) -> list[str]:
    names = [c.strip() for c in text.split(",") if c.strip()] if text else default_conditions(index, for_answers)
    for name in names:
        mode, _ = parse_condition(name)
        if mode != "keyword" and not index.has_vectors:
            raise ValueError(f"Condition {name!r} needs embeddings, but this index is keyword-only. "
                             "Rebuild it without --keyword-only, or use conditions like keyword and keyword+graph.")
    return names


def main(
    argv: list[str] | None = None,
    *,
    chat_factory: Callable[[Config], ChatBackend] = create_chat_client,
    embedder_factory: Callable[[Config], Embedder] = create_embedder,
    write: Callable[[str], None] = print,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m repo_qa.eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_index(p: argparse.ArgumentParser) -> None:
        p.add_argument("index_dir", type=Path)
        p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
        p.add_argument("--out", type=Path, help="results folder (default: <index>/eval)")

    p = sub.add_parser("validate", help="check that every gold location exists in the index")
    with_index(p)
    p = sub.add_parser("retrieval", help="retrieval-only evaluation (no chat model needed)")
    with_index(p)
    p.add_argument("--conditions")
    p.add_argument("--k", type=int)
    p = sub.add_parser("answers", help="ask every question and score the answers")
    with_index(p)
    p.add_argument("--conditions")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--resume", action="store_true", help="continue a previous run, skipping finished questions")
    p = sub.add_parser("report", help="build report.md from saved results")
    p.add_argument("results_dir", type=Path)
    p = sub.add_parser("sample-calls", help="sample call edges for hand-labelling")
    p.add_argument("index_dir", type=Path)
    p.add_argument("calls_csv", type=Path)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--include-external", action="store_true", help="also sample calls into libraries")
    p.add_argument("--out", type=Path, default=Path("call_sample.csv"))
    p = sub.add_parser("calls-precision", help="graph accuracy from a hand-labelled sample")
    p.add_argument("labelled_csv", type=Path)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    resources: list = []
    try:
        cfg = Config.from_env()
        if args.command == "report":
            text = build_report(args.results_dir)
            (args.results_dir / "report.md").write_text(text, encoding="utf-8")
            write(text)
            write(f"Saved to {args.results_dir / 'report.md'}")
        elif args.command == "calls-precision":
            write(precision_text(calls_precision(args.labelled_csv)))
        elif args.command == "sample-calls":
            index = RepoIndex.load(args.index_dir)
            rows = sample_calls(index, read_calls_csv(args.calls_csv), args.n, args.seed,
                                include_external=args.include_external)
            write_sample(rows, args.out)
            write(f"Wrote {len(rows)} sampled call edges to {args.out}. Fill the last column with y or n "
                  f"(is this call really made to that function?), then run: python -m repo_qa.eval calls-precision {args.out}")
        else:
            index = RepoIndex.load(args.index_dir)
            questions = load_questions(args.questions)
            out_dir = args.out or (args.index_dir / "eval")
            if args.command == "validate":
                problems = validate_against_index(questions, index)
                for problem in problems:
                    write(f"PROBLEM: {problem}")
                write(f"{len(questions)} questions checked; {len(problems)} problem(s).")
                return 1 if problems else 0
            if args.command == "retrieval":
                conditions = _conditions(args.conditions, index, for_answers=False)
                embedder = embedder_factory(cfg) if index.has_vectors and any(parse_condition(c)[0] != "keyword" for c in conditions) else None
                if embedder is not None:
                    resources.append(embedder)
                    embedder.health_check()
                retriever = Retriever(index, HybridSearcher(index, embedder, cfg), cfg)
                (out_dir / "retrieval.jsonl").unlink(missing_ok=True)
                rows = run_retrieval(retriever, questions, conditions, k=args.k,
                                     on_row=lambda r: append_jsonl(out_dir / "retrieval.jsonl", r))
                write(retrieval_report(rows))
                write(f"\nSaved to {out_dir / 'retrieval.jsonl'}")
            else:  # answers
                conditions = _conditions(args.conditions, index, for_answers=True)
                chat = RepoChat.open(args.index_dir, cfg, chat_factory=chat_factory, embedder_factory=embedder_factory)
                resources.append(chat)
                path = out_dir / "answers.jsonl"
                previous = read_jsonl(path, AnswerRow) if args.resume else []
                if not args.resume:
                    path.unlink(missing_ok=True)
                done = {(r.question_id, r.condition) for r in previous if r.error is None}
                kept = [r for r in previous if r.error is None]
                if args.resume:  # drop failed rows so they are retried, then rewrite the file
                    path.unlink(missing_ok=True)
                    for r in kept:
                        append_jsonl(path, r)

                def on_row(row: AnswerRow) -> None:
                    append_jsonl(path, row)
                    status = f"ERROR {row.error}" if row.error else f"{row.seconds:.0f}s, {len(row.cited)} citations"
                    write(f"  {row.question_id} [{row.condition}] {status}")

                new = run_answers(chat.answerer, questions, conditions, skip=done, limit=args.limit, on_row=on_row)
                rows = kept + new
                write_grading_csv(rows, out_dir / "answers_for_grading.csv")
                write("\n" + answer_report(rows))
                write(f"\nSaved to {path}. To grade by hand, fill in {out_dir / 'answers_for_grading.csv'} then run: "
                      f"python -m repo_qa.eval report {out_dir}")
    except (ChatbotError, ValueError) as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Finished questions are saved; add --resume to continue.", file=sys.stderr)
        return 130
    finally:
        for resource in resources:
            resource.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

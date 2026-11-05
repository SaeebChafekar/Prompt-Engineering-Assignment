"""Command line front end for the knowledge graph.

    python -m repo_qa.graph https://github.com/owner/repo
    python -m repo_qa.graph path/to/folder --callers verify_token
    python -m repo_qa.graph path/to/folder --callees login --min-confidence likely
    python -m repo_qa.graph path/to/folder --imports app/auth/routes.py
    python -m repo_qa.graph path/to/folder --out data/graphs/repo --labeling-csv calls.csv

Builds the graph (ingesting the repository first) and prints its size and how well calls
were resolved. The query options show what the graph knows about a name, which is the
quickest way to judge whether it is accurate.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.config import Config
from common.errors import ChatbotError
from repo_qa.ingest.pipeline import ingest_repo

from .build import BuildReport, build_graph, write_call_labeling_csv
from .model import KnowledgeGraph
from .schema import CONFIDENCE_RANK, EDGE_IMPORTS


def summary_text(graph: KnowledgeGraph, report: BuildReport) -> str:
    counts = graph.counts()
    calls = report.calls
    lines = [
        "Nodes : " + ", ".join(f"{n} {t}" for t, n in counts["nodes"].items()),
        "Edges : " + ", ".join(f"{n} {t}" for t, n in counts["edges"].items()),
        "",
        f"Calls found by the parser : {calls.total}",
        f"  resolved to repo code   : {calls.resolved}  "
        f"(certain {calls.certain}, likely {calls.likely}, guess {calls.guess})",
        f"  library calls (USES)    : {calls.external}",
        f"  built-ins / ignored     : {calls.builtin} / {calls.skipped}",
        f"  not resolved            : {calls.unresolved} unknown names, {calls.ambiguous} ambiguous, "
        f"{calls.foreign} probably library methods",
        "  (accuracy is not measured automatically: label a sample with --labeling-csv)",
        f"Imports : {report.imports_internal} inside the repo, {report.imports_external} libraries, "
        f"{report.imports_unresolved} unresolved",
        f"Doc links : {report.doc_links}",
    ]
    return "\n".join(lines)


def _print_edges(graph: KnowledgeGraph, title: str, rows: list) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (none)")
    for other, data in rows:
        how = f"{data['confidence']}, {data['via']}" if data.get("via") else data["confidence"]
        names = ", ".join(data.get("names", [])[:3])
        print(f"  {graph.label(other)}   [{how}]   as: {names}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m repo_qa.graph", description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="public GitHub URL or a local folder")
    parser.add_argument("--refresh", action="store_true", help="re-download even if already cloned")
    parser.add_argument("--out", type=Path, help="folder to save graph.json and stats.json")
    parser.add_argument("--labeling-csv", type=Path, help="write every call decision to a CSV for hand-labelling")
    parser.add_argument("--find", metavar="NAME", help="list nodes matching a name")
    parser.add_argument("--callers", metavar="NAME", help="who calls this function or class?")
    parser.add_argument("--callees", metavar="NAME", help="what does this function call?")
    parser.add_argument("--imports", metavar="FILE", help="what does this file import, and who imports it?")
    parser.add_argument("--min-confidence", choices=sorted(CONFIDENCE_RANK, key=CONFIDENCE_RANK.get), default="guess")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        cfg = Config.from_env()
        result = ingest_repo(args.source, cfg, refresh=args.refresh)
        graph, report = build_graph(result)
    except ChatbotError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

    print(f"Repository : {result.repo_name}  ({len(result.files)} files, {len(result.chunks)} chunks)")
    print(summary_text(graph, report))

    def lookup(name: str, kinds: tuple[str, ...] = ("Function", "Class")) -> list[str]:
        matches = [n for k in kinds for n in graph.find(name, k)]
        if not matches:
            print(f"\nNo node named {name!r}. Try --find with a shorter name.")
        return matches

    if args.find:
        print(f"\nNodes matching {args.find!r}:")
        for node_id in graph.find(args.find):
            print(f"  [{graph.node(node_id)['type']}] {graph.label(node_id)}")
    if args.callers:
        for node_id in lookup(args.callers):
            _print_edges(graph, f"Callers of {graph.label(node_id)}", graph.callers(node_id, min_confidence=args.min_confidence))
    if args.callees:
        for node_id in lookup(args.callees):
            _print_edges(graph, f"{graph.label(node_id)} calls", graph.callees(node_id, min_confidence=args.min_confidence))
    if args.imports:
        for node_id in graph.find(args.imports, "File"):
            _print_edges(graph, f"{graph.label(node_id)} imports",
                         graph.edges_of(node_id, incoming=False, edge_types=[EDGE_IMPORTS]))
            _print_edges(graph, f"{graph.label(node_id)} is imported by",
                         graph.edges_of(node_id, incoming=True, edge_types=[EDGE_IMPORTS]))

    if args.out:
        graph.save(args.out / "graph.json")
        (args.out / "stats.txt").write_text(summary_text(graph, report) + "\n", encoding="utf-8")
        print(f"\nSaved graph to {args.out}/graph.json")
    if args.labeling_csv:
        write_call_labeling_csv(report, args.labeling_csv)
        print(f"Wrote {len(report.call_rows)} call decisions to {args.labeling_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line front end: build an index of a repository, then search it.

    python -m repo_qa.index build https://github.com/owner/repo
    python -m repo_qa.index build path/to/folder --keyword-only      # no Ollama needed
    python -m repo_qa.index search data/index/owner__repo "where is authentication handled?"
    python -m repo_qa.index search data/index/owner__repo "verify_password" --mode keyword --show-text

`build` ingests the repository, builds the knowledge graph, embeds every chunk with the
local embedding model (slow part: it prints progress) and saves everything to a folder.
Re-running build on the same folder re-embeds only chunks that changed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.backends import create_embedder
from common.config import Config
from common.errors import ChatbotError
from repo_qa.service import index_repository

from .index import RepoIndex
from .search import MODES, HybridSearcher


def cmd_build(args: argparse.Namespace, cfg: Config) -> int:
    def progress(done: int, total: int) -> None:
        print(f"\r  embedded {done}/{total}", end="", file=sys.stderr, flush=True)

    try:
        report = index_repository(
            args.source, cfg, refresh=args.refresh, keyword_only=args.keyword_only, out_dir=args.out,
            embedder_factory=create_embedder, progress=progress, stage=lambda message: print(message),
        )
    finally:
        print(file=sys.stderr)
    print("\n" + report.summary)
    nodes, edges = sum(report.graph_counts["nodes"].values()), sum(report.graph_counts["edges"].values())
    print(f"\nGraph: {nodes} nodes, {edges} edges")
    print(f"\nIndex saved to {report.path}")
    if report.has_vectors:
        print(f"  vectors: {report.vectors_embedded} embedded now, {report.vectors_reused} reused")
    else:
        print("  keyword-only index (no vectors). Rebuild without --keyword-only for meaning-based search.")
    print(f'Try: python -m repo_qa.index search "{report.path}" "your question"')
    return 0


def cmd_search(args: argparse.Namespace, cfg: Config) -> int:
    index = RepoIndex.load(args.index_dir)
    embedder = None
    if args.mode != "keyword" and index.has_vectors:
        embedder = create_embedder(cfg)
    try:
        searcher = HybridSearcher(index, embedder, cfg)
        hits = searcher.search(args.query, k=args.k, mode=args.mode)
    finally:
        if embedder is not None:
            embedder.close()
    print(f"Index: {index.repo_name}  ({len(index.chunks)} chunks, vectors: {'yes' if index.has_vectors else 'no'})")
    if searcher.note:
        print(f"Note: {searcher.note}")
    if not hits:
        print("\nNo matching chunks.")
    for rank, hit in enumerate(hits, 1):
        c = hit.chunk
        signals = ", ".join(
            [f"vector #{hit.vector_rank}" if hit.vector_rank else "", f"keyword #{hit.keyword_rank}" if hit.keyword_rank else "",
             *hit.signals]
        ).strip(", ").replace(",  ", ", ")
        print(f"\n{rank}. {c.citation}  [{c.kind}] {c.name}   score {hit.score:.4f}")
        print(f"   {signals}")
        if args.show_text:
            body = c.text if len(c.text) <= 500 else c.text[:500] + " ..."
            print("   | " + body.replace("\n", "\n   | "))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m repo_qa.index", description=__doc__.split("\n\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="ingest, embed and save an index")
    build.add_argument("source", help="public GitHub URL or a local folder")
    build.add_argument("--out", type=Path, help="index folder (default: data/index/<repo>)")
    build.add_argument("--keyword-only", action="store_true", help="skip embeddings (no Ollama needed)")
    build.add_argument("--refresh", action="store_true", help="re-download the repository first")

    search = sub.add_parser("search", help="search a saved index")
    search.add_argument("index_dir", type=Path)
    search.add_argument("query")
    search.add_argument("-k", type=int, default=None, help="number of results (default 8)")
    search.add_argument("--mode", choices=MODES, default="hybrid")
    search.add_argument("--show-text", action="store_true", help="print the start of each chunk")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        cfg = Config.from_env()
        return cmd_build(args, cfg) if args.command == "build" else cmd_search(args, cfg)
    except (ChatbotError, ValueError) as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

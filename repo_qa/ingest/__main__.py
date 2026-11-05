"""Command line front end for ingestion.

    python -m repo_qa.ingest https://github.com/owner/repo
    python -m repo_qa.ingest path/to/local/folder --show-chunks 5
    python -m repo_qa.ingest owner-repo-url --out data/debug/myrepo   # dump to inspect

Prints a summary of what was kept, skipped and parsed. With --out it also writes
summary.json, files.json and chunks.jsonl so you can open the chunks and check that
they look sensible before anything is embedded.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from common.config import Config
from common.errors import ChatbotError

from .models import IngestResult
from .pipeline import ingest_repo


def write_dump(result: IngestResult, out_dir: Path) -> None:
    """Write human-readable files describing the ingestion result."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "repo": result.repo_name,
        "source": result.source,
        "files": len(result.files),
        "files_by_language": result.files_by_language(),
        "chunks": len(result.chunks),
        "skipped": result.skipped,
        "warnings": result.warnings,
        "elapsed_s": round(result.elapsed_s, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    files = [
        {"path": f.path, "language": f.language, "kind": f.kind, "lines": f.line_count, "is_test": f.is_test}
        for f in result.files
    ]
    (out_dir / "files.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as handle:
        for chunk in result.chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    (out_dir / "parsed_python.json").write_text(
        json.dumps([p.to_dict() for p in result.parsed], indent=1), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m repo_qa.ingest", description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="public GitHub URL or a local folder")
    parser.add_argument("--refresh", action="store_true", help="re-download even if already cloned")
    parser.add_argument("--out", type=Path, help="folder to write summary.json, files.json, chunks.jsonl")
    parser.add_argument("--show-chunks", type=int, default=0, metavar="N", help="print the first N chunks")
    parser.add_argument("-v", "--verbose", action="store_true", help="show progress messages")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        cfg = Config.from_env()
        result = ingest_repo(args.source, cfg, refresh=args.refresh)
    except ChatbotError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

    print(result.summary())
    for chunk in result.chunks[: max(args.show_chunks, 0)]:
        print(f"\n--- {chunk.kind} | {chunk.citation} | {chunk.name} ---")
        print(chunk.embed_text[:600])
    if args.out:
        write_dump(result, args.out)
        print(f"\nWrote details to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

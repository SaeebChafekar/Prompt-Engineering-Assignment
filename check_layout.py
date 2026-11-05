"""Check that the project folders are laid out correctly (and repair a flat extraction).

Run from the project root, the folder that contains check_setup.py:

    python check_layout.py          # report what is missing or in the wrong place
    python check_layout.py --fix    # move files that were extracted flat into their folders

If your unzip tool ignored the folders and dumped every file next to this one, --fix moves
each file back where it belongs. A few files share a name in several folders (__init__.py,
__main__.py) and cannot be told apart once flattened: for those, extract the archive again
with `unzip` or `tar -xzf` (see the message this script prints).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MANIFEST = (
    ".streamlit/config.toml",
    "README.md",
    "app.py",
    "check_layout.py",
    "check_setup.py",
    "code_review/__init__.py",
    "code_review/__main__.py",
    "code_review/prompts.py",
    "code_review/render.py",
    "code_review/reviewer.py",
    "code_review/schema.py",
    "common/__init__.py",
    "common/backends.py",
    "common/config.py",
    "common/errors.py",
    "common/gemini_client.py",
    "common/limits.py",
    "common/llm_client.py",
    "debug_gemini.py",
    "docs/CLOUD_TODO.md",
    "docs/EVALUATION.md",
    "eval/microblog_call_sample.csv",
    "eval/microblog_questions.json",
    "repo_qa/__init__.py",
    "repo_qa/eval/__init__.py",
    "repo_qa/eval/__main__.py",
    "repo_qa/eval/calls.py",
    "repo_qa/eval/dataset.py",
    "repo_qa/eval/metrics.py",
    "repo_qa/eval/report.py",
    "repo_qa/eval/runner.py",
    "repo_qa/graph/__init__.py",
    "repo_qa/graph/__main__.py",
    "repo_qa/graph/build.py",
    "repo_qa/graph/model.py",
    "repo_qa/graph/resolve.py",
    "repo_qa/graph/schema.py",
    "repo_qa/index/__init__.py",
    "repo_qa/index/__main__.py",
    "repo_qa/index/bm25.py",
    "repo_qa/index/index.py",
    "repo_qa/index/search.py",
    "repo_qa/index/store.py",
    "repo_qa/ingest/__init__.py",
    "repo_qa/ingest/__main__.py",
    "repo_qa/ingest/chunker.py",
    "repo_qa/ingest/clone.py",
    "repo_qa/ingest/filters.py",
    "repo_qa/ingest/models.py",
    "repo_qa/ingest/parse_python.py",
    "repo_qa/ingest/pipeline.py",
    "repo_qa/rag/__init__.py",
    "repo_qa/rag/__main__.py",
    "repo_qa/rag/answer.py",
    "repo_qa/rag/prompts.py",
    "repo_qa/rag/retrieve.py",
    "repo_qa/service.py",
    "requirements.txt",
    "tests/__init__.py",
    "tests/fakes.py",
    "tests/test_code_review.py",
    "tests/test_eval.py",
    "tests/test_gemini_client.py",
    "tests/test_graph_build.py",
    "tests/test_graph_model.py",
    "tests/test_index.py",
    "tests/test_ingest_chunker.py",
    "tests/test_ingest_parse.py",
    "tests/test_ingest_pipeline.py",
    "tests/test_layout.py",
    "tests/test_limits_and_backends.py",
    "tests/test_llm_client.py",
    "tests/test_rag.py",
    "tests/test_service.py",
    "tests/test_webapp.py",
    "webapp/__init__.py",
    "webapp/graph_canvas.py",
    "webapp/repo_page.py",
    "webapp/resources.py",
    "webapp/review_page.py",
    "webapp/sidebar.py",
    "webapp/theme.py",
)


SOURCE_DIRS = ("common", "repo_qa", "code_review", "webapp", "tests")


def unlisted_source_files(root: Path) -> list[str]:
    """Python files inside the project's own source folders that the manifest does not list.

    Only the project's source folders are scanned, so your own files (a downloaded zip, notes,
    a virtual environment, scratch scripts next to app.py) never count.
    """
    found: list[str] = []
    for folder in SOURCE_DIRS:
        base = root / folder
        if base.is_dir():
            for path in base.rglob("*.py"):
                rel = path.relative_to(root)
                if "__pycache__" not in rel.parts:
                    found.append(rel.as_posix())
    return sorted(rel for rel in found if rel not in MANIFEST)


def find_missing(root: Path) -> list[str]:
    """Manifest files that do not exist at their expected place under root."""
    return [rel for rel in MANIFEST if not (root / rel).is_file()]


def find_misplaced(root: Path, missing: list[str]) -> tuple[dict[str, Path], list[str]]:
    """Split missing files into (found flat in root and movable, names that cannot be recovered)."""
    by_name: dict[str, list[str]] = {}
    for rel in MANIFEST:
        by_name.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)
    movable: dict[str, Path] = {}
    unrecoverable: list[str] = []
    for rel in missing:
        name = rel.rsplit("/", 1)[-1]
        flat = root / name
        if "/" in rel and flat.is_file() and len(by_name[name]) == 1:
            movable[rel] = flat
        elif "/" in rel and len(by_name[name]) > 1:
            unrecoverable.append(rel)
    return movable, unrecoverable


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fix", action="store_true", help="move flat-extracted files into their folders")
    args = parser.parse_args(argv)
    root = root or Path.cwd()

    if not (root / "check_setup.py").is_file() and not (root / "check_layout.py").is_file():
        print("Run this from the project root (the folder that contains check_setup.py).")
        return 2

    missing = find_missing(root)
    if args.fix and missing:
        movable, _ = find_misplaced(root, missing)
        for rel, flat in movable.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            flat.replace(target)
            print(f"moved {flat.name} -> {rel}")
        missing = find_missing(root)

    print(f"{len(MANIFEST) - len(missing)}/{len(MANIFEST)} files in place")
    if not missing:
        print("Layout OK. Next: python -m unittest discover -s tests -t .")
        return 0

    movable, unrecoverable = find_misplaced(root, missing)
    for rel in missing:
        print(f"MISSING: {rel}")
    if movable:
        print(f"\n{len(movable)} of these exist loose in this folder. Run: python check_layout.py --fix")
    if unrecoverable:
        print(
            f"\n{len(unrecoverable)} file(s) share a name with others (such as __init__.py), so they cannot be\n"
            "recovered from a flat extraction. Extract the archive again with a tool that keeps folders:\n"
            "    unzip -o RAG-Prompt-Bot.zip -d ..            (run from inside the project folder)\n"
            "    tar -xzf RAG-Prompt-Bot.tar.gz -C ..          (same, from the .tar.gz)"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())

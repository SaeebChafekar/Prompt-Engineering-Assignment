"""The ingestion pipeline: repository in, files + parsed Python + chunks out.

Stages, each guarded by the overall time limit (ingest_timeout_s):

    1. resolve  a GitHub URL is cloned (shallow); a local folder is used as is
    2. discover walk the tree, applying the filters and the repository size limits
    3. read     load text, rejecting binary, empty, oversized and generated files
    4. parse    Python files are parsed into classes, functions, imports and calls
    5. chunk    every file is cut into chunks for embedding

Large or hostile repositories are stopped early with a clear error (RepoTooLargeError,
IngestError) instead of exhausting memory or time.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from common.config import Config
from common.errors import IngestError, RepoTooLargeError

from .chunker import chunk_file
from .clone import Runner, clone_repo, looks_like_github_url, parse_repo_url
from .filters import (
    FileClass,
    classify,
    is_test_path,
    looks_binary,
    looks_generated,
    should_skip_dir,
    size_limit,
    skip_file_reason,
)
from .models import IngestResult, ParsedFile, SourceFile
from .parse_python import parse_python

log = logging.getLogger(__name__)

_BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class _Candidate:
    rel_path: str
    abs_path: str
    file_class: FileClass
    size: int


def ingest_repo(
    source: str,
    config: Config | None = None,
    *,
    refresh: bool = False,
    runner: Runner = subprocess.run,
    clock: Callable[[], float] = time.monotonic,
) -> IngestResult:
    """Ingest a public GitHub URL or a local folder."""
    cfg = config if config is not None else Config.from_env()
    root, name = resolve_source(source, cfg, refresh=refresh, runner=runner)
    return ingest_directory(root, cfg, repo_name=name, source=source, clock=clock)


def resolve_source(
    source: str, cfg: Config, *, refresh: bool = False, runner: Runner = subprocess.run
) -> tuple[Path, str]:
    """Turn what the user typed into a folder on disk and a display name."""
    text = source.strip()
    if not text:
        raise IngestError("No repository given. Pass a GitHub URL or a folder path.")
    if os.path.isdir(text):
        folder = Path(text).resolve()
        return folder, folder.name
    if looks_like_github_url(text):
        ref = parse_repo_url(text)
        folder = clone_repo(
            ref,
            Path(cfg.data_dir) / "repos",
            timeout_s=cfg.clone_timeout_s,
            git=cfg.git_executable,
            refresh=refresh,
            runner=runner,
        )
        return folder.resolve(), f"{ref.owner}/{ref.repo}"
    raise IngestError(
        f"{source!r} is neither an existing folder nor a GitHub URL "
        "(expected something like https://github.com/owner/repo)."
    )


def ingest_directory(
    root: Path,
    config: Config | None = None,
    *,
    repo_name: str | None = None,
    source: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> IngestResult:
    """Run discover -> read -> parse -> chunk on a folder."""
    cfg = config if config is not None else Config.from_env()
    root = Path(root).resolve()
    if not root.is_dir():
        raise IngestError(f"Not a folder: {root}")

    started = clock()
    deadline = started + cfg.ingest_timeout_s

    def check_time(stage: str) -> None:
        if clock() > deadline:
            raise IngestError(
                f"Ingestion exceeded {cfg.ingest_timeout_s:.0f}s during '{stage}'. Use a smaller "
                "repository or raise CHATBOT_INGEST_TIMEOUT_S."
            )

    result = IngestResult(repo_name=repo_name or root.name, root=str(root), source=source or str(root))
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    candidates = _discover(root, cfg, skip, check_time)
    log.info("Discovered %d candidate files", len(candidates))

    for cand in candidates:
        check_time("reading files")
        loaded = _read_file(cand, cfg, skip)
        if loaded is not None:
            result.files.append(loaded)

    parsed_by_path: dict[str, ParsedFile] = {}
    for sf in result.files:
        check_time("parsing Python")
        if sf.language == "python":
            parsed = parse_python(sf.path, sf.text)
            parsed_by_path[sf.path] = parsed
            result.parsed.append(parsed)

    for sf in result.files:
        check_time("chunking")
        result.chunks.extend(chunk_file(sf, parsed_by_path.get(sf.path), cfg))

    result.skipped = dict(sorted(skipped.items()))
    broken = [p.path for p in result.parsed if p.error]
    if broken:
        shown = ", ".join(broken[:3]) + (" ..." if len(broken) > 3 else "")
        result.warnings.append(
            f"{len(broken)} Python file(s) could not be parsed and use plain chunking: {shown}"
        )
    if not result.files:
        result.warnings.append("No indexable files were found. Is this the right folder or repository?")
    result.elapsed_s = clock() - started
    return result


def _discover(
    root: Path, cfg: Config, skip: Callable[[str], None], check_time: Callable[[str], None]
) -> list[_Candidate]:
    """Walk the tree and return the files worth reading, enforcing the size limits."""
    found: list[_Candidate] = []
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        check_time("scanning folders")
        kept_dirs = []
        for d in sorted(dirnames):
            if should_skip_dir(d):
                skip("ignored_folder")
            elif os.path.islink(os.path.join(dirpath, d)):
                skip("symlink")
            else:
                kept_dirs.append(d)
        dirnames[:] = kept_dirs  # prune: os.walk will not enter the skipped folders

        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            reason = skip_file_reason(name)
            if reason:
                skip(reason)
                continue
            file_class = classify(name)
            if file_class is None:
                skip("unsupported_type")
                continue
            if os.path.islink(full):
                skip("symlink")
                continue
            rel = Path(full).relative_to(root).as_posix()
            if not cfg.include_tests and is_test_path(rel):
                skip("test_file")
                continue
            try:
                size = os.path.getsize(full)
            except OSError:
                skip("unreadable")
                continue
            if size == 0 and name != "__init__.py":  # an empty __init__.py still marks a package
                skip("empty")
                continue
            if size > size_limit(file_class, cfg.max_file_bytes):
                skip("too_large")
                continue
            found.append(_Candidate(rel, full, file_class, size))
            total_bytes += size
            if len(found) > cfg.max_repo_files:
                raise RepoTooLargeError(
                    f"The repository has more than {cfg.max_repo_files} indexable files. Pick a "
                    "smaller repository, or raise CHATBOT_MAX_REPO_FILES if your machine can cope."
                )
            if total_bytes > cfg.max_repo_bytes:
                raise RepoTooLargeError(
                    f"The indexable source exceeds {cfg.max_repo_bytes / 1_000_000:.0f} MB. Pick a "
                    "smaller repository, or raise CHATBOT_MAX_REPO_BYTES."
                )
    found.sort(key=lambda c: c.rel_path)
    return found


def _read_file(cand: _Candidate, cfg: Config, skip: Callable[[str], None]) -> SourceFile | None:
    """Load one file as text, or return None (and count why) if it should be skipped."""
    try:
        with open(cand.abs_path, "rb") as handle:
            raw = handle.read(cand.size + 1)
    except OSError:
        skip("unreadable")
        return None
    if looks_binary(raw[:_BINARY_SNIFF_BYTES]):
        skip("binary")
        return None
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text[1:]
    if not text.strip() and not cand.rel_path.endswith("__init__.py"):
        skip("empty")
        return None
    if cand.file_class.kind == "code" and looks_generated(text):
        skip("generated")
        return None
    return SourceFile(
        path=cand.rel_path,
        language=cand.file_class.language,
        kind=cand.file_class.kind,
        text=text,
        size=cand.size,
        is_test=is_test_path(cand.rel_path),
    )
